"""Offline checks for geometry-only substrate connectivity and masked node training."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

MODULE = Path(__file__).resolve().parents[1] / "03_htem_conductivity"
sys.path.insert(0, str(MODULE / 'lib'))
import htem_workshop as htem
import htem_graph as graph


def grid(width=3, height=3):
    return pd.DataFrame([(y * width + x, x * 4.0, y * 12.5)
                         for y in range(height) for x in range(width)], columns=["position", "x_mm", "y_mm"])


class GridTopologyTests(unittest.TestCase):
    def test_four_neighbors_not_diagonals(self):
        g = grid()
        edges = graph.grid_topology(g)["edge_index"]
        self.assertEqual(edges.shape, (2, 24))  # 12 undirected grid edges.
        degrees = np.bincount(edges[1], minlength=9)
        self.assertEqual(degrees.tolist(), [2, 3, 2, 3, 4, 3, 2, 3, 2])
        for a, b in edges.T:
            self.assertEqual((g.loc[a, ["x_mm", "y_mm"]] != g.loc[b, ["x_mm", "y_mm"]]).sum(), 1)
        self.assertEqual(graph.grid_topology(grid(11, 4))["edge_index"].shape, (2, 146))

    def test_missing_nodes_do_not_create_edges_across_a_hole(self):
        g = grid().drop(index=4).reset_index(drop=True)
        self.assertEqual(graph.grid_topology(g)["edge_index"].shape, (2, 16))
        # Known spacing also protects a grid missing an entire coordinate level.
        g = pd.DataFrame({"position": [1, 3], "x_mm": [0, 8], "y_mm": [0, 0]})
        self.assertEqual(graph.grid_topology(g, spacing_mm=[4, None])["edge_index"].shape, (2, 0))

    def test_node_order_and_position_numbers_do_not_define_connectivity(self):
        a, b = grid(), grid().sample(frac=1, random_state=0).reset_index(drop=True)
        def spatial_edges(g):
            edges = graph.grid_topology(g)["edge_index"]
            xy = g[["x_mm", "y_mm"]].to_numpy()
            return {(tuple(xy[i]), tuple(xy[j])) for i, j in edges.T}
        b.position += 10000
        self.assertEqual(spatial_edges(a), spatial_edges(b))

    def test_relative_displacements_preserve_direction_and_physical_distance(self):
        attrs = graph.grid_topology(grid())["edge_attr"]
        np.testing.assert_allclose(attrs[::2, :2], -attrs[1::2, :2])
        np.testing.assert_allclose(attrs[::2, 2], attrs[1::2, 2])
        np.testing.assert_allclose(np.sort(np.unique(attrs[:, 2])), [0.4, 1.25])

    def test_rejects_mixed_libraries_labels_duplicates_and_unknown_coordinates(self):
        g = grid(2, 1)
        for invalid in [g.assign(library_id=[1, 2]), g.assign(thickness=[1, 2]),
                        g.assign(position=1), g.assign(x_mm=0), g.assign(x_mm=np.nan)]:
            with self.assertRaises(ValueError):
                graph.grid_topology(invalid)

    def test_single_location_has_no_edges_but_can_be_predicted(self):
        g = grid(1, 1)
        topology = graph.grid_topology(g)
        self.assertEqual(topology["edge_index"].shape, (2, 0))
        model = graph.SubstrateGNN(3, hidden_dim=8)
        result = graph.predict_grid(model, np.ones((1, 3)), g)
        self.assertEqual(result.shape, (1, 2))
        self.assertTrue(np.isfinite(result).all())


class GraphModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.prepared = Path(cls.temp.name)
        rows = []
        for lib in range(1, 21):
            for point in range(3):
                row = {key: None for key in htem.RAW_FIELDS}
                row.update(id=lib * 10 + point, sample_library_id=lib, position=point,
                           x_mm=point * 4.0, y_mm=0, absolute_temp_c=300 + point * 50,
                           pdac="tool", deposition_sample_time_min=30 + lib,
                           deposition_initial_temp_c=300, deposition_compounds=["ZnO"], deposition_power=[50],
                           deposition_gases=["Ar"], deposition_gas_flow_sccm=[10],
                           deposition_substrate_material="glass", thickness=0.3 if point == 0 else None,
                           fpm_conductivity=10 + lib if point < 2 else None)
                rows.append(row)
        htem.prepare(pd.DataFrame(rows), cls.prepared, "graph-fixture")
        cls.data, cls.transforms, _ = htem.load_prepared(cls.prepared)
        cls.graphs, cls.report = graph.build_graph_splits(cls.prepared, cls.data, cls.transforms)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_full_grid_keeps_unlabeled_nodes_and_matches_existing_features(self):
        self.assertEqual(self.report.nodes.sum(), 60)
        self.assertEqual(self.report.labeled_nodes.sum(), 40)
        for split, graphs in self.graphs.items():
            for g in graphs:
                self.assertEqual(len(g.x), 3)
                self.assertEqual(g.label_mask.tolist(), [True, True, False])
                np.testing.assert_allclose(g.x[g.label_mask], self.data[split]["X"][g.output_index[g.label_mask]])
        self.assertEqual(set(self.report.split), set(htem.SPLITS))

    def test_batch_edges_never_cross_libraries_or_change_predictions(self):
        graphs = self.graphs["train"][:2]
        batch = graph.pack_graphs(graphs)
        edges, ids = batch["edge_index"], batch["node_library"]
        torch.testing.assert_close(ids[edges[0]], ids[edges[1]])
        model = graph.SubstrateGNN(graphs[0].x.shape[1], hidden_dim=8, dropout=0)
        model.eval()
        together = model(batch["x"], edges, batch["edge_attr"]).detach().numpy()
        first = graph.predict_grid(model, graphs[0].x, graphs[0].grid)
        np.testing.assert_allclose(together[:len(first)], first, rtol=1e-5, atol=1e-6)

    def test_node_permutation_equivariance(self):
        g = self.graphs["train"][0]
        model = graph.SubstrateGNN(g.x.shape[1], hidden_dim=8, dropout=0)
        order = np.array([2, 0, 1])
        original = graph.predict_grid(model, g.x, g.grid)
        shuffled = graph.predict_grid(model, g.x[order], g.grid.iloc[order])
        np.testing.assert_allclose(shuffled, original[order], rtol=1e-5, atol=1e-6)

    def test_messages_affect_neighbors_but_not_disconnected_component(self):
        g = pd.DataFrame({"position": [0, 1, 2, 3], "x_mm": [0, 1, 10, 11], "y_mm": 0})
        torch.manual_seed(0)
        model = graph.SubstrateGNN(3, hidden_dim=8, dropout=0)
        x = np.ones((4, 3), dtype="float32")
        a = graph.predict_grid(model, x, g)
        x[0] += 5
        b = graph.predict_grid(model, x, g)
        self.assertFalse(np.allclose(a[1], b[1]))
        np.testing.assert_allclose(a[2:], b[2:])

    def test_loss_masks_labels_and_restores_validation_minimum(self):
        settings = dict(hidden_dim=8, dropout=0, epochs=5, patience=2, batch_size=4)
        model, history, note = graph.fit_graph_model(self.graphs["train"], self.graphs["validation"], **settings)
        prediction = graph.predict_labeled(model, self.graphs["validation"], len(self.data["validation"]["X"]))
        loss = htem.observed_mse(prediction, self.data["validation"]["y"]).mean()
        self.assertAlmostEqual(loss, note["best_validation_MSE"], places=5)
        self.assertEqual(note["best_epoch"], int(history.loc[history.validation_loss.idxmin(), "epoch"]))
        self.assertLess(history.learning_rate.iloc[-1], history.learning_rate.iloc[0])
        mutated = copy.deepcopy(self.graphs)
        for graphs in mutated.values():
            for g in graphs:
                g.y[~g.label_mask] = 1e20
        masked, _, _ = graph.fit_graph_model(mutated["train"], mutated["validation"], **settings)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, masked.state_dict()[key])

    def test_refuses_graph_split_leakage(self):
        with self.assertRaisesRegex(ValueError, "disjoint"):
            graph.fit_graph_model(self.graphs["train"], self.graphs["train"], epochs=1)

    def test_reloaded_graph_model_predicts_entire_grid_without_labels(self):
        g = self.graphs["test"][0]
        model = graph.SubstrateGNN(g.x.shape[1], hidden_dim=8)
        a = graph.predict_grid(model, g.x, g.grid)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.joblib"
            joblib.dump(model, path)
            loaded = joblib.load(path)
            np.testing.assert_allclose(graph.predict_grid(loaded, g.x, g.grid), a)


if __name__ == "__main__":
    unittest.main()
