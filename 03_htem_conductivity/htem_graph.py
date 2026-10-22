"""Grid-local node regression in plain PyTorch; no measured neighbor inputs.

One independent graph per deposition. The full planned XY grid determines the
topology; incomplete labels affect the loss mask, never the edges or node set.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import htem_workshop as htem

EDGE_LENGTH_SCALE_MM = 10.0  # Fixed physical reference, not fitted to any outcomes.
GRAPH_RULE = "bidirectional horizontal/vertical nearest grid-level neighbors; no diagonals or cross-library edges"


def grid_topology(grid, spacing_mm=None):
    """Build four-neighbor edges without sorting/relabeling the input nodes.

    Infer each axis's nominal step from its smallest observed level spacing.
    Gaps larger than 1.05 times that step are not bridged. Supply spacing_mm
    explicitly if the planned grid omits whole coordinate levels: their nominal
    spacing cannot always be inferred from surviving coordinates alone.
    """
    grid = pd.DataFrame(grid)
    allowed = {"position", "x_mm", "y_mm", "sample_library_id", "library_id", "id"}
    if set(grid) - allowed:
        raise ValueError("Graph construction accepts grid identities/XY only, not measured properties.")
    if not {"position", "x_mm", "y_mm"}.issubset(grid) or grid.empty:
        raise ValueError("Provide a nonempty position/x_mm/y_mm grid.")
    for field in ["sample_library_id", "library_id"]:
        if field in grid and (grid[field].isna().any() or grid[field].nunique() != 1):
            raise ValueError("Build exactly one deposition graph at a time.")
    xy = grid[["x_mm", "y_mm"]].to_numpy(dtype=float)
    if not np.isfinite(xy).all() or grid.position.isna().any():
        raise ValueError("Graph positions and coordinates must be known and finite.")
    if grid.position.duplicated().any() or grid.duplicated(["x_mm", "y_mm"]).any():
        raise ValueError("Ambiguous duplicate positions/coordinates; do not average graph nodes.")
    levels = [np.unique(xy[:, axis]) for axis in [0, 1]]
    steps = [float(np.diff(v).min()) if len(v) > 1 else None for v in levels]
    if spacing_mm is not None:
        if len(spacing_mm) != 2 or any(v is not None and (not np.isfinite(v) or v <= 0) for v in spacing_mm):
            raise ValueError("Nominal grid spacing must be two positive values (or None for a single-level axis).")
        steps = list(spacing_mm)
    lookup = {tuple(point): i for i, point in enumerate(xy)}
    pairs = []
    for i, point in enumerate(xy):
        for axis in [0, 1]:
            index = np.searchsorted(levels[axis], point[axis])
            if index + 1 == len(levels[axis]) or steps[axis] is None:
                continue
            following = levels[axis][index + 1]
            if not np.isclose(following - point[axis], steps[axis], rtol=0.05, atol=1e-6):
                continue
            neighbor = point.copy()
            neighbor[axis] = following
            j = lookup.get(tuple(neighbor))
            if j is not None:
                pairs.extend([(i, j), (j, i)])
    edge_index = np.asarray(pairs, dtype=np.int64).reshape(-1, 2).T.copy()
    displacement = xy[edge_index[1]] - xy[edge_index[0]]
    edge_attr = np.column_stack([displacement, np.linalg.norm(displacement, axis=1)])
    edge_attr = (edge_attr / EDGE_LENGTH_SCALE_MM).astype("float32")
    return {"edge_index": edge_index, "edge_attr": edge_attr, "spacing_mm": steps}


@dataclass
class SubstrateGraph:
    library_id: int
    leakage_group: int
    split: str
    grid: pd.DataFrame
    x: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    y: np.ndarray
    label_mask: np.ndarray
    output_index: np.ndarray


def build_graph_splits(prepared, data, transforms):
    """Reuse the exact point-model split, preprocessing, targets, and row order."""
    prepared = Path(prepared)
    clean = pd.read_parquet(prepared / "clean_process_table.parquet")
    grids = pd.read_parquet(prepared / "substrate_grids.parquet")
    result, reports = {}, []
    for split in htem.SPLITS:
        part = data[split]
        lookup = {(int(lib), pos): i for i, (lib, pos) in enumerate(zip(part["library_id"], part["position"]))}
        if len(lookup) != len(part["X"]):
            raise ValueError("Duplicate labeled library/position keys.")
        result[split] = []
        for library_id, rows in clean[clean.split.eq(split)].groupby("sample_library_id", sort=True):
            if rows[htem.RECIPE_FEATURES].nunique(dropna=False).gt(1).any() or rows.leakage_group.nunique() != 1:
                raise ValueError("Recipe/group changes within a deposition; review metadata.")
            planned = grids[grids.sample_library_id.eq(library_id)][[*htem.GEOMETRY_FIELDS, "local_temperature_C"]].reset_index(drop=True)
            grid = planned[htem.GEOMETRY_FIELDS]
            topology = grid_topology(grid)
            frame = htem.grid_feature_frame(rows, planned)
            x = transforms["preprocessor"].transform(frame).astype("float32")
            output_index = np.array([lookup.get((int(library_id), pos), -1) for pos in grid.position])
            label_mask = output_index >= 0
            y = np.zeros((len(grid), len(htem.TARGETS)), dtype="float32")
            y[label_mask] = part["y"][output_index[label_mask]]
            np.testing.assert_allclose(x[label_mask], part["X"][output_index[label_mask]], rtol=1e-5, atol=1e-6)
            graph = SubstrateGraph(int(library_id), int(rows.leakage_group.iloc[0]), split, grid, x,
                                   topology["edge_index"], topology["edge_attr"], y, label_mask, output_index)
            result[split].append(graph)
            degrees = np.bincount(graph.edge_index[1], minlength=len(grid))
            reports.append({"library_id": int(library_id), "split": split, "nodes": len(grid),
                            "labeled_nodes": int(label_mask.sum()),
                            "thickness_labels": int(np.isfinite(y[label_mask, 0]).sum()),
                            "conductivity_labels": int(np.isfinite(y[label_mask, 1]).sum()),
                            "undirected_edges": graph.edge_index.shape[1] // 2,
                            "isolated_nodes": int((degrees == 0).sum()), "spacing_x_mm": topology["spacing_mm"][0],
                            "spacing_y_mm": topology["spacing_mm"][1]})
        indices = np.concatenate([g.output_index[g.label_mask] for g in result[split]])
        if not np.array_equal(np.sort(indices), np.arange(len(part["X"]))):
            raise ValueError("Every labeled row must map to exactly one graph node.")
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if {g.leakage_group for g in result[a]} & {g.leakage_group for g in result[b]}:
            raise ValueError("Connected recipe groups cross graph splits.")
    return result, pd.DataFrame(reports)


def pack_graphs(graphs):
    """Disjoint-union batching: offset edges, never connect different libraries."""
    if not graphs or len({g.library_id for g in graphs}) != len(graphs):
        raise ValueError("Batch needs distinct, nonempty deposition graphs.")
    edges, offset = [], 0
    for graph in graphs:
        if graph.edge_index.size and (graph.edge_index.min() < 0 or graph.edge_index.max() >= len(graph.x)):
            raise ValueError("Graph edge is outside its own deposition.")
        edges.append(graph.edge_index + offset)
        offset += len(graph.x)
    return {
        "x": torch.from_numpy(np.concatenate([g.x for g in graphs])),
        "edge_index": torch.from_numpy(np.concatenate(edges, axis=1)),
        "edge_attr": torch.from_numpy(np.concatenate([g.edge_attr for g in graphs])),
        "y": torch.from_numpy(np.concatenate([g.y for g in graphs])),
        "mask": torch.from_numpy(np.concatenate([g.label_mask for g in graphs])),
        "node_library": torch.tensor(np.concatenate([np.full(len(g.x), g.library_id) for g in graphs])),
    }


class GridMessageLayer(nn.Module):
    """Directional edge-conditioned neighbor mean plus a separate self update."""
    def __init__(self, width, dropout):
        super().__init__()
        self.message = nn.Sequential(nn.Linear(width + 3, width), nn.ReLU())
        self.self_update = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, edge_index, edge_attr):
        source, destination = edge_index
        messages = self.message(torch.cat([h[source], edge_attr], dim=1))
        total = torch.zeros_like(h).index_add_(0, destination, messages)
        degree = torch.bincount(destination, minlength=len(h)).clamp_min(1).unsqueeze(1)
        return h + self.dropout(torch.relu(self.self_update(h) + total / degree))


class SubstrateGNN(nn.Module):
    """Two-output node regressor, not graph-average prediction or label diffusion."""
    def __init__(self, input_dim, hidden_dim=64, message_layers=2, dropout=0.05):
        super().__init__()
        if input_dim < 1 or hidden_dim < 2 or message_layers < 1 or not 0 <= dropout < 1:
            raise ValueError("Use positive dimensions/layers and dropout in [0,1).")
        self.config = dict(input_dim=input_dim, hidden_dim=hidden_dim, message_layers=message_layers, dropout=dropout)
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.layers = nn.ModuleList([GridMessageLayer(hidden_dim, dropout) for _ in range(message_layers)])
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 2))

    def forward(self, x, edge_index, edge_attr):
        h = self.encoder(x)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        return self.head(h)


def _forward(model, batch):
    return model(batch["x"], batch["edge_index"], batch["edge_attr"])


def fit_graph_model(train, validation, hidden_dim=64, message_layers=2, dropout=0.05,
                    lr=5e-4, weight_decay=1e-3, epochs=300, patience=40, batch_size=16, seed=42):
    """Masked node MSE, AdamW, cosine decay, validation-minimum checkpoint."""
    if not train or not validation or lr <= 0 or min(epochs, patience, batch_size) < 1:
        raise ValueError("Need nonempty graph splits and positive training settings.")
    if ({g.library_id for g in train} & {g.library_id for g in validation}
            or {g.leakage_group for g in train} & {g.leakage_group for g in validation}):
        raise ValueError("Training and validation graphs must have disjoint depositions/recipe groups.")
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    rng = np.random.default_rng(seed)
    model = SubstrateGNN(train[0].x.shape[1], hidden_dim, message_layers, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 100)
    full_train, full_val = pack_graphs(train), pack_graphs(validation)
    if not full_train["mask"].any() or not full_val["mask"].any():
        raise ValueError("Each split needs usable node labels; never invent missing measurements.")
    loss_fn = htem.masked_mse
    best, best_epoch, weights, waiting, history = float("inf"), 0, None, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train))
        for start in range(0, len(order), batch_size):
            batch = pack_graphs([train[i] for i in order[start:start + batch_size]])
            if not batch["mask"].any():
                continue
            optimizer.zero_grad()
            loss = loss_fn(_forward(model, batch)[batch["mask"]], batch["y"][batch["mask"]])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_loss = float(loss_fn(_forward(model, full_train)[full_train["mask"]], full_train["y"][full_train["mask"]]))
            val_loss = float(loss_fn(_forward(model, full_val)[full_val["mask"]], full_val["y"][full_val["mask"]]))
        if not np.isfinite([train_loss, val_loss]).all():
            raise ValueError("Nonfinite graph loss; review inputs/hyperparameters.")
        history.append(dict(epoch=epoch, train_loss=train_loss, validation_loss=val_loss,
                            learning_rate=optimizer.param_groups[0]["lr"]))
        if val_loss < best:
            best, best_epoch, weights, waiting = val_loss, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            waiting += 1
        scheduler.step()
        if waiting >= patience:
            break
    model.load_state_dict(weights)
    model.eval()
    return model, pd.DataFrame(history), {
        "best_epoch": best_epoch, "epochs_run": len(history), "best_validation_MSE": best,
        "stop_reason": "validation patience" if waiting >= patience else "epoch cap (not proof of convergence)",
        "topology": GRAPH_RULE, "edge_length_scale_mm": EDGE_LENGTH_SCALE_MM,
    }


def predict_labeled(model, graphs, output_size):
    """Return predictions in the original point-array order, not graph/node order."""
    model.eval()
    prediction, counts = np.empty((output_size, 2), dtype="float32"), np.zeros(output_size, dtype=int)
    with torch.no_grad():
        batch = pack_graphs(graphs)
        values = _forward(model, batch).numpy()
    offset = 0
    for graph in graphs:
        idx = graph.output_index[graph.label_mask]
        prediction[idx] = values[offset:offset + len(graph.x)][graph.label_mask]
        np.add.at(counts, idx, 1)
        offset += len(graph.x)
    if not (counts == 1).all() or not np.isfinite(prediction).all():
        raise ValueError("Predictions must map to each labeled point exactly once.")
    return prediction


def predict_grid(model, x, grid, spacing_mm=None):
    """Recipe + entire planned grid; predict ALL nodes, including unlabeled ones."""
    topology = grid_topology(pd.DataFrame(grid)[htem.GEOMETRY_FIELDS], spacing_mm)
    x = np.asarray(x, dtype="float32")
    if x.ndim != 2 or len(x) != len(grid) or not np.isfinite(x).all():
        raise ValueError("Provide one finite recipe/XY feature row per grid node.")
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(x), torch.from_numpy(topology["edge_index"]),
                     torch.from_numpy(topology["edge_attr"])).numpy()


def plot_grid_graph(grid, title="Planned substrate connectivity"):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    topology = grid_topology(grid)
    xy = pd.DataFrame(grid)[["x_mm", "y_mm"]].to_numpy()
    edges = topology["edge_index"][:, ::2].T  # Display each undirected edge once.
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.add_collection(LineCollection(xy[edges], colors="#94a3b8", linewidths=1.2))
    degrees = np.bincount(topology["edge_index"][1], minlength=len(xy))
    dots = ax.scatter(*xy.T, c=degrees, cmap="Blues", vmin=0, vmax=4, edgecolors="#17365d", s=45, zorder=3)
    fig.colorbar(dots, ax=ax, label="Number of neighboring locations", ticks=range(5))
    ax.set(title=title, xlabel="Substrate x (mm)", ylabel="Substrate y (mm)", aspect="equal")
    ax.invert_yaxis()
    ax.margins(0.08)
    fig.tight_layout()
    return fig
