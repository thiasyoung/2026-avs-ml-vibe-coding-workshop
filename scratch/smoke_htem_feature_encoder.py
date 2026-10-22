"""Execute notebook 01's optional atomic encoder against a real cached vector."""
from pathlib import Path
import hashlib
import io
import json
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "03_htem_conductivity"
for site in [ROOT / ".venv/Lib/site-packages", ROOT / ".venv/lib/python3.11/site-packages"]:
    if site.exists():
        sys.path.append(str(site))
sys.path.insert(0, str(MODULE))
import numpy as np
import torch
from alignn.models.alignn import ALIGNN, ALIGNNConfig
import htem_source_dos as dos

torch.set_num_threads(4)
notebook = json.loads((MODULE / "01_HTEM_Process_Data_Carpentry.ipynb").read_text(encoding="utf-8"))
source = next("".join(cell["source"]) for cell in notebook["cells"]
              if cell["cell_type"] == "code" and "def encode_reference_crystal(" in "".join(cell["source"]))
namespace = {}
exec(compile(source, "notebook_01_optional_encoder", "exec"), namespace)
reference = json.loads((MODULE / "data/htem_feature_candidates.json").read_text(encoding="utf-8"))
metadata = reference["provenance"]["crystal_embedding"]
path = MODULE / "outputs/htem_feature_study/reference_cache/jv_formation_energy_peratom_alignn.zip"
assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["archive_sha256"]
with zipfile.ZipFile(path) as archive:
    model = ALIGNN(ALIGNNConfig(**metadata["model"]))
    model.load_state_dict(torch.load(io.BytesIO(archive.read(metadata["checkpoint"])),
                                    map_location="cpu", weights_only=False)["model"], strict=True)
cache = json.loads((MODULE / "outputs/htem_feature_study/reference_cache/alignn_structure_embeddings.json").read_text(encoding="utf-8"))
selected = next(iter(cache))
with zipfile.ZipFile(MODULE / "outputs/htem_source_reference_cache/jarvis_edos_29216859.zip") as archive:
    with archive.open(dos.MEMBER) as member:
        record = next(item for item in dos.iter_json_array(io.TextIOWrapper(member, encoding="utf-8")) if str(item["jid"]) == selected)
vector = namespace["encode_reference_crystal"](model, record["atoms"])
np.testing.assert_allclose(vector, cache[selected]["embedding"], rtol=1e-6, atol=1e-7)
assert vector.shape == (256,) and np.isfinite(vector).all()
print("Notebook 01 optional atomic encoder reproduced cached official ALIGNN vector:", selected, vector.shape)
