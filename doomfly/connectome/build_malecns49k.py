"""Alternative backbone: the 49,393-neuron MaleCNS central-brain graph packaged at
https://huggingface.co/datasets/fernandofernandes/fly-connectome-49k (CC BY 4.0).

Converts it into the same packed npz schema as `build.py` so FlyNet can run on
either connectome with a flag (`--connectome flywire783|malecns49k`).

Schema (identical to connectome_783.npz):
  n, src, dst (sorted by dst), sign (+1/-1), syn_count (float), group, is_input,
  is_readout, pos (soma xyz, nan if missing), super_class (str)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = "fernandofernandes/fly-connectome-49k"
OUT = Path("data/processed/connectome_malecns49k.npz")


def sha(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(out: Path = OUT):
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(REPO, repo_type="dataset"))
    manifest = json.loads((root / "manifest.json").read_text())
    n = manifest["neurons"]
    off = np.fromfile(root / "graph/edges_offsets.i32", dtype=np.int32).astype(np.int64)
    src = np.fromfile(root / "graph/edges_source.u16", dtype=np.uint16).astype(np.int64)
    w = np.fromfile(root / "graph/edges_weight.f32", dtype=np.float32)
    assert len(off) == n + 1 and off[-1] == len(src) == manifest["edges"]
    dst = np.repeat(np.arange(n, dtype=np.int64), np.diff(off))

    # recover synapse counts: |w| is quantised on a lattice whose unit is the
    # smallest non-zero magnitude (empirically 0.275 * provenance global scale);
    # count = |w| / unit is an integer to float32 precision.
    unit = float(np.abs(w).min())
    counts = np.abs(w) / unit
    frac_err = np.abs(counts - np.round(counts)).max()
    print(f"weight unit {unit:.6g}; synapse-count reconstruction: max deviation from integer {frac_err:.4f}")
    assert frac_err < 0.05, "weights are not an integer lattice; refusing to invent synapse counts"
    syn = np.round(counts).astype(np.float32)
    assert syn.min() >= 1
    sign = np.sign(w).astype(np.int8)
    assert (sign != 0).all()

    superclass = np.array(json.loads((root / "neurons/superclass.json").read_text()))
    in_index = np.fromfile(root / "interface/in_index.i32", dtype=np.int32)
    is_input = np.zeros(n, bool)
    is_input[in_index] = True
    is_readout = ~is_input  # central brain intrinsic + ascending + descending + the few unused sensory
    group = np.zeros(n, np.int8)
    group[superclass == "visual_projection"] = 1
    group[is_input] = 2
    group[superclass == "descending_neuron"] = 3

    soma = np.fromfile(root / "neurons/soma_position.i32", dtype=np.int32).reshape(n, 3).astype(np.float32)
    valid = np.fromfile(root / "neurons/soma_valid.u8", dtype=np.uint8).astype(bool)
    pos = soma * 8e-3  # 8nm voxels -> micrometres
    pos[~valid] = np.nan

    # edges already sorted by dst (CSR by destination)
    assert (np.diff(dst) >= 0).all()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, n=n, src=src, dst=dst, sign=sign, syn_count=syn, group=group, is_input=is_input,
             is_readout=is_readout, pos=pos, super_class=superclass)
    meta = {
        "name": "malecns49k", "source": REPO, "license": "CC BY 4.0",
        "hf_revision": root.name, "neurons": int(n), "edges": int(len(src)),
        "synapses": int(syn.sum()), "inputs": int(is_input.sum()), "readout": int(is_readout.sum()),
        "descending": int((superclass == "descending_neuron").sum()), "positioned": int(valid.sum()),
        "excitatory_fraction": float((sign > 0).mean()),
        "superclass_counts": {k: int(v) for k, v in zip(*np.unique(superclass, return_counts=True))},
        "manifest_sha256": sha(root / "manifest.json"),
        "note": manifest["provenance"].get("omission_reason"),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT)
