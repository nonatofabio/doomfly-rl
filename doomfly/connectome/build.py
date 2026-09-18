"""Connectome ETL: FlyWire v783 (Shiu et al. author release) -> packed graph.

Inputs
------
* Connectivity_783.parquet  (Shiu et al. 2024, github.com/philshiu/Drosophila_brain_model, MIT)
    columns: Presynaptic_ID, Postsynaptic_ID, Presynaptic_Index, Postsynaptic_Index,
             Connectivity (synapse count), Excitatory (+1/-1), Excitatory x Connectivity
* Completeness_783.csv      (same repo) -> the 138,639 neuron ids in index order
* Supplemental_file1_neuron_annotations.tsv (Schlegel et al. 2024,
    github.com/flyconnectome/flywire_annotations) -> super_class, cell_class, positions

Output
------
data/processed/connectome_783.npz with
  n                : int, number of neurons (138,639)
  root_id          : int64[n]
  src, dst         : int32[E] edge endpoints (pre -> post), E = 15,091,983
  syn_count        : int16[E]
  sign             : int8[E]  (+1 excitatory, -1 inhibitory) -- frozen, never learned
  group            : int8[n]  0 other, 1 optic, 2 input (visual sensory), 3 descending(+motor)
  is_input         : bool[n]  10,855 visual sensory neurons (encoder targets)
  is_readout       : bool[n]  central + descending + motor neurons (decoder sources)
  pos              : float32[n,3] soma/arbor position in nm (nan when unknown)
  super_class      : str[n]

These counts reproduce chessfly's connectome_meta.json exactly:
138,639 neurons / 15,091,983 edges / 54,492,922 synapses / 10,855 inputs / 1,409 descending.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SHIU = "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/"
SCHLEGEL = "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/"
SOURCES = {
    "Connectivity_783.parquet": SHIU + "Connectivity_783.parquet",
    "Completeness_783.csv": SHIU + "Completeness_783.csv",
    "LICENSE": SHIU + "LICENSE",
    "Supplemental_file1_neuron_annotations.tsv": SCHLEGEL + "Supplemental_file1_neuron_annotations.tsv",
}
# hashes published in mlabonne/chessfly connectome_meta.json -- we verify we use the same files
EXPECTED_SHA256 = {
    "Connectivity_783.parquet": "efeb23fb99098e9c390f6869969b2a121a2ee92c833cfc45ecb2c1d8e1af0347",
    "Completeness_783.csv": "bbb847a4cc2caaa7a16349722d220c087317b946d148d4d592d94d250617a311",
    "Supplemental_file1_neuron_annotations.tsv": "9a4f8b2f843196074431ebd7cd883536afa1be86c8a4ce90970441e8be81d1be",
}
GROUPS = {"other": 0, "optic": 1, "input": 2, "descending": 3}
READOUT_CLASSES = ("central", "descending", "motor")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(raw_dir: Path) -> dict[str, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, url in SOURCES.items():
        p = raw_dir / name
        if not p.exists():
            print(f"downloading {name}")
            with requests.get(url, stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(p, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
        out[name] = p
        if name in EXPECTED_SHA256:
            got = sha256(p)
            status = "ok" if got == EXPECTED_SHA256[name] else "MISMATCH (upstream file changed?)"
            print(f"  {name}: sha256 {got[:16]}... {status}")
    return out


def build(raw: dict[str, Path], out_path: Path) -> dict:
    comp = pd.read_csv(raw["Completeness_783.csv"])
    root_id = comp.iloc[:, 0].to_numpy(dtype=np.int64)
    n = len(root_id)

    con = pd.read_parquet(raw["Connectivity_783.parquet"])
    # Shiu's index columns are positions into Completeness order; verify rather than trust.
    pre_idx = con["Presynaptic_Index"].to_numpy(np.int64)
    post_idx = con["Postsynaptic_Index"].to_numpy(np.int64)
    assert (root_id[pre_idx] == con["Presynaptic_ID"].to_numpy()).all()
    assert (root_id[post_idx] == con["Postsynaptic_ID"].to_numpy()).all()
    syn = con["Connectivity"].to_numpy(np.int64)
    sign = con["Excitatory"].to_numpy(np.int64)
    assert set(np.unique(sign)) <= {-1, 1}
    assert syn.max() < 32768

    ann = pd.read_csv(raw["Supplemental_file1_neuron_annotations.tsv"], sep="\t", low_memory=False)
    ann = ann.drop_duplicates("root_id").set_index("root_id")
    ann = ann.reindex(root_id)
    super_class = ann["super_class"].fillna("unknown").to_numpy(dtype=object).astype(str)
    cell_class = ann["cell_class"].fillna("unknown").to_numpy(dtype=object).astype(str)
    pos = ann[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=np.float32)

    is_input = (super_class == "sensory") & (cell_class == "visual")
    is_readout = np.isin(super_class, READOUT_CLASSES)
    group = np.zeros(n, dtype=np.int8)
    group[np.isin(super_class, ["optic", "visual_projection", "visual_centrifugal"])] = GROUPS["optic"]
    group[is_input] = GROUPS["input"]
    group[np.isin(super_class, ["descending", "motor"])] = GROUPS["descending"]

    # sort edges by destination for CSR (rows = post-synaptic neuron)
    order = np.lexsort((pre_idx, post_idx))
    src = pre_idx[order].astype(np.int32)
    dst = post_idx[order].astype(np.int32)
    syn = syn[order].astype(np.int16)
    sign = sign[order].astype(np.int8)

    meta = {
        "dataset": "FlyWire FAFB v783, Shiu et al. author release",
        "neurons": int(n),
        "edges": int(len(src)),
        "synapses": int(syn.astype(np.int64).sum()),
        "inputs": int(is_input.sum()),
        "readout": int(is_readout.sum()),
        "descending": int((group == GROUPS["descending"]).sum()),
        "positioned": int(np.isfinite(pos).all(axis=1).sum()),
        "excitatory_fraction": float((sign > 0).mean()),
        "groups": GROUPS,
        "readout_classes": list(READOUT_CLASSES),
        "super_class_counts": {k: int(v) for k, v in zip(*np.unique(super_class, return_counts=True))},
        "sources": {k: EXPECTED_SHA256.get(k) for k in SOURCES},
        "references": [
            "https://doi.org/10.1038/s41586-024-07763-9",  # Shiu et al. 2024
            "https://doi.org/10.1038/s41586-024-07686-5",  # Schlegel et al. 2024
            "https://doi.org/10.1038/s41586-024-07558-y",  # Dorkenwald et al. 2024
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path, n=n, root_id=root_id, src=src, dst=dst, syn_count=syn, sign=sign,
        group=group, is_input=is_input, is_readout=is_readout, pos=pos,
        super_class=super_class.astype("U24"),
    )
    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out", default="data/processed/connectome_783.npz")
    a = ap.parse_args()
    raw = download(Path(a.raw_dir))
    meta = build(raw, Path(a.out))
    print(json.dumps({k: v for k, v in meta.items() if k not in ("super_class_counts", "sources")}, indent=2))
    # invariants that must match chessfly's connectome_meta.json
    assert meta["neurons"] == 138639, meta["neurons"]
    assert meta["edges"] == 15091983, meta["edges"]
    assert meta["synapses"] == 54492922, meta["synapses"]
    assert meta["inputs"] == 10855, meta["inputs"]
    assert meta["descending"] == 1409, meta["descending"]
    print("connectome invariants OK")


if __name__ == "__main__":
    main()
