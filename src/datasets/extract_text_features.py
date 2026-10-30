"""Estrae le feature testuali delle narrazioni con uno dei text encoder congelati.

Legge narration_id e narration da data/splits/split.csv (tutte le clip del subset).
Molte clip hanno la stessa frase ("take plate"), quindi codifica ogni frase unica una
volta sola e poi assegna il vettore a ogni narration_id.

Output: data/features/text/<encoder>.npz con features (n, dim), narration_id, narration.
Stesso formato delle feature video, per allinearle per narration_id.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from models.text_encoders import ENCODERS  # noqa: E402

SPLIT = ROOT / "data/splits/split.csv"
OUT_DIR = ROOT / "data/features/text"


def load_narrations():
    with open(SPLIT, newline="") as f:
        rows = list(csv.DictReader(f))
    return [r["narration_id"] for r in rows], [r["narration"] for r in rows]


def encode_unique(encoder, texts, batch):
    # codifica ogni frase unica una sola volta, a blocchi
    unique = sorted(set(texts))
    vectors = []
    for start in range(0, len(unique), batch):
        vectors.append(encoder.encode(unique[start:start + batch]))
        print(f"  [{min(start + batch, len(unique))}/{len(unique)}] frasi uniche")
    by_text = dict(zip(unique, np.concatenate(vectors)))
    return np.stack([by_text[t] for t in texts]).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoder", nargs="+", default=list(ENCODERS), choices=list(ENCODERS))
    parser.add_argument("--batch", type=int, default=256, help="frasi per batch")
    args = parser.parse_args()

    ids, texts = load_narrations()
    print(f"clip: {len(ids)} | frasi uniche: {len(set(texts))}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name in args.encoder:
        print(f"== {name}")
        features = encode_unique(ENCODERS[name](), texts, args.batch)
        out = OUT_DIR / f"{name}.npz"
        np.savez(out, features=features, narration_id=np.array(ids), narration=np.array(texts))
        print(f"salvato: {out} {features.shape}")


if __name__ == "__main__":
    main()
