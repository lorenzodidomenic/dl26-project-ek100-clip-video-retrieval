"""Caricamento delle feature video e testo allineate per narration_id.

I file di feature hanno ordini di righe diversi (video per sessione, testo come split.csv).
Qui si allinea sempre per narration_id, mai per posizione: un errore di allineamento
non fa fallire niente, ma rende sbagliato ogni risultato successivo.
"""
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VIDEO = ROOT / "data/features/timesformer_k400_8f/timesformer_k400_8f.npz"
TEXT_DIR = ROOT / "data/features/text"
SPLIT = ROOT / "data/splits/split.csv"
SPLITS = ("train", "val", "test")


def l2_normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def load_split(split):
    if split not in SPLITS:
        raise ValueError(f"split sconosciuto: {split}")
    with open(SPLIT, newline="") as f:
        return [r for r in csv.DictReader(f) if r["split"] == split]


def _rows_by_id(path, ids):
    # righe del file nell'ordine di ids, cercate per narration_id
    data = np.load(path)
    index = {nid: i for i, nid in enumerate(data["narration_id"])}
    missing = [i for i in ids if i not in index]
    if missing:
        raise KeyError(f"{len(missing)} narration_id assenti da {path.name}, es. {missing[:3]}")
    return data["features"][[index[i] for i in ids]]


def load_pair(text_encoder, split, normalize=True):
    """Feature video e testo di uno split, riga i = stessa clip in entrambe.

    Ritorna (video (n, 768), testo (n, d), meta) dove meta e' un dict di array
    con narration_id, video_id, participant_id, narration, verb_class, noun_class.
    """
    rows = load_split(split)
    ids = [r["narration_id"] for r in rows]
    video = _rows_by_id(VIDEO, ids)
    text = _rows_by_id(TEXT_DIR / f"{text_encoder}.npz", ids)
    if normalize:
        video, text = l2_normalize(video), l2_normalize(text)

    meta = {k: np.array([r[k] for r in rows])
            for k in ["narration_id", "video_id", "participant_id", "narration"]}
    meta["verb_class"] = np.array([int(r["verb_class"]) for r in rows])
    meta["noun_class"] = np.array([int(r["noun_class"]) for r in rows])
    return video, text, meta
