"""Estrae le feature TimeSformer per le clip presenti nei manifest.

Per ogni clip: legge i frame indicati in frame_indices (ffmpeg), applica il
preprocessing del processor e salva il vettore CLS (768 dim) con il suo narration_id.

Salvataggio incrementale: un file .npz per sessione in <out>/sessions/<video_id>.npz.
Le sessioni gia' presenti vengono saltate, quindi lo script si puo' interrompere e
rilanciare. A fine lavoro unisce tutto in <out>/timesformer_k400_8f.npz.
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# il filtro --session/--limit e' volutamente parallelo a cut_clips.py (stessa CLI)
# pylint: disable=duplicate-code
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
# pylint: disable=wrong-import-position
from models.timesformer import TimesformerEncoder  # noqa: E402

CLIPS = ROOT / "data/clips/clips"
MANIFESTS = ROOT / "data/clips/manifest"
SHORT_SIDE = 224

# legge le dimensioni del video dopo il preprocess, senza caricare il videos
def frame_size(clip_path):
    """Restituisce (larghezza, altezza) della clip dopo lo scale del lato corto a 224."""
    # larghezza dopo lo scale che porta il lato corto a 224, come fa il processor
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", str(clip_path)],
        capture_output=True, text=True, check=True,
    ).stdout
    stream = json.loads(out)["streams"][0]
    w, h = int(stream["width"]), int(stream["height"])
    return round(w * SHORT_SIDE / h), SHORT_SIDE

# legge solo i frame che servono (queli del manifest)
# ritorna un array di frame per quella clip basata sugli indici nel manifest
def read_frames(clip_path, frame_indices):
    """Legge dalla clip i frame agli indici indicati e li restituisce come immagini PIL."""
    w, h = frame_size(clip_path)
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(clip_path),
           "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    # gli indici del manifest sono stimati dal fps:
    # se la clip ha qualche frame in meno, si ferma all'ultimo
    idx = np.minimum(frame_indices, len(frames) - 1)
    return [Image.fromarray(frames[i]) for i in idx]    # immagini pil dagli indici richiesti

# carica le clip da encodare
def load_rows():
    """Legge tutti i manifest e tiene solo le clip per cui esiste il file mp4."""
    rows = []
    for manifest in sorted(MANIFESTS.glob("*.csv")):
        with open(manifest, newline="", encoding="utf-8") as f:
            rows += list(csv.DictReader(f))
    return [r for r in rows if (CLIPS / f"{r['narration_id']}.mp4").exists()]


def group_by_session(rows):
    """Raggruppa le righe per video_id."""
    sessions = {}
    for r in rows:
        sessions.setdefault(r["video_id"], []).append(r)
    return sessions

#funzione del processing
# per ogni riga -> quindi per ogni clip -> applica l'encoding -> salva le features
def is_complete(session_file, expected):
    """True se il file della sessione esiste e ha una riga per ogni clip attesa."""
    # un file esistente non basta: deve avere una riga per ogni clip attesa
    if not session_file.exists():
        return False
    return len(np.load(session_file)["narration_id"]) == expected


def process_session(encoder, rows, batch, out_file):
    """Estrae le feature di una sessione a blocchi di clip e le salva in out_file."""
    features, ids = [], []
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        clips = [read_frames(CLIPS / f"{r['narration_id']}.mp4",
                             np.array([int(i) for i in r["frame_indices"].split()]))
                 for r in chunk]
        features.append(encoder.encode(clips))
        ids += chunk

    # scrittura su file temporaneo e rinomina: un'interruzione non lascia una sessione a meta'
    tmp = out_file.with_name(out_file.stem + ".tmp.npz")
    np.savez(
        tmp,
        features=np.concatenate(features).astype(np.float32),
        narration_id=np.array([r["narration_id"] for r in ids]),
        video_id=np.array([r["video_id"] for r in ids]),
        participant_id=np.array([r["participant_id"] for r in ids]),
    )
    tmp.replace(out_file)

#funzione utilitaria per poter stoppare/riprendere encoding
def merge_sessions(session_dir, final_file):
    """Unisce i file delle singole sessioni in un unico .npz. Ritorna il numero di feature."""
    parts = [np.load(f) for f in sorted(session_dir.glob("*.npz"))]
    keys = ["features", "narration_id", "video_id", "participant_id"]
    merged = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    np.savez(final_file, **merged)
    return len(merged["narration_id"])


def main():
    """Punto di ingresso: estrae le feature delle sessioni non ancora completate."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", type=int, default=8, help="clip per batch")
    parser.add_argument("--limit", type=int, help="processa solo le prime N sessioni (prova)")
    parser.add_argument("--session", help="processa solo questa sessione, es. P22_01")
    parser.add_argument("--out", type=Path, default=ROOT / "data/features/timesformer_k400_8f")
    args = parser.parse_args()

    sessions = group_by_session(load_rows())
    video_ids = sorted(sessions)
    if args.session:
        video_ids = [v for v in video_ids if v == args.session]
    if args.limit:
        video_ids = video_ids[: args.limit]
    session_dir = args.out / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)

    todo = [v for v in video_ids if not is_complete(session_dir / f"{v}.npz", len(sessions[v]))]
    print(f"sessioni totali: {len(video_ids)} | da fare: {len(todo)}")

    encoder = TimesformerEncoder()
    for i, video_id in enumerate(todo, 1):
        process_session(encoder, sessions[video_id], args.batch, session_dir / f"{video_id}.npz")
        print(f"[{i}/{len(todo)}] {video_id}: {len(sessions[video_id])} clip")

    final = args.out / "timesformer_k400_8f.npz"
    n = merge_sessions(session_dir, final)
    print(f"unite {n} feature in {final}")


if __name__ == "__main__":
    main()
