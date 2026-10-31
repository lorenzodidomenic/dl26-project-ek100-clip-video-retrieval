"""Taglia i segmenti dei CSV di retrieval dalle sessioni EPIC-KITCHENS in clip mp4.

Per ogni riga del CSV (train e/o test, partecipanti del subset):
- legge start_timestamp e stop_timestamp e li converte in secondi;
- taglia il segmento dalla sessione con ffmpeg (ri-codifica, taglio preciso);
- salva la clip in <out>/clips/<narration_id>.mp4;
- calcola k indici di frame uniformi nella clip e li scrive nel manifest.

Lavora per sessione: a fine sessione scrive <out>/manifest/<video_id>.csv.
Le clip gia' presenti non vengono ritagliate di nuovo, quindi lo script si puo'
interrompere e rilanciare, e una prova con --clips 200 non blocca il taglio completo.
Con --clips N il campione e' casuale e ripetibile (--seed), non le prime N righe.
"""
import argparse
import csv
import random
import subprocess
from pathlib import Path

# radice del progetto, qualunque sia la cartella di lancio
ROOT = Path(__file__).resolve().parents[2]
ANN = ROOT / "src/datasets/epic-kitchens-100-annotations"
DATA = ROOT / "data/EPIC-KITCHENS"
SUBSET = ["P22", "P30", "P24", "P27"]
MANIFEST_FIELDS = [
    "narration_id", "participant_id", "video_id", "start_s", "stop_s",
    "fps", "n_frames", "frame_indices", "clip_path",
]


def to_seconds(timestamp):
    """Converte un timestamp HH:MM:SS.mmm in secondi."""
    # formato HH:MM:SS.mmm
    h, m, s = timestamp.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def uniform_indices(n_frames, k):
    """Restituisce k indici di frame distribuiti uniformemente nella clip."""
    # k indici distribuiti uniformemente tra primo e ultimo frame della clip
    if k == 1:
        return [(n_frames - 1) // 2]
    return [round(i * (n_frames - 1) / (k - 1)) for i in range(k)]


def find_session(video_id):
    """Trova il file video della sessione cercandolo nella cartella del partecipante."""
    folder = DATA / video_id.split("_")[0] / "videos"
    for f in folder.iterdir():
        if f.stem == video_id and f.suffix.lower() == ".mp4":
            return f
    raise FileNotFoundError(f"sessione non trovata: {video_id}")


def cut_clip(src, start, stop, dst):
    """Taglia il segmento [start, stop] di src in dst con ffmpeg, senza audio."""
    # -ss prima di -i: seek veloce. Con la ri-codifica il taglio resta preciso.
    cmd = [
        "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{stop - start:.3f}",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def process_session(video_id, rows, fps, out_dir, k):  # pylint: disable=too-many-locals
    """Taglia le clip di una sessione e scrive il suo manifest. Ritorna il numero di clip."""
    # lo skip e' per clip, non per sessione: una prova parziale non blocca il taglio completo
    manifest = out_dir / "manifest" / f"{video_id}.csv"
    src = None
    records = []
    for r in rows:
        start = to_seconds(r["start_timestamp"])
        stop = to_seconds(r["stop_timestamp"])
        dst = out_dir / "clips" / f"{r['narration_id']}.mp4"

        if not dst.exists():
            src = src or find_session(video_id)
            # scrivo in un file temporaneo e rinomino a fine taglio:
            # un'interruzione non lascia clip troncate con il nome finale
            tmp = dst.parent / f"{dst.stem}.tmp.mp4"
            dst.parent.mkdir(parents=True, exist_ok=True)
            cut_clip(src, start, stop, tmp)
            tmp.replace(dst)

        n_frames = max(1, round((stop - start) * fps))
        records.append({
            "narration_id": r["narration_id"],
            "participant_id": r["participant_id"],
            "video_id": video_id,
            "start_s": f"{start:.3f}",
            "stop_s": f"{stop:.3f}",
            "fps": fps,
            "n_frames": n_frames,
            "frame_indices": " ".join(map(str, uniform_indices(n_frames, k))),
            "clip_path": str(dst),
        })

    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    return len(records)


def load_rows(splits, participants):
    """Legge le righe di retrieval dei split indicati, solo per i partecipanti scelti."""
    rows = []
    for split in splits:
        path = ANN / "retrieval_annotations" / f"EPIC_100_retrieval_{split}.csv"
        with open(path, newline="", encoding="utf-8") as f:
            rows += [r for r in csv.DictReader(f) if r["participant_id"] in participants]
    return rows


def main():
    """Punto di ingresso: seleziona le sessioni richieste e le taglia."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", nargs="+", default=["train", "test"], choices=["train", "test"])
    parser.add_argument("--participant", nargs="+", default=SUBSET, choices=SUBSET)
    parser.add_argument("--session", help="processa solo questa sessione, es. P22_01")
    parser.add_argument(
        "--clips", type=int,
        help="processa solo N clip scelte a caso tra tutte (per prove, es. 200)")
    parser.add_argument("--seed", type=int, default=0, help="seme del campione di --clips")
    parser.add_argument("--limit", type=int, help="numero massimo di sessioni da processare")
    parser.add_argument("--frames", type=int, default=8, help="frame campionati per clip (8 o 16)")
    parser.add_argument("--out", type=Path, default=DATA.parent / "clips")
    args = parser.parse_args()

    with open(ANN / "EPIC_100_video_info.csv", newline="", encoding="utf-8") as f:
        fps_by_video = {r["video_id"]: float(r["fps"]) for r in csv.DictReader(f)}

    rows = load_rows(args.split, args.participant)
    if args.clips:
        # campione casuale con seme fisso, poi ordine originale del CSV
        picked = set(random.Random(args.seed).sample(range(len(rows)), args.clips))
        rows = [r for i, r in enumerate(rows) if i in picked]
    sessions = {}
    for r in rows:
        sessions.setdefault(r["video_id"], []).append(r)
    video_ids = sorted(sessions)
    if args.session:
        video_ids = [v for v in video_ids if v == args.session]
    if args.limit:
        video_ids = video_ids[: args.limit]

    total = 0
    for i, video_id in enumerate(video_ids, 1):
        n = process_session(
            video_id, sessions[video_id], fps_by_video[video_id], args.out, args.frames)
        total += n
        print(f"[{i}/{len(video_ids)}] {video_id}: {n} clip")
    print(f"fatto: {total} clip processate")


if __name__ == "__main__":
    main()
