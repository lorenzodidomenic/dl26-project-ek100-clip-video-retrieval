"""Scarica, taglia ed estrae le feature di TUTTE le sessioni di retrieval, un blocco alla volta.

Il disco non basta per tutti i video (~1.1 TB), quindi per ogni blocco di sessioni (--batch-gb):
1. scarica i video con parallel_download (pezzi in parallelo, md5 verificato);
2. ricontrolla l'md5 del video, taglia le clip (cut_clips) ed estrae le feature (extract_features);
3. solo se la sessione ha una feature per ogni clip delle annotazioni cancella clip e video;
4. aggiorna data/features/timesformer_k400_8f/timesformer_k400_8f.npz con tutte le sessioni fatte.

Mentre un blocco viene tagliato ed estratto, il successivo si scarica: sul disco ci sono al massimo
due blocchi di video. Le sessioni gia' complete (es. P22, P24, P27, P30) vengono saltate e i loro
video non vengono toccati. Si puo' interrompere (Ctrl+C) e rilanciare: riprende da dove era.

Uso:
    python src/datasets/full_pipeline.py --limit 2 --keep-videos   # prova su 2 sessioni
    caffeinate -i python src/datasets/full_pipeline.py              # tutto
"""
import argparse
import contextlib
import csv
import hashlib
import io
import os
import queue
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "epic-kitchens-download-scripts"))
# pylint: disable=wrong-import-position,import-error
import cut_clips  # noqa: E402
import extract_features  # noqa: E402
import parallel_download  # noqa: E402

ROOT = cut_clips.ROOT
FEATURES = ROOT / "data/features/timesformer_k400_8f"
SESSIONS = FEATURES / "sessions"
FINAL = FEATURES / "timesformer_k400_8f.npz"
CLIPS_OUT = ROOT / "data/clips"
GB = 1e9


def log(msg):
    """Stampa con l'ora davanti, subito (l'output finisce spesso in un file di log)."""
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def annotation_sessions():
    """Righe di retrieval train+test di tutti i partecipanti, raggruppate per video_id."""
    rows = []
    for split in ["train", "test"]:
        path = cut_clips.ANN / "retrieval_annotations" / f"EPIC_100_retrieval_{split}.csv"
        with open(path, newline="", encoding="utf-8") as f:
            rows += list(csv.DictReader(f))
    sessions = {}
    for r in rows:
        sessions.setdefault(r["video_id"], []).append(r)
    return sessions


def make_batches(video_ids, sizes, batch_bytes):
    """Divide le sessioni in blocchi consecutivi di al massimo batch_bytes (un video troppo grande sta da solo)."""
    batches, current, used = [], [], 0
    for v in video_ids:
        if current and used + sizes[v] > batch_bytes:
            batches.append(current)
            current, used = [], 0
        current.append(v)
        used += sizes[v]
    if current:
        batches.append(current)
    return batches


def missing_bytes(batch, jobs, sizes):
    """Byte ancora da scrivere su disco per i video del blocco (stima per eccesso: ignora i pezzi gia' scaricati)."""
    return sum(sizes[v] for v in batch
               if not (os.path.exists(jobs[v][1]) and os.path.getsize(jobs[v][1]) == sizes[v]))


def md5_ok(path, expected):
    """True se il file ha l'md5 atteso (o se l'md5 non e' noto)."""
    if expected is None:
        return True
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(16 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest() == expected


def downloader(batches, jobs, sizes, md5, args, ready, state):
    """Thread che scarica i blocchi uno dopo l'altro, al massimo uno in anticipo e solo se c'e' spazio."""
    try:
        for batch in batches:
            need = missing_bytes(batch, jobs, sizes) + args.margin_gb * GB
            with state["cv"]:
                # aspetta che il consumatore liberi spazio; se non c'e' niente da liberare, il disco e' troppo piccolo
                while state["in_flight"] >= 2 or (
                        shutil.disk_usage(ROOT).free < need and state["in_flight"] > 0):
                    state["cv"].wait(60)
                free = shutil.disk_usage(ROOT).free
                if free < need:
                    raise RuntimeError(f"spazio insufficiente: servono {need / GB:.1f} GB, "
                                       f"liberi {free / GB:.1f} GB (riduci --batch-gb o libera spazio)")
                state["in_flight"] += 1
            log(f"download blocco di {len(batch)} sessioni ({missing_bytes(batch, jobs, sizes) / GB:.1f} GB)")
            failed = parallel_download.download_all([jobs[v] for v in batch], md5,
                                                    args.workers, args.chunk_mb)
            ready.put((batch, failed))
        ready.put(None)
    except Exception as e:  # pylint: disable=broad-except
        ready.put(e)


def process_batch(batch, failed, ctx, args):
    """Taglia, estrae, verifica e pulisce le sessioni di un blocco. Ritorna le sessioni con problemi."""
    problems = []
    for video_id in batch:
        rows = ctx["sessions"][video_id]
        url, video = ctx["jobs"][video_id]
        if any(video in f for f in failed) or not os.path.exists(video):
            problems.append(f"{video_id}: download non riuscito")
            continue
        if not md5_ok(video, ctx["md5"].get(url)):
            os.replace(video, video + ".badmd5")
            problems.append(f"{video_id}: md5 sbagliato (rinominato in .badmd5)")
            continue

        t0 = time.time()
        cut_clips.process_session(video_id, rows, ctx["fps"][video_id], CLIPS_OUT, args.frames)
        with open(CLIPS_OUT / "manifest" / f"{video_id}.csv", newline="", encoding="utf-8") as f:
            manifest = list(csv.DictReader(f))
        t1 = time.time()
        out = SESSIONS / f"{video_id}.npz"
        if ctx["encoder"] is None:  # caricato al primo uso, una volta sola
            ctx["encoder"] = extract_features.TimesformerEncoder()
        extract_features.process_session(ctx["encoder"], manifest, args.batch, out)
        t2 = time.time()

        # si cancella solo se la sessione ha una feature per ogni clip delle annotazioni
        if not extract_features.is_complete(out, len(rows)):
            problems.append(f"{video_id}: feature incomplete, video e clip tenuti")
            continue
        for r in rows:
            (CLIPS_OUT / "clips" / f"{r['narration_id']}.mp4").unlink(missing_ok=True)
        if not args.keep_videos:
            os.remove(video)
        ctx["done"] += 1
        log(f"[{ctx['done']}/{ctx['todo']}] {video_id}: {len(rows)} clip | "
            f"taglio {(t1 - t0) / 60:.1f} min, feature {(t2 - t1) / 60:.1f} min")
    return problems


def main():
    """Punto di ingresso: elabora a blocchi tutte le sessioni senza feature complete."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-gb", type=float, default=10, help="GB di video per blocco")
    # le clip tagliate pesano ~60% del video: per il video piu' grande (13.9 GB) ~8.4 GB
    parser.add_argument("--margin-gb", type=float, default=12,
                        help="spazio da lasciare libero oltre ai video (clip tagliate, pezzi, sistema)")
    parser.add_argument("--workers", type=int, default=32, help="connessioni di download simultanee")
    parser.add_argument("--chunk-mb", type=int, default=128)
    parser.add_argument("--frames", type=int, default=8, help="frame per clip (come cut_clips.py)")
    parser.add_argument("--batch", type=int, default=8, help="clip per batch del TimeSformer")
    parser.add_argument("--limit", type=int, help="elabora solo le prime N sessioni mancanti (prova)")
    parser.add_argument("--keep-videos", action="store_true", help="non cancellare i video (prova)")
    args = parser.parse_args()

    sessions = annotation_sessions()
    with open(cut_clips.ANN / "EPIC_100_video_info.csv", newline="", encoding="utf-8") as f:
        fps = {r["video_id"]: float(r["fps"]) for r in csv.DictReader(f)}
    SESSIONS.mkdir(parents=True, exist_ok=True)
    todo = [v for v in sorted(sessions)
            if not extract_features.is_complete(SESSIONS / f"{v}.npz", len(sessions[v]))]
    log(f"sessioni: {len(sessions)} | gia' complete: {len(sessions) - len(todo)} | da fare: {len(todo)}")
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        return

    with contextlib.redirect_stdout(io.StringIO()):  # EpicDownloader stampa la lista di tutti i video
        job_list, md5 = parallel_download.collect_jobs(str(ROOT / "data"), "all", "all", todo)
    jobs = {Path(out).stem: (url, out) for url, out in job_list}
    if set(jobs) != set(todo):
        raise RuntimeError(f"video senza URL: {sorted(set(todo) - set(jobs))[:5]}")
    log("lettura delle dimensioni dei video...")
    with parallel_download.ThreadPoolExecutor(32) as ex:
        sizes = dict(zip(todo, ex.map(lambda v: parallel_download.remote_size(jobs[v][0]), todo)))
    batches = make_batches(todo, sizes, args.batch_gb * GB)
    log(f"{sum(sizes.values()) / GB:.0f} GB da scaricare in {len(batches)} blocchi")

    ctx = {"sessions": sessions, "fps": fps, "jobs": jobs, "md5": md5,
           "encoder": None, "done": 0, "todo": len(todo)}
    ready = queue.Queue()
    state = {"cv": threading.Condition(), "in_flight": 0}
    threading.Thread(target=downloader, daemon=True,
                     args=(batches, jobs, sizes, md5, args, ready, state)).start()

    problems = []
    try:
        while (item := ready.get()) is not None:
            if isinstance(item, Exception):
                raise item
            batch, failed = item
            problems += process_batch(batch, failed, ctx, args)
            n = extract_features.merge_sessions(SESSIONS, FINAL)
            log(f"aggiornato {FINAL.name}: {n} feature")
            with state["cv"]:
                state["in_flight"] -= 1
                state["cv"].notify_all()
    except KeyboardInterrupt:
        # i thread di download non si fermano da soli: si esce subito, tutto e' riprendibile
        log("interrotto: rilancia lo stesso comando per riprendere")
        os._exit(1)

    if problems:
        print("Problemi (rilancia lo script per riprovare queste sessioni):", *problems, sep="\n  ")
        sys.exit(1)
    log(f"fatto: {np.load(FINAL)['narration_id'].shape[0]} feature in {FINAL}")


if __name__ == "__main__":
    main()
