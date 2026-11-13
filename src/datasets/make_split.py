"""Crea lo split train/val/test per le clip di retrieval (default: tutti i partecipanti).

- test: tutte le righe di EPIC_100_retrieval_test.csv. Non si usano fino alla valutazione finale.
- val: sessioni intere prese dal train, una quota per ogni partecipante (seed fisso).
- train: il resto del train.

Sessioni intere e non clip a caso: clip vicine nella stessa sessione si somigliano,
e un val fatto di clip sparse darebbe risultati troppo ottimisti.

Output:
- data/splits/split.csv: una riga per clip, con split e metadati (narrazione, classi verbo/nome)
- data/splits/split_info.json: seed, quota e sessioni di val, per poter rifare lo stesso split
"""
import argparse
import collections
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# pylint: disable=wrong-import-position
from cut_clips import ANN, ROOT, load_rows  # noqa: E402

OUT_DIR = ROOT / "data/splits"
FIELDS = [
    "narration_id", "participant_id", "video_id", "split",
    "narration", "verb_class", "noun_class", "all_noun_classes",
]


def all_participants():
    """Partecipanti presenti nelle annotazioni di retrieval (train e test), in ordine."""
    found = set()
    for split in ["train", "test"]:
        path = ANN / "retrieval_annotations" / f"EPIC_100_retrieval_{split}.csv"
        with open(path, newline="", encoding="utf-8") as f:
            found.update(r["participant_id"] for r in csv.DictReader(f))
    return sorted(found)


def pick_val_sessions(train_rows, frac, seed):
    """Sceglie le sessioni di val: una quota per partecipante, con seed fisso."""
    # per ogni partecipante, una quota delle sue sessioni di train (almeno una)
    rng = random.Random(seed)
    sessions = collections.defaultdict(set)
    for r in train_rows:
        sessions[r["participant_id"]].add(r["video_id"])
    val = set()
    for p in sorted(sessions):
        candidates = sorted(sessions[p])
        n = max(1, round(frac * len(candidates)))
        val.update(rng.sample(candidates, n))
    return val


def report(rows):
    """Stampa clip per partecipante e split, sessioni e combinazioni verbo/nome nuove."""
    counts = collections.Counter((r["participant_id"], r["split"]) for r in rows)
    print(f"{'':6}{'train':>8}{'val':>8}{'test':>8}")
    for p in sorted({r["participant_id"] for r in rows}):
        print(f"{p:6}" + "".join(f"{counts[(p, s)]:>8}" for s in ["train", "val", "test"]))

    sessions = collections.defaultdict(set)
    for r in rows:
        sessions[r["split"]].add(r["video_id"])
    print("sessioni:", {s: len(v) for s, v in sessions.items()})

    # combinazioni verbo/nome assenti dal train: misurano la generalizzazione
    def combo(r):
        return (r["verb_class"], r["noun_class"])

    seen = {combo(r) for r in rows if r["split"] == "train"}
    for split in ["val", "test"]:
        part = [r for r in rows if r["split"] == split]
        unseen = [r for r in part if combo(r) not in seen]
        n_combos = len({combo(r) for r in part})
        n_unseen = len({combo(r) for r in unseen})
        print(f"{split}: combinazioni verbo/nome {n_combos}, assenti dal train {n_unseen} "
              f"({len(unseen)} clip su {len(part)})")


def main():
    """Punto di ingresso: crea split.csv e split_info.json per i partecipanti scelti."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--val-frac", type=float, default=0.15,
        help="quota di sessioni di train per il val")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--participant", nargs="+",
        help="solo questi partecipanti, es. P22 P24 (default: tutti quelli delle annotazioni)")
    args = parser.parse_args()

    participants = args.participant or all_participants()
    train_rows = load_rows(["train"], participants)
    test_rows = load_rows(["test"], participants)
    val_sessions = pick_val_sessions(train_rows, args.val_frac, args.seed)

    rows = []
    for r in train_rows:
        rows.append({**r, "split": "val" if r["video_id"] in val_sessions else "train"})
    for r in test_rows:
        rows.append({**r, "split": "test"})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "split.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with open(OUT_DIR / "split_info.json", "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "val_frac": args.val_frac, "participants": participants,
                   "val_sessions": sorted(val_sessions)}, f, indent=2)

    report(rows)
    print(f"salvato: {OUT_DIR / 'split.csv'}")


if __name__ == "__main__":
    main()
