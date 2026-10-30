"""Confronto delle strutture video e testo con la CKA lineare. Nessun training.

Il coseno diretto tra un vettore TimeSformer e un vettore testo non ha significato
(spazi addestrati separatamente, a volte con dimensioni diverse). La CKA confronta invece
le relazioni interne: se due clip si somigliano nel video, le loro frasi si somigliano
anche nel testo? 0 = strutture scorrelate, 1 = stessa struttura.

Per ogni text encoder, sulle stesse clip:
- CKA(video, testo)
- controllo: CKA(video, testo con le righe mescolate), cioe' il valore di fondo
- CKA(testo, etichette verbo) e CKA(testo, etichette nome)
Una volta sola, come riferimento:
- CKA(video, etichette verbo) e CKA(video, etichette nome): il video cattura piu' azioni o oggetti?

Output: tabella a schermo e experiments/logs/representation_similarity.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from utils.features import load_pair  # noqa: E402

ENCODERS = ["bert", "mpnet", "clip"]
OUT = ROOT / "experiments/logs/representation_similarity.csv"


def linear_cka(X, Y):
    # X (n, d1), Y (n, d2): stesse n clip, dimensioni anche diverse
    X = X.astype(np.float64) - X.mean(0)
    Y = Y.astype(np.float64) - Y.mean(0)
    return np.linalg.norm(Y.T @ X) ** 2 / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y))


def one_hot(labels):
    # etichette -> matrice (n, classi): una "rappresentazione" che contiene solo la classe
    _, index = np.unique(labels, return_inverse=True)
    return np.eye(index.max() + 1)[index]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoder", nargs="+", default=ENCODERS, choices=ENCODERS)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--perms", type=int, default=5, help="mescolamenti per il controllo")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    results = []
    video_ref = None
    for name in args.encoder:
        video, text, meta = load_pair(name, args.split)
        verb, noun = one_hot(meta["verb_class"]), one_hot(meta["noun_class"])
        if video_ref is None:
            video_ref = {"video_verbo": linear_cka(video, verb), "video_nome": linear_cka(video, noun)}

        control = [linear_cka(video, text[rng.permutation(len(text))]) for _ in range(args.perms)]
        results.append({
            "encoder": name, "split": args.split, "clip": len(text),
            "cka_video_testo": round(linear_cka(video, text), 4),
            "controllo_media": round(float(np.mean(control)), 4),
            "controllo_std": round(float(np.std(control)), 4),
            "cka_testo_verbo": round(linear_cka(text, verb), 4),
            "cka_testo_nome": round(linear_cka(text, noun), 4),
        })

    print(f"split: {args.split} ({results[0]['clip']} clip)")
    print(f"riferimento video: CKA con verbo {video_ref['video_verbo']:.4f} | CKA con nome {video_ref['video_nome']:.4f}\n")
    print(f"{'encoder':8}{'video-testo':>13}{'controllo':>18}{'testo-verbo':>13}{'testo-nome':>12}")
    for r in results:
        print(f"{r['encoder']:8}{r['cka_video_testo']:>13.4f}{r['controllo_media']:>11.4f}±{r['controllo_std']:<6.4f}"
              f"{r['cka_testo_verbo']:>13.4f}{r['cka_testo_nome']:>12.4f}")

    for r in results:
        r.update({k: round(v, 4) for k, v in video_ref.items()})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nsalvato: {OUT}")


if __name__ == "__main__":
    main()
