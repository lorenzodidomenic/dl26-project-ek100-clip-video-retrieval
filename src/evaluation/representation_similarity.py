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
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
# pylint: disable=wrong-import-position
from utils.features import load_pair, write_csv  # noqa: E402

ENCODERS = ["bert", "mpnet", "clip"]
OUT = ROOT / "experiments/logs/representation_similarity.csv"


def linear_cka(x, y):
    """CKA lineare tra x (n, d1) e y (n, d2), con le stesse n clip ma dimensioni anche diverse."""
    x = x.astype(np.float64) - x.mean(0)
    y = y.astype(np.float64) - y.mean(0)
    return np.linalg.norm(y.T @ x) ** 2 / (np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y))


def one_hot(labels):
    """Etichette -> matrice one-hot (n, classi)."""
    # etichette -> matrice (n, classi): una "rappresentazione" che contiene solo la classe
    _, index = np.unique(labels, return_inverse=True)
    return np.eye(index.max() + 1)[index]


def main():
    """Punto di ingresso: calcola le CKA per ogni encoder e salva il CSV."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
            video_ref = {
                "video_verbo": linear_cka(video, verb),
                "video_nome": linear_cka(video, noun),
            }

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
    print(f"riferimento video: CKA con verbo {video_ref['video_verbo']:.4f} | "
          f"CKA con nome {video_ref['video_nome']:.4f}\n")
    print(f"{'encoder':8}{'video-testo':>13}{'controllo':>18}{'testo-verbo':>13}{'testo-nome':>12}")
    for r in results:
        print(f"{r['encoder']:8}{r['cka_video_testo']:>13.4f}"
              f"{r['controllo_media']:>11.4f}±{r['controllo_std']:<6.4f}"
              f"{r['cka_testo_verbo']:>13.4f}{r['cka_testo_nome']:>12.4f}")

    for r in results:
        r.update({k: round(v, 4) for k, v in video_ref.items()})
    write_csv(OUT, results)
    print(f"\nsalvato: {OUT}")


if __name__ == "__main__":
    main()
