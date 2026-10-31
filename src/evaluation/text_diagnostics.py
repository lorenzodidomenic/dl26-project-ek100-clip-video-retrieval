"""Diagnostica del solo spazio testuale, per ogni text encoder. Nessun training, nessun video.

- Anisotropia: coseno medio tra coppie di frasi scelte a caso. Vicino a 0 = frasi ben
  distribuite nello spazio; vicino a 1 = tutte nella stessa zona (il coseno distingue poco).
- Purezza dei vicini: per ogni frase, le k piu' vicine; quota con lo stesso verbo,
  lo stesso nome, entrambi. Dice se lo spazio e' gia' organizzato per verbo e nome.
  Accanto, lo stesso calcolo con vicini scelti a caso (il valore del caso).

Si lavora sulle frasi uniche di train + val (il test resta fuori), altrimenti le frasi
ripetute ("take plate" compare molte volte) gonfierebbero i numeri.

Output: tabella a schermo e experiments/logs/text_diagnostics.csv
"""
import argparse
import collections
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
# pylint: disable=wrong-import-position
from utils.features import TEXT_DIR, l2_normalize, load_split, write_csv  # noqa: E402

ENCODERS = ["bert", "mpnet", "clip"]
OUT = ROOT / "experiments/logs/text_diagnostics.csv"
EXAMPLES = ["take plate", "open fridge", "cut onion"]


def majority(rows, key):
    """Valore piu' frequente di una colonna intera tra le righe."""
    return collections.Counter(int(r[key]) for r in rows).most_common(1)[0][0]


def unique_sentences(encoder):
    """Frasi uniche di train + val con vettore normalizzato, verbo e nome."""
    # una riga per frase unica di train + val: vettore normalizzato, verbo e nome
    data = np.load(TEXT_DIR / f"{encoder}.npz")
    features = data["features"]
    index = {nid: i for i, nid in enumerate(data["narration_id"])}

    by_text = collections.defaultdict(list)
    for r in load_split("train") + load_split("val"):
        by_text[r["narration"]].append(r)
    texts = sorted(by_text)

    vectors = l2_normalize(features[[index[by_text[t][0]["narration_id"]] for t in texts]])
    # la stessa frase ha quasi sempre le stesse classi;
    # nei rari casi diversi vince la piu' frequente
    verb = np.array([majority(by_text[t], "verb_class") for t in texts])
    noun = np.array([majority(by_text[t], "noun_class") for t in texts])
    return texts, vectors, verb, noun


def anisotropy(vectors, n_pairs, rng):
    """Media e deviazione standard del coseno tra coppie casuali di vettori."""
    i = rng.integers(0, len(vectors), n_pairs)
    j = rng.integers(0, len(vectors), n_pairs)
    keep = i != j
    cos = (vectors[i[keep]] * vectors[j[keep]]).sum(1)
    return cos.mean(), cos.std()


def nearest(vectors, k):
    """Indici dei k vettori piu' vicini (coseno) di ogni vettore, senza se stesso."""
    sim = vectors @ vectors.T
    np.fill_diagonal(sim, -np.inf)                       # una frase non e' vicina di se' stessa
    return np.argsort(-sim, axis=1)[:, :k]               # (n, k): indici dei k piu' vicini


def purity(neighbors, verb, noun):
    """Quota dei vicini con stesso verbo, stesso nome e entrambi."""
    same_verb = verb[neighbors] == verb[:, None]
    same_noun = noun[neighbors] == noun[:, None]
    return same_verb.mean(), same_noun.mean(), (same_verb & same_noun).mean()


def main():  # pylint: disable=too-many-locals
    """Punto di ingresso: calcola diagnostiche per ogni encoder e salva il CSV."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoder", nargs="+", default=ENCODERS, choices=ENCODERS)
    parser.add_argument("--k", type=int, default=10, help="vicini per frase")
    parser.add_argument("--pairs", type=int, default=10000,
                        help="coppie casuali per l'anisotropia")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    results = []
    for name in args.encoder:
        rng = np.random.default_rng(args.seed)
        texts, vectors, verb, noun = unique_sentences(name)
        cos_mean, cos_std = anisotropy(vectors, args.pairs, rng)
        neighbors = nearest(vectors, args.k)
        verb_p, noun_p, both_p = purity(neighbors, verb, noun)
        random_neighbors = rng.integers(0, len(texts), neighbors.shape)
        verb_r, noun_r, both_r = purity(random_neighbors, verb, noun)

        results.append({
            "encoder": name, "frasi": len(texts), "k": args.k,
            "cos_medio": round(cos_mean, 3), "cos_std": round(cos_std, 3),
            "stesso_verbo": round(verb_p, 3), "stesso_nome": round(noun_p, 3),
            "entrambi": round(both_p, 3),
            "caso_verbo": round(verb_r, 3), "caso_nome": round(noun_r, 3),
            "caso_entrambi": round(both_r, 3),
        })

        print(f"== {name}: {len(texts)} frasi uniche")
        position = {t: i for i, t in enumerate(texts)}
        for example in EXAMPLES:
            if example in position:
                close = [texts[j] for j in neighbors[position[example]][:5]]
                print(f"   {example!r:15} -> {close}")

    print()
    print(f"{'encoder':8}{'cos medio':>11}{'stesso verbo':>14}{'stesso nome':>13}{'entrambi':>10}"
          f"   (caso: verbo / nome / entrambi)")
    for r in results:
        print(f"{r['encoder']:8}{r['cos_medio']:>8.3f}±{r['cos_std']:<4.2f}"
              f"{r['stesso_verbo']:>12.3f}{r['stesso_nome']:>13.3f}{r['entrambi']:>10.3f}"
              f"   ({r['caso_verbo']:.3f} / {r['caso_nome']:.3f} / {r['caso_entrambi']:.3f})")

    write_csv(OUT, results)
    print(f"\nsalvato: {OUT}")


if __name__ == "__main__":
    main()
