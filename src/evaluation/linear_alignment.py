"""Allineamento lineare testo -> video e retrieval, per scegliere il text encoder.

Per ogni text encoder:
1. sul train impara una matrice W che porta il testo nello spazio video (ridge regression,
   soluzione in forma chiusa, nessuna epoca);
2. sul val trasforma ogni frase in un "video previsto" e lo confronta con tutti i video del val;
3. misura R@1/5/10 e rango mediano testo -> video, in due versioni:
   - esatto: giusto solo il video con lo stesso narration_id;
   - classe: giusto ogni video con lo stesso verbo e lo stesso nome;
   e le stesse metriche solo sulle clip del val con combinazioni verbo/nome assenti dal train.

λ si sceglie su una griglia, per ogni encoder, con il criterio di decisione (R@10 classe sul val).
Sceglierlo sul val e' un po' ottimista, ma allo stesso modo per tutti gli encoder.
La riga "caso" usa similarita' casuali: e' il valore del retrieval senza alcuna informazione.

Con --split test non si sceglie nulla: per ogni encoder si usa il λ migliore sul val (dal log di val)
e si valuta una sola volta sul test, come riferimento.

Output: tabella a schermo e experiments/logs/linear_alignment.csv (tutte le combinazioni encoder x λ),
oppure experiments/logs/linear_alignment_test.csv con --split test.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from evaluation.metrics import class_relevance, exact_relevance, retrieval_metrics  # noqa: E402
from utils.features import l2_normalize, load_pair  # noqa: E402

ENCODERS = ["bert", "mpnet", "clip"]
LAMBDAS = [1, 10, 100, 1000, 10000, 100000]
CRITERION = "classe_R@10"
OUT = ROOT / "experiments/logs/linear_alignment.csv"
OUT_TEST = ROOT / "experiments/logs/linear_alignment_test.csv"


def standardize(train, other):
    # media e deviazione solo dal train, applicate anche al val
    mean, std = train.mean(0), train.std(0) + 1e-8
    return (train - mean) / std, (other - mean) / std


def ridge(T, V, lam):
    # W = argmin ||T W - V||^2 + λ ||W||^2  ->  W = (T'T + λI)^-1 T'V
    T, V = T.astype(np.float64), V.astype(np.float64)
    return np.linalg.solve(T.T @ T + lam * np.eye(T.shape[1]), T.T @ V)


def unseen_mask(meta_train, meta_val):
    seen = set(zip(meta_train["verb_class"], meta_train["noun_class"]))
    return np.array([c not in seen for c in zip(meta_val["verb_class"], meta_val["noun_class"])])


def evaluate(sim, meta, unseen):
    exact = exact_relevance(len(sim))
    same_class = class_relevance(meta["verb_class"], meta["noun_class"])
    result = {}
    for tag, relevant in [("esatto", exact), ("classe", same_class)]:
        # tutte le query, e solo quelle con combinazioni nuove (sempre cercando tra tutti i video del val)
        for k, v in retrieval_metrics(sim, relevant).items():
            result[f"{tag}_{k}"] = round(v, 2)
        for k, v in retrieval_metrics(sim[unseen], relevant[unseen]).items():
            result[f"nuove_{tag}_{k}"] = round(v, 2)
    return result


def print_table(rows):
    print(f"{'encoder':8}{'λ':>8}{'esatto R@1/5/10':>22}{'classe R@1/5/10':>22}{'rango classe':>14}{'nuove classe R@10':>19}")
    for r in rows:
        print(f"{r['encoder']:8}{str(r['lambda']):>8}"
              f"{r['esatto_R@1']:>8.2f}{r['esatto_R@5']:>7.2f}{r['esatto_R@10']:>7.2f}"
              f"{r['classe_R@1']:>8.2f}{r['classe_R@5']:>7.2f}{r['classe_R@10']:>7.2f}"
              f"{r['classe_rango_mediano']:>14.0f}{r['nuove_classe_R@10']:>19.2f}")


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"salvato: {path}")


def best_val_lambdas():
    # λ migliore sul val, letto dal log di val: il test non entra nella scelta
    with open(OUT, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["encoder"] in ENCODERS]
    best = {}
    for name in ENCODERS:
        candidates = [r for r in rows if r["encoder"] == name]
        best[name] = float(max(candidates, key=lambda r: float(r[CRITERION]))["lambda"])
    return best


def run_test(encoders, seed):
    lambdas = best_val_lambdas()
    rows = []
    for name in encoders:
        video_tr, text_tr, meta_tr = load_pair(name, "train")
        video_te, text_te, meta_te = load_pair(name, "test")
        text_tr, text_te = standardize(text_tr, text_te)
        video_tr, video_te = standardize(video_tr, video_te)
        unseen = unseen_mask(meta_tr, meta_te)
        target = l2_normalize(video_te)

        lam = lambdas[name]
        W = ridge(text_tr, video_tr, lam)
        sim = l2_normalize(text_te @ W) @ target.T          # (frasi test, video test)
        rows.append({"encoder": name, "lambda": lam, **evaluate(sim, meta_te, unseen)})
        print(f"{name:6} λ={lam:<9g} (scelto sul val) {CRITERION} = {rows[-1][CRITERION]:.2f}")

    # riferimento: similarita' casuali, stesse clip del test
    rng = np.random.default_rng(seed)
    chance = {"encoder": "caso", "lambda": "", **evaluate(rng.random((len(meta_te["verb_class"]),) * 2), meta_te, unseen)}
    rows.append(chance)

    print(f"\ntest: {len(meta_te['verb_class'])} clip, di cui {int(unseen.sum())} con combinazioni nuove")
    print_table(rows)
    write_rows(OUT_TEST, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoder", nargs="+", default=ENCODERS, choices=ENCODERS)
    parser.add_argument("--lambdas", nargs="+", type=float, default=LAMBDAS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", default="val", choices=["val", "test"],
                        help="test: usa il λ scelto sul val, una sola valutazione (ignora --lambdas)")
    args = parser.parse_args()

    if args.split == "test":
        run_test(args.encoder, args.seed)
        return

    rows, best = [], {}
    for name in args.encoder:
        video_tr, text_tr, meta_tr = load_pair(name, "train")
        video_va, text_va, meta_va = load_pair(name, "val")
        text_tr, text_va = standardize(text_tr, text_va)
        video_tr, video_va = standardize(video_tr, video_va)
        unseen = unseen_mask(meta_tr, meta_va)
        target = l2_normalize(video_va)

        for lam in args.lambdas:
            W = ridge(text_tr, video_tr, lam)
            sim = l2_normalize(text_va @ W) @ target.T          # (frasi val, video val)
            rows.append({"encoder": name, "lambda": lam, **evaluate(sim, meta_va, unseen)})
            print(f"{name:6} λ={lam:<9g} {CRITERION} = {rows[-1][CRITERION]:.2f}")

        candidates = [r for r in rows if r["encoder"] == name]
        best[name] = max(candidates, key=lambda r: r[CRITERION])
        if best[name]["lambda"] in (min(args.lambdas), max(args.lambdas)):
            print(f"   attenzione: per {name} il λ migliore e' al bordo della griglia, allargala")

    # riferimento: similarita' casuali, stesse clip del val
    rng = np.random.default_rng(args.seed)
    chance = {"encoder": "caso", "lambda": "", **evaluate(rng.random((len(meta_va["verb_class"]),) * 2), meta_va, unseen)}
    rows.append(chance)

    print(f"\nval: {len(meta_va['verb_class'])} clip, di cui {int(unseen.sum())} con combinazioni nuove")
    print_table([*best.values(), chance])

    winner = max(best.values(), key=lambda r: r[CRITERION])
    print(f"\nmigliore secondo {CRITERION}: {winner['encoder']} (λ={winner['lambda']:g})")

    write_rows(OUT, rows)


if __name__ == "__main__":
    main()
