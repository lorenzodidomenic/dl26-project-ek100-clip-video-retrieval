"""Valutazione finale (P6): tutti i modelli, stesse metriche, una sola tabella.

Modelli valutati (D1):
- caso: esatto analitico (R@K = K/N, rango mediano (N+1)/2), classe con similarita' casuali;
- baseline: proiezioni lineari casuali non allenate, come baseline_zero_shot.py, media su 10 seed;
- ridge: λ letto dal log del val (experiments/logs/linear_alignment.csv), come linear_alignment.py;
- MLP diag e MLP classe: i 3 seed di experiments/final/manifest.json, con sha256 verificato.

Metriche (D2), testo -> video, ogni query cerca tra tutti i video dello split:
R@1/5/10 e rango mediano, "esatto" (stesso narration_id) e "classe" (stesso verbo e nome).
Sottoinsiemi zero-shot (D3), rispetto al train: combinazione verbo/nome nuova (prefisso
nuove_), nome mai visto (nomi_nuovi_), verbo mai visto (verbi_nuovi_).

Il rango di una query dipende solo dalla sua riga di similarita' (si cerca sempre tra tutti i
video), quindi si calcola una volta per query e le metriche dei sottoinsiemi si ottengono
selezionando le query: stesso risultato di evaluate() in linear_alignment.py.

Uso:
    python src/evaluation/evaluate_test.py --split val     # passo 3: verifica, il test non si tocca
    python src/evaluation/evaluate_test.py --split test    # passo 4: una sola volta

Output:
- val:  experiments/final/check_val.csv, piu' il confronto con manifest e log del ridge a schermo
- test: experiments/final/final_test.csv e experiments/final/final_test_ranks.npz
  (ranghi per query di ogni modello e maschere zero-shot, per figure e analisi qualitativa).
  Se final_test.csv esiste gia' lo script si ferma: il test si valuta una volta sola.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
# pylint: disable=wrong-import-position
from evaluation.baseline_zero_shot import (  # noqa: E402
    N_SEEDS, PROJ_DIM, SEED as BASELINE_SEED, TEXT_SEED_OFFSET, proiezione_casuale)
from evaluation.linear_alignment import (  # noqa: E402
    OUT as RIDGE_LOG, best_val_lambdas, ridge, standardize, unseen_mask)
from evaluation.metrics import class_relevance, exact_relevance, first_relevant_rank  # noqa: E402
from training.heads import DualHead  # noqa: E402
from utils.features import l2_normalize, load_pair, write_csv  # noqa: E402

FINAL = ROOT / "experiments/final"
MANIFEST = FINAL / "manifest.json"
OUT = {"val": FINAL / "check_val.csv", "test": FINAL / "final_test.csv"}
OUT_RANKS = FINAL / "final_test_ranks.npz"
ENCODER = "mpnet"
KS = (1, 5, 10)
CHANCE_SEED = 0
# sottoinsiemi di query: prefisso delle colonne -> chiave della maschera
SUBSETS = {"": "tutte", "nuove_": "combinazione_nuova", "nomi_nuovi_": "nome_nuovo",
           "verbi_nuovi_": "verbo_nuovo"}


def sha256(path):
    """Hash sha256 del file, per controllare che sia il checkpoint scelto sul val."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest():
    """Modelli del manifest, dopo aver verificato lo sha256 di ogni checkpoint.
        Per ogni pt ricalcola lo sh256 e se non coincide col manifest si ferma.
    """
    with open(MANIFEST, encoding="utf-8") as f:
        models = json.load(f)["modelli"]
    for m in models:
        if sha256(FINAL / m["file"]) != m["sha256"]:
            raise SystemExit(f"{m['file']}: sha256 diverso dal manifest, checkpoint cambiato")
    print(f"manifest: {len(models)} checkpoint, sha256 verificati")
    return models


def zero_shot_masks(meta_tr, meta):
    """Maschere delle query zero-shot rispetto al train, piu' quella di tutte le query."""
    return {
        "tutte": np.ones(len(meta["verb_class"]), dtype=bool),
        "combinazione_nuova": unseen_mask(meta_tr, meta),
        "nome_nuovo": ~np.isin(meta["noun_class"], meta_tr["noun_class"]),
        "verbo_nuovo": ~np.isin(meta["verb_class"], meta_tr["verb_class"]),
    }


def ranks_of(sim, relevance):
    """Rango (da 1) del primo video giusto per ogni query, per ogni definizione di giusto."""
    return {tag: first_relevant_rank(sim, rel) for tag, rel in relevance.items()}


def metrics_from_ranks(ranks, masks):
    """R@K e rango mediano per ogni definizione di giusto e ogni sottoinsieme di query."""
    out = {}
    for prefix, mask_name in SUBSETS.items():
        mask = masks[mask_name]
        for tag, r in ranks.items():
            sub = r[mask]
            for k in KS:
                out[f"{prefix}{tag}_R@{k}"] = round(float((sub <= k).mean() * 100), 2)
            out[f"{prefix}{tag}_rango_mediano"] = float(np.median(sub))
    return out


def chance_ranks(n, relevance):
    """Caso: classe con similarita' casuali; esatto lo si sostituisce col valore analitico."""
    rng = np.random.default_rng(CHANCE_SEED)
    return ranks_of(rng.random((n, n), dtype=np.float32), relevance)


def analytic_exact(n):
    """Valori attesi del retrieval esatto a caso: R@K = K/N, rango mediano = (N+1)/2."""
    out = {}
    for prefix in SUBSETS:
        out.update({f"{prefix}esatto_R@{k}": round(100.0 * k / n, 2) for k in KS})
        out[f"{prefix}esatto_rango_mediano"] = (n + 1) / 2
    return out


def baseline_sims(split):
    """Similarita' della baseline per ogni seed, come baseline_zero_shot.py (feature L2)."""
    video, text, _ = load_pair(ENCODER, split)
    for seed in range(BASELINE_SEED, BASELINE_SEED + N_SEEDS):
        zv = proiezione_casuale(video, PROJ_DIM, seed)
        zt = proiezione_casuale(text, PROJ_DIM, seed + TEXT_SEED_OFFSET)
        yield seed, zt @ zv.T


def ridge_sim(split, lam):
    """Similarita' del ridge, come linear_alignment.py: L2, standardizzazione del train, W."""
    video_tr, text_tr, _ = load_pair(ENCODER, "train")
    video, text, _ = load_pair(ENCODER, split)
    text_tr, text = standardize(text_tr, text)
    video_tr, video = standardize(video_tr, video)
    w = ridge(text_tr, video_tr, lam)
    return l2_normalize(text @ w) @ l2_normalize(video).T


@torch.no_grad()
def mlp_sim(model_info, video, text):
    """Similarita' coseno di un checkpoint MLP, con le statistiche del train salvate dentro."""
    ckpt = torch.load(FINAL / model_info["file"], map_location="cpu", weights_only=False)
    args, stats = ckpt["args"], ckpt["stats"]
    model = DualHead(video.shape[1], text.shape[1], args["hidden"], args["dim"], args["dropout"])
    model.load_state_dict(ckpt["model"])
    model.eval()                                            # niente dropout
    # standardizzazione con media e dev. del train (std contiene gia' il +1e-8 di train.py)
    v = (torch.from_numpy(video).float() - stats["video"][0]) / stats["video"][1]
    t = (torch.from_numpy(text).float() - stats["text"][0]) / stats["text"][1]
    z_v, z_t = model(v, t)                                  # gia' normalizzati L2 dalle teste
    return (z_t @ z_v.T).numpy()


def aggregate(rows, model):
    """Righe media e dev. std campionaria (ddof=1) sulle righe dei seed di un modello."""
    keys = [k for k in rows[0] if k not in ("modello", "seed", "dettagli")]
    mean = {"modello": model, "seed": "media", "dettagli": f"n={len(rows)}"}
    std = {"modello": model, "seed": "std", "dettagli": f"n={len(rows)}"}
    for k in keys:
        values = np.array([r[k] for r in rows], dtype=float)
        mean[k] = round(float(values.mean()), 2)
        std[k] = round(float(values.std(ddof=1)), 2)
    return [mean, std]


def evaluate_all(split):
    """Ranghi e metriche di tutti i modelli di D1 sullo split indicato."""
    models = load_manifest()
    _, _, meta_tr = load_pair(ENCODER, "train")
    video, text, meta = load_pair(ENCODER, split, normalize=False)
    n = len(video)
    relevance = {"esatto": exact_relevance(n),
                 "classe": class_relevance(meta["verb_class"], meta["noun_class"])}
    masks = zero_shot_masks(meta_tr, meta)
    print(f"{split}: {n} clip | query zero-shot: combinazione nuova "
          f"{masks['combinazione_nuova'].sum()}, nome nuovo {masks['nome_nuovo'].sum()}, "
          f"verbo nuovo {masks['verbo_nuovo'].sum()}")

    rows, all_ranks = [], {}

    def add(model, seed, details, sim):
        ranks = ranks_of(sim, relevance)
        name = model if seed == "" else f"{model}_s{seed}"   # es. mlp_classe_s0, ridge
        all_ranks.update({f"{name}_{tag}": r for tag, r in ranks.items()})
        row = {"modello": model, "seed": seed, "dettagli": details,
               **metrics_from_ranks(ranks, masks)}
        rows.append(row)
        print(f"  {model:11} seed {str(seed):>3}  esatto R@10 {row['esatto_R@10']:6.2f}  "
              f"classe R@10 {row['classe_R@10']:6.2f}")
        return row

    # caso: classe da similarita' casuali, esatto sostituito dal valore analitico
    caso = {"modello": "caso", "seed": CHANCE_SEED, "dettagli": "esatto analitico K/N",
            **metrics_from_ranks(chance_ranks(n, relevance), masks)}
    caso.update(analytic_exact(n))
    rows.append(caso)

    base_rows = [add("baseline", seed, f"proj_dim={PROJ_DIM}", sim)
                 for seed, sim in baseline_sims(split)]
    rows += aggregate(base_rows, "baseline")

    lam = best_val_lambdas([ENCODER])[ENCODER]
    add("ridge", "", f"lambda={lam:g} (dal val)", ridge_sim(split, lam))

    for positives in ("diag", "classe"):
        seed_rows = [add(f"mlp_{positives}", m["seed"], m["file"], mlp_sim(m, video, text))
                     for m in models if m["positives"] == positives]
        rows += aggregate(seed_rows, f"mlp_{positives}")
    return rows, all_ranks, masks, meta, models


def check_val(rows, models):
    """Passo 3: le righe del val devono riprodurre manifest (MLP) e log del val (ridge)."""
    ok = True
    for m in models:
        row = next(r for r in rows if r["modello"] == f"mlp_{m['positives']}"
                   and r["seed"] == m["seed"])
        diff = {k: (row[k], v) for k, v in m["val"].items() if abs(row[k] - v) > 1e-6}
        ok &= not diff
        print(f"  {m['file']:14} {'OK' if not diff else f'DIVERSO {diff}'}")
    lam = best_val_lambdas([ENCODER])[ENCODER]
    with open(RIDGE_LOG, newline="", encoding="utf-8") as f:
        logged = next(r for r in csv.DictReader(f)
                      if r["encoder"] == ENCODER and float(r["lambda"]) == lam)
    row = next(r for r in rows if r["modello"] == "ridge")
    diff = {k: (row[k], float(v)) for k, v in logged.items()
            if k in row and abs(row[k] - float(v)) > 1e-6}
    ok &= not diff
    print(f"  ridge λ={lam:g}    {'OK' if not diff else f'DIVERSO {diff}'}")
    print("verifica superata: si puo' passare al test" if ok
          else "verifica NON superata: non lanciare il test")


def print_summary(rows):
    """Tabella riassuntiva: caso, ridge e media (± dev. std sui seed) di baseline e MLP."""
    columns = {"esatto R@10": "esatto_R@10", "classe R@1": "classe_R@1",
               "classe R@10": "classe_R@10", "nuove classe R@10": "nuove_classe_R@10",
               "nomi nuovi classe R@10": "nomi_nuovi_classe_R@10"}
    print("\n" + f"{'modello':12}" + "".join(f"{c:>25}" for c in columns))
    std = {r["modello"]: r for r in rows if r["seed"] == "std"}
    for r in rows:
        # una riga per modello: caso e ridge hanno una sola run, gli altri la media sui seed
        if not (r["modello"] in ("caso", "ridge") or r["seed"] == "media"):
            continue
        s = std.get(r["modello"])
        cells = [f"{r[k]:.2f}" + (f" ± {s[k]:.2f}" if s else "") for k in columns.values()]
        print(f"{r['modello']:12}" + "".join(f"{c:>25}" for c in cells))


def main():
    """Valuta tutti i modelli sullo split scelto e salva i risultati."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", required=True, choices=["val", "test"])
    parser.add_argument("--overwrite", action="store_true",
                        help="riscrive final_test.csv se esiste (solo per rigenerare lo stesso "
                             "risultato, mai dopo aver cambiato qualcosa)")
    args = parser.parse_args()
    if args.split == "test" and OUT["test"].exists() and not args.overwrite:
        raise SystemExit(f"{OUT['test']} esiste gia': il test si valuta una volta sola")

    rows, ranks, masks, meta, models = evaluate_all(args.split)
    write_csv(OUT[args.split], rows)
    print_summary(rows)
    print(f"\nsalvato: {OUT[args.split]}")

    if args.split == "val":
        print("\nconfronto con i valori noti del val:")
        check_val(rows, models)
    else:
        np.savez_compressed(OUT_RANKS, narration_id=meta["narration_id"],
                            **{f"mask_{k}": v for k, v in masks.items()}, **ranks)
        print(f"salvato: {OUT_RANKS}")


if __name__ == "__main__":
    main()
