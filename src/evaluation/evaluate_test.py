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

Encoder di testo (--encoder): MPNet, il default, e' il modello finale di P6. Con --encoder bert
si valuta BERT con la stessa config (P5, fatto dopo P6: confronto informativo, il modello finale
resta MPNet). Caso, baseline e ridge si ricalcolano con le feature dell'encoder; i checkpoint MLP
sono experiments/final/bert_{diag,classe}_s{0,1,2}.ckpt, copie dei checkpoint di train.py
verificate contro l'originale. Non sono nel manifest, che resta quello di P6.

Uso:
    python src/evaluation/evaluate_test.py --split val     # passo 3: verifica, il test non si tocca
    python src/evaluation/evaluate_test.py --split test    # passo 4: una sola volta
    python src/evaluation/evaluate_test.py --encoder bert --split val    # P5, idem per BERT
    python src/evaluation/evaluate_test.py --encoder bert --split test

Output (per BERT gli stessi nomi con suffisso _bert, es. final_test_bert.csv):
- val:  experiments/final/check_val.csv, piu' il confronto con i valori noti del val a schermo
  (MLP: manifest per MPNet, log del training per BERT; ridge: log del val)
- test: experiments/final/final_test.csv e experiments/final/final_test_ranks.npz
  (ranghi per query di ogni modello e maschere zero-shot, per figure e analisi qualitativa).
  Se final_test.csv esiste gia' lo script si ferma: il test si valuta una volta sola.
  In piu' una run wandb "test-finale" ("test-bert" per BERT; gruppo "test", job_type "eval")
  nello stesso progetto del training: la tabella del CSV e, nel summary, le metriche principali
  di ogni modello (media sui seed). Fonte ufficiale resta il CSV su git; --wandb-mode disabled
  la salta.
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
    CRITERION, OUT as RIDGE_LOG, best_val_lambdas, ridge, standardize, unseen_mask)
from evaluation.metrics import class_relevance, exact_relevance, first_relevant_rank  # noqa: E402
from training.task import RetrievalTask  # noqa: E402
from training.train import CKPT_DIR, LOG_DIR, WANDB_DIR, WANDB_PROJECT  # noqa: E402
# standardize di linear_alignment (numpy, per il ridge) e' un'altra funzione: alias distinto
from utils.dataset import standardize as standardize_with_stats  # noqa: E402
from utils.features import l2_normalize, load_pair, write_csv  # noqa: E402

FINAL = ROOT / "experiments/final"
MANIFEST = FINAL / "manifest.json"
OUT = {"val": FINAL / "check_val.csv", "test": FINAL / "final_test.csv"}
OUT_RANKS = FINAL / "final_test_ranks.npz"
ENCODER = "mpnet"                   # encoder del modello finale (P6)
ENCODERS = ("mpnet", "bert")        # valutabili con --encoder
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


def outputs(encoder):
    """File di output: per MPNet quelli di P6, per gli altri encoder con suffisso _<encoder>."""
    if encoder == ENCODER:
        return {**OUT, "ranks": OUT_RANKS}
    return {"val": FINAL / f"check_val_{encoder}.csv", "test": FINAL / f"final_test_{encoder}.csv",
            "ranks": FINAL / f"final_test_ranks_{encoder}.npz"}


def load_models(encoder):
    """Checkpoint MLP da valutare: per MPNet il manifest, per gli altri encoder le copie in final/.

    Gli altri encoder usano la config del modello finale: per ogni voce del manifest c'e'
    <encoder>_<positives>_s<seed>.ckpt, copia del checkpoint di train.py della stessa run con
    l'encoder cambiato (es. mpnet_h1024_..._s0 -> bert_h1024_..._s0). Si verifica che la copia
    sia identica all'originale; i valori di val per check_val si leggono dal log del training
    all'epoca migliore (la prima con il massimo del criterio, come il checkpoint salvato).
    """
    manifest = load_manifest()
    if encoder == ENCODER:
        return manifest
    models = []
    for m in manifest:
        run = encoder + m["run"].removeprefix(ENCODER)
        file = f"{encoder}_{m['positives']}_s{m['seed']}.ckpt"
        source = CKPT_DIR / f"{run}.ckpt"
        if not (FINAL / file).exists():
            raise SystemExit(f"manca {FINAL / file}: copialo da {source}")
        if sha256(FINAL / file) != sha256(source):
            raise SystemExit(f"{file}: diverso da {source.name}, checkpoint cambiato")
        with open(LOG_DIR / f"train_{run}.csv", newline="", encoding="utf-8") as f:
            best = max(csv.DictReader(f), key=lambda r: float(r[CRITERION]))  # primo massimo
        models.append({"file": file, "positives": m["positives"], "seed": m["seed"], "run": run,
                       "val": {k: float(best[k]) for k in m["val"]}})
    print(f"{encoder}: {len(models)} checkpoint, identici agli originali di train.py")
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


def baseline_sims(split, encoder=ENCODER):
    """Similarita' della baseline per ogni seed, come baseline_zero_shot.py (feature L2)."""
    video, text, _ = load_pair(encoder, split)
    for seed in range(BASELINE_SEED, BASELINE_SEED + N_SEEDS):
        zv = proiezione_casuale(video, PROJ_DIM, seed)
        zt = proiezione_casuale(text, PROJ_DIM, seed + TEXT_SEED_OFFSET)
        yield seed, zt @ zv.T


def ridge_sim(split, lam, encoder=ENCODER):
    """Similarita' del ridge, come linear_alignment.py: L2, standardizzazione del train, W."""
    video_tr, text_tr, _ = load_pair(encoder, "train")
    video, text, _ = load_pair(encoder, split)
    text_tr, text = standardize(text_tr, text)
    video_tr, video = standardize(video_tr, video)
    w = ridge(text_tr, video_tr, lam)
    return l2_normalize(text @ w) @ l2_normalize(video).T


@torch.no_grad()
def mlp_embed(model_info, video, text):
    """Embedding (z_v, z_t) di un checkpoint MLP in experiments/final, normalizzati L2.

    load_from_checkpoint ricostruisce RetrievalTask dagli iperparametri salvati e carica pesi e
    buffer: tra questi media e deviazione del train usate in training.
    """
    task = RetrievalTask.load_from_checkpoint(FINAL / model_info["file"], map_location="cpu")
    task.eval()                                             # niente dropout
    # standardizzazione con media e dev. del train salvate nel checkpoint
    # (std contiene gia' il +EPS di utils/dataset.py), stessa formula del training
    v = standardize_with_stats(torch.from_numpy(video).float(), task.stats("video"))
    t = standardize_with_stats(torch.from_numpy(text).float(), task.stats("text"))
    z_v, z_t = task(v, t)                                   # gia' normalizzati L2 dalle teste
    return z_v.numpy(), z_t.numpy()


def mlp_sim(model_info, video, text):
    """Similarita' coseno frasi x video di un checkpoint MLP, con le statistiche del train."""
    z_v, z_t = mlp_embed(model_info, video, text)
    # prodotto in torch come prima del refactor: stessi numeri bit per bit di final_test.csv
    return (torch.from_numpy(z_t) @ torch.from_numpy(z_v).T).numpy()


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


def evaluate_all(split, encoder=ENCODER):
    """Ranghi e metriche di tutti i modelli di D1 sullo split indicato, per un encoder."""
    models = load_models(encoder)
    _, _, meta_tr = load_pair(encoder, "train")
    video, text, meta = load_pair(encoder, split, normalize=False)
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
                 for seed, sim in baseline_sims(split, encoder)]
    rows += aggregate(base_rows, "baseline")

    lam = best_val_lambdas([encoder])[encoder]
    add("ridge", "", f"lambda={lam:g} (dal val)", ridge_sim(split, lam, encoder))

    for positives in ("diag", "classe"):
        seed_rows = [add(f"mlp_{positives}", m["seed"], m["file"], mlp_sim(m, video, text))
                     for m in models if m["positives"] == positives]
        rows += aggregate(seed_rows, f"mlp_{positives}")
    return rows, all_ranks, masks, meta, models


def check_val(rows, models, encoder=ENCODER):
    """Passo 3: le righe del val devono riprodurre i valori noti delle MLP (manifest per MPNet,
    log del training per gli altri encoder) e il log del val del ridge."""
    ok = True
    for m in models:
        row = next(r for r in rows if r["modello"] == f"mlp_{m['positives']}"
                   and r["seed"] == m["seed"])
        diff = {k: (row[k], v) for k, v in m["val"].items() if abs(row[k] - v) > 1e-6}
        ok &= not diff
        print(f"  {m['file']:20} {'OK' if not diff else f'DIVERSO {diff}'}")
    lam = best_val_lambdas([encoder])[encoder]
    with open(RIDGE_LOG, newline="", encoding="utf-8") as f:
        logged = next(r for r in csv.DictReader(f)
                      if r["encoder"] == encoder and float(r["lambda"]) == lam)
    row = next(r for r in rows if r["modello"] == "ridge")
    diff = {k: (row[k], float(v)) for k, v in logged.items()
            if k in row and abs(row[k] - float(v)) > 1e-6}
    ok &= not diff
    print(f"  {f'ridge λ={lam:g}':20} {'OK' if not diff else f'DIVERSO {diff}'}")
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


def log_to_wandb(rows, models, mode, encoder=ENCODER):
    """Run wandb di valutazione: tabella completa e metriche principali nel summary.

    Nel summary una voce per modello: caso e ridge (una sola run), baseline e MLP con la media
    sui seed (es. "mlp_classe/classe_R@10"); le dev. std e i singoli seed sono nella tabella.
    """
    import wandb  # pylint: disable=import-outside-toplevel

    name = "test-finale" if encoder == ENCODER else f"test-{encoder}"
    run = wandb.init(project=WANDB_PROJECT, name=name, group="test", job_type="eval",
                     dir=str(WANDB_DIR), mode=mode,
                     config={"encoder": encoder,
                             "checkpoint": {m["file"]: m["run"] for m in models}})
    columns = list(rows[0])
    # seed misto (numeri, "media", "std"): come testo, wandb.Table vuole un tipo per colonna
    data = [[str(r[c]) if c in ("seed", "dettagli") else r[c] for c in columns] for r in rows]
    run.log({"final_test": wandb.Table(columns=columns, data=data)})
    main_metrics = ("esatto_R@1", "esatto_R@5", "esatto_R@10", "esatto_rango_mediano",
                    "classe_R@10", "nuove_classe_R@10")
    for r in rows:
        if r["modello"] in ("caso", "ridge") or r["seed"] == "media":
            for k in main_metrics:
                run.summary[f"{r['modello']}/{k}"] = r[k]
    run.finish()


def main():
    """Valuta tutti i modelli sullo split scelto e salva i risultati."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", required=True, choices=["val", "test"])
    parser.add_argument("--encoder", default=ENCODER, choices=ENCODERS,
                        help="encoder di testo (default mpnet, il modello finale di P6)")
    parser.add_argument("--overwrite", action="store_true",
                        help="riscrive final_test.csv se esiste (solo per rigenerare lo stesso "
                             "risultato, mai dopo aver cambiato qualcosa)")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"],
                        help="run wandb del test (solo con --split test); disabled: niente wandb")
    args = parser.parse_args()
    out = outputs(args.encoder)
    if args.split == "test" and out["test"].exists() and not args.overwrite:
        raise SystemExit(f"{out['test']} esiste gia': il test si valuta una volta sola")

    rows, ranks, masks, meta, models = evaluate_all(args.split, args.encoder)
    write_csv(out[args.split], rows)
    print_summary(rows)
    print(f"\nsalvato: {out[args.split]}")

    if args.split == "val":
        print("\nconfronto con i valori noti del val:")
        check_val(rows, models, args.encoder)
    else:
        np.savez_compressed(out["ranks"], narration_id=meta["narration_id"],
                            **{f"mask_{k}": v for k, v in masks.items()}, **ranks)
        print(f"salvato: {out['ranks']}")
        if args.wandb_mode != "disabled":
            log_to_wandb(rows, models, args.wandb_mode, args.encoder)


if __name__ == "__main__":
    main()
