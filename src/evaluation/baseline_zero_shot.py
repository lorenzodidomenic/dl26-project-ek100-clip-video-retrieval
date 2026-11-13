#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
baseline_zero_shot.py — Fase 3 (obiettivo minimo 1): baseline zero-shot.

Cosa fa, in tre passi:
  1. Carica le feature di TEST gia' estratte (video = TimeSformer, testo = MPNet)
     e le allinea per narration_id (mai per posizione). Usa già load_pair ("mpnet", "test")
  2. Le proietta con due matrici di pesi CASUALI e indipendenti (una per il video,
     una per il testo), mai allenate, e normalizza L2.
  3. Calcola la similarita' coseno tra ogni testo e ogni video del test set e
     misura R@1, R@5, R@10 e rango mediano testo->video, confrontandoli col valore
     atteso dal caso puro (R@K = K/N, rango mediano = (N+1)/2).
     Per la misurazione usa retrieval_metrics da metrics.py

Perche' deve uscire male: le due proiezioni casuali sono indipendenti, quindi
video e testo finiscono in direzioni scorrelate. Il ranking risultante e' al
livello del caso. Questo numero e' il "pavimento": senza di esso il R@K finale
del modello allenato (fase 5) non significa nulla.

Attenzione: video e testo hanno entrambi 768 dimensioni, quindi con lo stesso seed
le due matrici sarebbero IDENTICHE. Una proiezione casuale condivisa conserva circa
i prodotti scalari (Johnson-Lindenstrauss), e la similarita' diventerebbe il prodotto
scalare grezzo tra spazi non allineati: non casuale, e qui sistematicamente peggiore
del caso (rango mediano ~5600 invece di ~4834). Per questo il testo usa un seed diverso.

Con pochi successi (circa K query su N) un solo seed e' rumoroso: si ripete su
--n-seeds coppie di matrici e si riporta anche la media.

USO:
    python3 baseline_zero_shot.py
    python3 baseline_zero_shot.py --proj-dim 512 --seed 0 --n-seeds 20

Output: tabella a schermo e experiments/logs/baseline_zero_shot.csv
(una riga per seed, piu' la riga "media" e la riga "atteso" dal caso).
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
# pylint: disable=wrong-import-position
from evaluation.metrics import exact_relevance, retrieval_metrics
from utils.features import load_pair


# Proiezione casuale. La dimensione NON cambia il risultato (si normalizza L2).
PROJ_DIM = 256
SEED = 42       # per riproducibilita' delle matrici casuali
N_SEEDS = 10    # quante coppie di matrici casuali provare
TEXT_SEED_OFFSET = 1_000_000  # il testo usa seed + offset: mai la stessa matrice del video

# ============================================================================

KS = (1, 5, 10)     # ranghi che vengono testati
OUT = ROOT / "experiments/logs/baseline_zero_shot.csv"


def proiezione_casuale(x, dim_out, seed):
    """x [N, D] -> z [N, dim_out]: proiezione lineare casuale + normalizzazione L2.

    La scala dei pesi non conta: dopo la normalizzazione ogni vettore e'
    diviso per la propria norma, quindi una costante moltiplicativa si elide.
    """
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((dim_out, x.shape[1]))  # [dim_out, D]
    z = x @ w.T                                      # [N, dim_out]
    z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-12       # normalizzazione l2
    return z


def valori_attesi(n, ks):
    """Retrieval esatto a caso: R@K = K/N (in %), rango mediano = (N+1)/2."""
    attesi = {f"R@{k}": 100.0 * k / n for k in ks}
    attesi["rango_mediano"] = (n + 1) / 2
    return attesi


def main():
    """Punto di ingresso: calcola R@K della proiezione casuale e salva il CSV."""
    ap = argparse.ArgumentParser(description="Baseline zero-shot (proiezione casuale)")
    ap.add_argument("--proj-dim", type=int, default=PROJ_DIM)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-seeds", type=int, default=N_SEEDS)
    args = ap.parse_args()

    xv, xt, _ = load_pair("mpnet", "test")   # prendo feature video e feature testo
    n = xv.shape[0]
    relevant = exact_relevance(n)
    colonne = [f"R@{k}" for k in KS] + ["rango_mediano"]

    print(f"[test] N = {n} coppie testo-video, proj_dim = {args.proj_dim}")
    print(f"{'seed':>8}" + "".join(f"{c:>15}" for c in colonne))

    righe = []
    for seed in range(args.seed, args.seed + args.n_seeds):
        zv = proiezione_casuale(xv, args.proj_dim, seed)
        zt = proiezione_casuale(xt, args.proj_dim, seed + TEXT_SEED_OFFSET)
        sim = zt @ zv.T  # [N, N] coseno: riga = query testo, colonna = video candidato
        metriche = retrieval_metrics(sim, relevant, KS)
        righe.append({"proj_dim": args.proj_dim, "seed": seed, **metriche})
        print(f"{seed:>8}" + "".join(f"{metriche[c]:>15.4f}" for c in colonne))

    media = {c: float(np.mean([r[c] for r in righe])) for c in colonne}
    attesi = valori_attesi(n, KS)
    righe.append({"proj_dim": args.proj_dim, "seed": "media", **media})
    righe.append({"proj_dim": args.proj_dim, "seed": "atteso", **attesi})
    print(f"{'media':>8}" + "".join(f"{media[c]:>15.4f}" for c in colonne))
    print(f"{'atteso':>8}" + "".join(f"{attesi[c]:>15.4f}" for c in colonne))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(righe[0]))
        writer.writeheader()
        writer.writerows(righe)
    print(f"salvato: {OUT}")


if __name__ == "__main__":
    main()
