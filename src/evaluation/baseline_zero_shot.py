#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
baseline_zero_shot.py — Fase 3 (obiettivo minimo 1): baseline zero-shot.

Cosa fa, in tre passi:
  1. Carica le feature di TEST gia' estratte (video = TimeSformer, testo = MPNet)
     e le allinea per narration_id (mai per posizione). Usa già load_pair ("mpnet", "test")
  2. Le proietta con una matrice di pesi CASUALE, mai allenata, e normalizza L2.
  3. Calcola la similarita' coseno tra ogni testo e ogni video del test set,
     ordina e misura R@1, R@5, R@10 in entrambe le direzioni (testo->video,
     video->testo), confrontandole col valore atteso dal caso puro (K/N). Per la misutazione usa retrieval_metrics da metrics.py

Perche' deve uscire male: le due proiezioni casuali sono indipendenti, quindi
video e testo finiscono in direzioni scorrelate. Il ranking risultante e' al
livello del caso. Questo numero e' il "pavimento": senza di esso il R@K finale
del modello allenato (fase 5) non significa nulla.

Formati supportati per le feature:
  - .npz con chiavi "features" e "narration_id"          (es. mpnet.npz)
  - .npy + .json con la lista parallela di narration_id  (es. features_timesformer.npy
    + narration_ids_timesformer.json, prodotto da estrai_feature_timesformer.py)

USO:
    python3 baseline_zero_shot.py
    python3 baseline_zero_shot.py --proj-dim 512 --seed 0

Output: tabella a schermo e experiments/logs/baseline_zero_shot.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from evaluation.metrics import exact_relevance,retrieval_metrics  
from utils.features import load_pair  



# Proiezione casuale. La dimensione NON cambia il risultato (si normalizza L2).
PROJ_DIM = 256
SEED     = 42       # per riproducibilita' della matrice casuale

# ============================================================================

KS = (1, 5, 10)     # ranghi che vengono testati
OUT = ROOT / "experiments/logs/baseline_zero_shot.csv"


def proiezione_casuale(X, dim_out, seed):
    """X [N, D] -> Z [N, dim_out]: proiezione lineare casuale + normalizzazione L2.

    La scala dei pesi non conta: dopo la normalizzazione ogni vettore e'
    diviso per la propria norma, quindi una costante moltiplicativa si elide.
    """
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((dim_out, X.shape[1]))  # [dim_out, D]
    Z = X @ W.T                                      # [N, dim_out]
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12       #normalizzazione l2
    return Z

def main():
    ap = argparse.ArgumentParser(description="Baseline zero-shot (proiezione casuale)")
    ap.add_argument("--proj-dim", type=int, default=PROJ_DIM)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    Xv, Xt, meta = load_pair("mpnet", "test")   #prendo feature video e feature testo

    Zv = proiezione_casuale(Xv, args.proj_dim, args.seed)
    Zt = proiezione_casuale(Xt, args.proj_dim, args.seed)

    S = Zt @ Zv.T  # [N, N] coseno: riga = query testo, colonna = video candidato
    print(f"[proiezione] S = {S.shape}, similarita' min={S.min():.3f}  max={S.max():.3f}")

    metriche = retrieval_metrics(S, exact_relevance(S.shape[0]), KS)
    print(metriche)

    row = {"proj_dim": args.proj_dim, "seed": args.seed, **metriche}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(f"salvato: {OUT}")

if __name__ == "__main__":
    main()