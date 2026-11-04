"""Grafici di una run di training (Fase P3), letti dal suo CSV di log.

Quattro pannelli, ognuno con la sua scala:
1. loss di train e di val per epoca: dove si separano, il modello inizia a memorizzare;
2. R@10 esatto sul val, con la baseline lineare (ridge) come riferimento;
3. R@10 classe sul val, idem: e' il criterio di early stopping;
4. temperatura tau.
La linea verticale tratteggiata e' l'epoca scelta (migliore R@10 classe sul val).

Uso: python src/evaluation/plot_training.py <nome run>   (es. mpnet_h1024_d256_..._s0)
Output: figures/p3_train_<nome run>.png
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=wrong-import-position

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "experiments/logs"
RIDGE_LOG = LOG_DIR / "linear_alignment.csv"
FIG_DIR = ROOT / "figures"
CRITERION = "classe_R@10"
ENCODERS = ("bert", "mpnet", "clip")

# colori: serie 1 e 2 della palette di riferimento; inchiostri neutri per testo e assi
TRAIN, VAL = "#2a78d6", "#eb6834"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def read_log(path):
    """Righe del CSV di training, con i valori numerici convertiti in float."""
    with open(path, newline="", encoding="utf-8") as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def ridge_reference(encoder):
    """Riga migliore della baseline lineare sul val per questo encoder, se esiste."""
    if encoder not in ENCODERS or not RIDGE_LOG.exists():
        return None
    with open(RIDGE_LOG, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["encoder"] == encoder]
    return max(rows, key=lambda r: float(r[CRITERION])) if rows else None


def style(ax, title, ylabel):
    """Assi e griglia discreti, titolo e etichette in inchiostro neutro."""
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel("epoca", color=INK_MUTED)
    ax.set_ylabel(ylabel, color=INK_MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)


def line(ax, epochs, values, color, label):
    """Una serie: linea di 2 px, punti piccoli, etichetta diretta in fondo alla linea."""
    ax.plot(epochs, values, color=color, linewidth=2, marker="o", markersize=3, label=label)
    ax.annotate(label, (epochs[-1], values[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=9, color=INK)


def reference(ax, value, text):
    """Linea orizzontale tratteggiata grigia per un valore di riferimento."""
    ax.axhline(value, color=INK_MUTED, linewidth=1, linestyle="--")
    ax.annotate(text, (0, value), xycoords=("axes fraction", "data"), xytext=(4, 4),
                textcoords="offset points", fontsize=9, color=INK_MUTED)


def main():
    """Legge il log della run e salva la figura a quattro pannelli."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", help="nome della run (il CSV e' experiments/logs/train_<nome>.csv)")
    args = parser.parse_args()

    rows = read_log(LOG_DIR / f"train_{args.name}.csv")
    epochs = [int(r["epoca"]) for r in rows]
    best = max(rows, key=lambda r: r[CRITERION])           # stessa scelta di train.py
    ridge = ridge_reference(args.name.split("_")[0])

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    (ax_loss, ax_exact), (ax_class, ax_tau) = axes

    line(ax_loss, epochs, [r["loss_train"] for r in rows], TRAIN, "train")
    line(ax_loss, epochs, [r["loss_val"] for r in rows], VAL, "val")
    ax_loss.legend(frameon=False, fontsize=9, labelcolor=INK)
    style(ax_loss, "Loss InfoNCE (batch da 256)", "loss")

    line(ax_exact, epochs, [r["esatto_R@10"] for r in rows], VAL, "val")
    style(ax_exact, "R@10 esatto, testo → video (val)", "%")

    line(ax_class, epochs, [r[CRITERION] for r in rows], VAL, "val")
    style(ax_class, "R@10 classe, testo → video (val) — criterio", "%")

    if ridge:
        lam = float(ridge["lambda"])
        reference(ax_exact, float(ridge["esatto_R@10"]), f"ridge λ={lam:g}")
        reference(ax_class, float(ridge[CRITERION]), f"ridge λ={lam:g}")

    line(ax_tau, epochs, [r["tau"] for r in rows], TRAIN, "τ")
    style(ax_tau, "Temperatura τ", "τ")

    for ax in axes.flat:
        ax.axvline(best["epoca"], color=INK_MUTED, linewidth=1, linestyle=":")
        ax.set_xlim(0.5, epochs[-1] + 1.5)                  # spazio per le etichette dirette
    ax_class.annotate(f"epoca scelta: {int(best['epoca'])}", (best["epoca"], 1),
                      xycoords=("data", "axes fraction"), xytext=(4, -4),
                      textcoords="offset points", va="top", fontsize=9, color=INK_MUTED)

    fig.suptitle(args.name, x=0.01, ha="left", fontsize=10, color=INK_MUTED)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"p3_train_{args.name}.png"
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"salvato: {out}")


if __name__ == "__main__":
    main()
