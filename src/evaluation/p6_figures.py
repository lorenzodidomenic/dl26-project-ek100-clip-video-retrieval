"""Figure della valutazione finale sul test (P6, passo 6 di docs/roadmap_p6_test.md).

Figure solo descrittive: non ne segue nessuna scelta. Seed fissati prima del test (D4):
MLP seed 0, baseline seed 42 (il primo dei suoi 10).

1. figures/p6_recall_k.png: curva R@K al variare di K (da 1 a N, asse logaritmico), due pannelli
   (esatto, classe). R@K = % di query con il primo video giusto entro i primi K: a K = 1, 5, 10 la
   curva vale le R@1/5/10 della tabella. Baseline ("prima del training"), ridge, MLP diag e MLP
   classe ("dopo"). Legge i ranghi gia' salvati in experiments/final/final_test_ranks.npz.

2. figures/p6_coseni.png: distribuzione del coseno testo-video su tutte le coppie del test, in tre
   gruppi: coppia esatta, stessa classe (stesso verbo e nome, clip diversa: include anche le frasi
   identiche), classe diversa. Due pannelli, MLP diag e MLP classe. Istogrammi normalizzati
   (densita'), perche' i gruppi hanno dimensioni molto diverse. Le similarita' non sono salvate
   (circa 93 milioni per modello): si ricalcolano dai checkpoint con mlp_sim di evaluate_test.py.

Uso: python src/evaluation/p6_figures.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
# pylint: disable=wrong-import-position
from evaluation.evaluate_test import ENCODER, OUT_RANKS, load_manifest, mlp_sim  # noqa: E402
from evaluation.metrics import class_relevance  # noqa: E402
from evaluation.plot_training import GRID, INK, INK_MUTED, plt  # noqa: E402
from utils.features import load_pair  # noqa: E402

FIG_DIR = ROOT / "figures"
MLP_SEED, BASELINE_SEED = 0, 42
KS = (1, 5, 10)

# figura 1: un colore per modello, slot 1-3 della palette di riferimento (validata);
# la baseline e' un riferimento, quindi grigio tratteggiato
MODELS = [  # (chiave nel file dei ranghi, etichetta, colore, stile)
    (f"baseline_s{BASELINE_SEED}", "baseline casuale (prima del training)", INK_MUTED, "--"),
    ("ridge", "ridge lineare (λ=1000)", "#1baf7a", "-"),
    (f"mlp_diag_s{MLP_SEED}", "MLP, loss diag", "#eb6834", "-"),
    (f"mlp_classe_s{MLP_SEED}", "MLP, loss classe (finale)", "#2a78d6", "-"),
]
# figura 2: tre gruppi di coppie, tinte diverse da quelle dei modelli della figura 1
GROUPS = [  # (nome, colore)
    ("coppia esatta", "#6250d6"),
    ("stessa classe (non esatta)", "#e87ba4"),
    ("classe diversa", "#a3a29e"),
]
BINS = np.linspace(-1, 1, 161)


def style(ax, title, xlabel, ylabel):
    """Assi e griglia discreti, testo in inchiostro neutro (come plot_training.py)."""
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=INK_MUTED)
    ax.set_ylabel(ylabel, color=INK_MUTED)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)


def recall_curve(ranks):
    """R@K in % per K = 1..N: frazione di query con il primo giusto a rango <= K."""
    n = len(ranks)
    return np.bincount(ranks, minlength=n + 1)[1:].cumsum() / n * 100


def plot_recall_k(data, out):
    """Figura 1: curve R@K per i quattro modelli, pannelli esatto e classe."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True, constrained_layout=True)
    for ax, tag in zip(axes, ("esatto", "classe")):
        for k in KS:                                        # i tagli della tabella
            ax.axvline(k, color=GRID, linewidth=1.2, zorder=0)
        for key, label, color, ls in MODELS:
            curve = recall_curve(data[f"{key}_{tag}"])
            k_axis = np.arange(1, len(curve) + 1)
            ax.plot(k_axis, curve, color=color, linestyle=ls, linewidth=2,
                    drawstyle="steps-post", label=label)
            # punto a K = 10: e' la R@10 della tabella
            ax.plot(10, curve[9], "o", color=color, markersize=6, zorder=3,
                    markeredgecolor="white", markeredgewidth=1.5)
        ax.set_xscale("log")
        ax.set_xlim(1, len(curve))
        ax.set_ylim(0, 101)
        ax.set_xticks([1, 5, 10, 100, 1000, len(curve)])
        ax.set_xticklabels(["1", "5", "10", "100", "1000", f"N={len(curve)}"])
        style(ax, f"R@K testo → video, giusto = {tag}", "K (scala logaritmica)",
              "R@K (%)" if tag == "esatto" else "")
    # una legenda sola sotto i pannelli: e' la stessa nei due e cosi' non copre le curve
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=4,
               fontsize=9, frameon=False, labelcolor=INK)
    fig.suptitle(f"Test ({len(curve)} query): % di query con il primo video giusto entro i "
                 f"primi K. MLP seed {MLP_SEED}, baseline seed {BASELINE_SEED}",
                 x=0.01, ha="left", fontsize=10, color=INK_MUTED)
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def cosine_groups(sim, same_class):
    """Coseni delle coppie divise nei tre gruppi: esatta, stessa classe non esatta, diversa."""
    exact = np.eye(len(sim), dtype=bool)
    return {"coppia esatta": sim[exact],
            "stessa classe (non esatta)": sim[same_class & ~exact],
            "classe diversa": sim[~same_class]}


def plot_cosines(groups_by_model, out):
    """Figura 2: densita' del coseno per gruppo di coppie, un pannello per modello."""
    fig, axes = plt.subplots(1, len(groups_by_model), figsize=(13, 5.0), sharex=True,
                             sharey=True, constrained_layout=True)
    for ax, (title, groups) in zip(axes, groups_by_model.items()):
        means = []
        for name, color in GROUPS:
            values = groups[name]
            density, edges = np.histogram(values, bins=BINS, density=True)
            centers = (edges[:-1] + edges[1:]) / 2
            if name == "classe diversa":                    # lo sfondo: area grigia
                ax.fill_between(centers, density, color=color, alpha=0.35, linewidth=0)
            n_pairs = f"{len(values):,}".replace(",", ".")
            ax.plot(centers, density, color=color, linewidth=2, label=f"{name} (n={n_pairs})")
            ax.axvline(values.mean(), color=color, linewidth=1.2, linestyle="--")
            means.append(f"{name}: {values.mean():.2f}")
        # le medie (diverse nei due pannelli) in alto a sinistra, dove non ci sono curve
        ax.text(0.02, 0.97, "medie del coseno\n" + "\n".join(means), transform=ax.transAxes,
                va="top", fontsize=8.5, color=INK, linespacing=1.5)
        style(ax, title, "coseno testo–video", "densità" if ax is axes[0] else "")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3,
               fontsize=9, frameon=False, labelcolor=INK)
    fig.suptitle(f"Test: coseno tra frase e video per tipo di coppia (seed {MLP_SEED}). "
                 "Tratteggio: media del gruppo", x=0.01, ha="left", fontsize=10, color=INK_MUTED)
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def main():
    """Disegna le due figure e stampa i numeri che le riassumono."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = np.load(OUT_RANKS)
    plot_recall_k(data, FIG_DIR / "p6_recall_k.png")
    print(f"salvato: {FIG_DIR / 'p6_recall_k.png'}")

    video, text, meta = load_pair(ENCODER, "test", normalize=False)
    same_class = class_relevance(meta["verb_class"], meta["noun_class"])
    groups_by_model = {}
    for m in load_manifest():                               # verifica anche gli sha256
        if m["seed"] != MLP_SEED:
            continue
        groups = cosine_groups(mlp_sim(m, video, text), same_class)
        groups_by_model[f"MLP, loss {m['positives']}"] = groups
        print(f"\nMLP {m['positives']} (seed {m['seed']}): coseno per gruppo")
        for name, values in groups.items():
            print(f"  {name:28} media {values.mean():6.3f}  mediana {np.median(values):6.3f}  "
                  f"dev. std {values.std():5.3f}  n {len(values):>11,}")
    order = ["MLP, loss diag", "MLP, loss classe"]
    plot_cosines({k: groups_by_model[k] for k in order}, FIG_DIR / "p6_coseni.png")
    print(f"\nsalvato: {FIG_DIR / 'p6_coseni.png'}")


if __name__ == "__main__":
    main()
