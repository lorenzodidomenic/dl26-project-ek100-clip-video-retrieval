"""Analisi dei risultati sul test per verbo e per frequenza della classe (P7). Solo descrittiva.

Non rilancia nessun modello: legge i ranghi per query salvati da evaluate_test.py
(experiments/final/final_test_ranks.npz) e i metadati dello split. Metrica unica: classe R@10,
cioe' % di query con almeno un video della stessa classe verbo+nome tra i primi 10.
Modelli: ridge lineare (λ=1000, riferimento) e MLP con loss classe (modello finale), seed 0 come
fissato in D4 di docs/roadmap_p6_test.md. Stessi colori di p6_figures.py.

Scelte fissate prima di guardare i risultati (dipendono solo dai metadati):
1. figures/p7_recall_per_verbo.png: classe R@10 per i N_VERBS verbi piu' frequenti del test,
   ordinati per frequenza. Linea tratteggiata = classe R@10 del modello finale su tutte le query.
   Domanda: su quali azioni il modello va bene e su quali male?
2. figures/p7_recall_per_frequenza.png: query divise per quante clip della loro classe
   verbo+nome ci sono nel train (FREQ_BINS: 0, 1-10, 11-100, oltre 100). Il gruppo "0" e' il
   sottoinsieme "combinazione nuova" di D3. Domanda: quanto conta aver visto la classe nel train?

Uso: python src/evaluation/p7_analysis.py
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
# pylint: disable=wrong-import-position
from evaluation.evaluate_test import OUT_RANKS  # noqa: E402
from evaluation.p6_figures import style  # noqa: E402
from evaluation.p7_tsne import top_verbs, verb_names  # noqa: E402
from evaluation.plot_training import INK, INK_MUTED, plt  # noqa: E402
from utils.features import load_split  # noqa: E402

FIG_DIR = ROOT / "figures"
MLP_SEED = 0                    # D4: le figure usano il seed 0
K = 10
N_VERBS = 12                    # verbi del test con almeno ~200 query
# (etichetta, minimo, massimo incluso; None = senza limite): clip della classe nel train
FREQ_BINS = [("mai vista", 0, 0), ("rara", 1, 10), ("media", 11, 100), ("frequente", 101, None)]
MODELS = [  # (chiave nel file dei ranghi, etichetta, colore): come in p6_figures.py
    ("ridge", "ridge lineare (λ=1000)", "#1baf7a"),
    (f"mlp_classe_s{MLP_SEED}", "MLP, loss classe (finale)", "#2a78d6"),
]
FINAL_MODEL = MODELS[-1]


def recall_at_k(ranks):
    """% di query con il primo video giusto entro i primi K."""
    return float((ranks <= K).mean() * 100)


def test_classes(ids):
    """Verbo e nome di ogni query del test, nello stesso ordine dei ranghi salvati."""
    rows = {r["narration_id"]: r for r in load_split("test")}
    verbs = np.array([int(rows[i]["verb_class"]) for i in ids])
    nouns = np.array([int(rows[i]["noun_class"]) for i in ids])
    return verbs, nouns


def train_frequency(verbs, nouns):
    """Per ogni query: quante clip del train hanno la sua stessa classe verbo+nome."""
    counts = Counter((int(r["verb_class"]), int(r["noun_class"])) for r in load_split("train"))
    return np.array([counts[(v, n)] for v, n in zip(verbs, nouns)])


def bin_masks(freq):
    """Una maschera di query per ogni gruppo di FREQ_BINS."""
    return [(freq >= lo) & (freq <= (hi if hi is not None else freq.max()))
            for _, lo, hi in FREQ_BINS]


def bar_values(ax, xs, values, horizontal):
    """Scrive il valore accanto (orizzontali) o sopra (verticali) ogni barra."""
    for x, v in zip(xs, values):
        if horizontal:                                      # sfondo bianco: la linea di riferimento
            ax.text(v + 0.8, x, f"{v:.1f}", va="center", fontsize=8, color=INK, zorder=4,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5})
        else:
            ax.text(x, v + 0.8, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5, color=INK)


def plot_per_verb(data, verbs, out):
    """Figura 1: barre orizzontali di classe R@10 per verbo, una barra per modello."""
    names = verb_names()
    top = top_verbs(verbs, N_VERBS)
    width = 0.38
    y = np.arange(len(top))[::-1]                           # verbo piu' frequente in alto
    fig, ax = plt.subplots(figsize=(10, 7.5), constrained_layout=True)
    table = {}
    for j, (key, label, color) in enumerate(MODELS):
        ranks = data[f"{key}_classe"]
        values = [recall_at_k(ranks[verbs == v]) for v in top]
        table[label] = values
        pos = y + (0.5 - j) * width                         # primo modello sopra
        ax.barh(pos, values, width, color=color, label=label, zorder=2)
        bar_values(ax, pos, values, horizontal=True)
    overall = recall_at_k(data[f"{FINAL_MODEL[0]}_classe"])
    ax.axvline(overall, color=INK_MUTED, linestyle="--", linewidth=1.2, zorder=3,
               label=f"MLP finale, tutte le query ({overall:.1f})")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{names[v]}  (n={int((verbs == v).sum())})" for v in top],
                       fontsize=9.5, color=INK)
    ax.set_xlim(0, 105)
    style(ax, f"Classe R@{K} per verbo: i {N_VERBS} verbi più frequenti del test",
          f"classe R@{K} (%)", "", grid_axis="x")
    # legenda sotto il grafico: dentro coprirebbe le barre
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=3,
               fontsize=9, frameon=False, labelcolor=INK)
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    return top, table


def plot_per_frequency(data, masks, out):
    """Figura 2: classe R@10 per gruppo di frequenza della classe nel train."""
    width = 0.36
    x = np.arange(len(FREQ_BINS))
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    table = {}
    for j, (key, label, color) in enumerate(MODELS):
        ranks = data[f"{key}_classe"]
        values = [recall_at_k(ranks[m]) for m in masks]
        table[label] = values
        pos = x + (j - 0.5) * width
        ax.bar(pos, values, width, color=color, label=label, zorder=2)
        bar_values(ax, pos, values, horizontal=False)
    labels = []
    for (name, lo, hi), m in zip(FREQ_BINS, masks):
        span = "0" if hi == 0 else (f"{lo}–{hi}" if hi is not None else f"oltre {lo - 1}")
        labels.append(f"{name}\n{span} clip nel train\n(n={int(m.sum())} query)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, color=INK)
    ax.set_ylim(0, 105)
    style(ax, f"Classe R@{K} del test in base a quante volte la classe verbo+nome compare nel "
          "train", "", f"classe R@{K} (%)", grid_axis="y")
    ax.legend(loc="upper left", fontsize=9, frameon=False, labelcolor=INK)
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    return table


def print_table(title, rows, table):
    """Tabella a schermo: una riga per gruppo, una colonna per modello."""
    print(f"\n{title}")
    print(f"  {'':<34}" + "".join(f"{label[:26]:>28}" for label in table))
    for i, row in enumerate(rows):
        print(f"  {row:<34}" + "".join(f"{values[i]:>28.2f}" for values in table.values()))


def main():
    """Disegna le due figure e stampa i numeri che le riassumono."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = np.load(OUT_RANKS)
    verbs, nouns = test_classes(data["narration_id"])

    out = FIG_DIR / "p7_recall_per_verbo.png"
    top, table = plot_per_verb(data, verbs, out)
    names = verb_names()
    print_table(f"classe R@{K} per verbo (test, seed {MLP_SEED})",
                [f"{names[v]} (n={int((verbs == v).sum())})" for v in top], table)
    print(f"salvato: {out}")

    freq = train_frequency(verbs, nouns)
    masks = bin_masks(freq)
    # controllo: il gruppo "mai vista" deve coincidere con le combinazioni nuove di D3
    if not np.array_equal(masks[0], data["mask_combinazione_nuova"]):
        raise SystemExit("il gruppo 'mai vista' non coincide con mask_combinazione_nuova")
    out = FIG_DIR / "p7_recall_per_frequenza.png"
    table = plot_per_frequency(data, masks, out)
    print_table(f"classe R@{K} per frequenza della classe nel train (test, seed {MLP_SEED})",
                [f"{name} (n={int(m.sum())})" for (name, *_), m in zip(FREQ_BINS, masks)],
                table)
    print(f"salvato: {out}")


if __name__ == "__main__":
    main()
