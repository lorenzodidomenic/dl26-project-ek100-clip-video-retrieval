"""Figure della valutazione finale sul test (P6, passo 6 di docs/roadmap_p6_test.md).

Figure solo descrittive: non ne segue nessuna scelta. Seed fissati prima del test (D4):
MLP seed 0, baseline seed 42 (il primo dei suoi 10). Modelli e colori uguali in tutte le figure:
baseline casuale ("prima del training"), ridge, MLP diag, MLP classe ("dopo").

1. figures/p6_recall_k.png: curva R@K al variare di K (da 1 a N, asse logaritmico), due pannelli
   (esatto, classe). R@K = % di query con il primo video giusto entro i primi K: a K = 1, 5, 10 la
   curva vale le R@1/5/10 della tabella. Legge i ranghi salvati in final_test_ranks.npz.

2. figures/p6_risultati.png: la tabella finale come barre. Pannelli esatto e classe; per R@1, R@5
   e R@10 una barra per modello, con la dev. std sui seed come barra di errore (baseline 10 seed,
   MLP 3; il ridge non ha seed). Legge final_test.csv.

3. figures/p6_zero_shot.png: R@10 su tutte le query del test e sulle query con combinazione
   verbo+nome mai vista nel train (D3, metrica zero-shot principale). Pannelli esatto e classe,
   barre e barre di errore come sopra. Legge final_test.csv (i nomi mai visti, 24 query, restano
   solo in tabella: troppo pochi per una barra affidabile).

4. figures/p6_matrice_similarita.png: matrice frasi x video "stile CLIP" su 30 clip del test,
   ordinate per classe (6 classi verbo+nome x 5 clip). Righe = frasi, colonne = i video delle
   stesse clip nello stesso ordine: la diagonale (bordata) e' la coppia esatta, i quadrati lungo la
   diagonale le coppie della stessa classe. In ogni cella la probabilita' che la frase scelga quel
   video tra i 30: softmax della riga dei coseni con tau = 0.07, la tau fissa dei modelli finali,
   cioe' la stessa distribuzione che la loss InfoNCE ottimizza. Il coseno grezzo non basta: le
   differenze tra coppia esatta e stessa classe sono piccole in valore assoluto, ma decidono il
   rango. Due pannelli, MLP diag e MLP classe (seed 0), stessa scala. Mostra il trade-off della
   loss: diag concentra la probabilita' sulla diagonale, classe la distribuisce nel blocco.
   Scelta delle clip fissata prima del test e indipendente dai risultati (solo metadati, seed 0):
   classi con almeno 5 frasi diverse; 6 classi a caso; in ognuna 5 frasi diverse a caso, una
   clip per frase (frasi diverse: frasi identiche hanno lo stesso embedding e la diagonale non si
   distinguerebbe dal blocco per costruzione). Le similarita' si ricalcolano dai checkpoint con
   mlp_sim di evaluate_test.py.

Uso: python src/evaluation/p6_figures.py
"""
import csv
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
# pylint: disable=wrong-import-position
from evaluation.evaluate_test import (  # noqa: E402
    ENCODER, OUT as RESULTS, OUT_RANKS, by_model, load_manifest, mlp_sim)
from evaluation.plot_training import GRID, INK, INK_MUTED, clean_axes, plt  # noqa: E402
from utils.features import load_pair  # noqa: E402

FIG_DIR = ROOT / "figures"
MLP_SEED, BASELINE_SEED = 0, 42
KS = (1, 5, 10)
# matrice di similarita': quante classi, quante clip per classe, seed della scelta
MATRIX_CLASSES, MATRIX_PER_CLASS, MATRIX_SEED = 6, 5, 0
TAU = 0.07          # tau fissa dei modelli finali (manifest: fixed_tau), per la softmax di riga

# un colore per modello, slot 1-3 della palette di riferimento (validata);
# la baseline e' un riferimento, quindi grigio (tratteggiato nelle curve)
MODELS = [  # (chiave nel file dei ranghi, modello in final_test.csv, etichetta, colore, stile)
    (f"baseline_s{BASELINE_SEED}", "baseline", "baseline casuale (prima del training)",
     INK_MUTED, "--"),
    ("ridge", "ridge", "ridge lineare (λ=1000)", "#1baf7a", "-"),
    (f"mlp_diag_s{MLP_SEED}", "mlp_diag", "MLP, loss diag", "#eb6834", "-"),
    (f"mlp_classe_s{MLP_SEED}", "mlp_classe", "MLP, loss classe (finale)", "#2a78d6", "-"),
]
# scala sequenziale blu della palette di riferimento (step 100 -> 700), per la matrice;
# parte quasi dal bianco: probabilita' ~0 si confonde con lo sfondo
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#f7fafe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95",
              "#0d366b"])


def style(ax, title, xlabel, ylabel, grid_axis="both"):
    """Assi e griglia discreti, testo in inchiostro neutro (come plot_training.py)."""
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=INK_MUTED)
    ax.set_ylabel(ylabel, color=INK_MUTED)
    clean_axes(ax, grid_axis)


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
        for key, _, label, color, ls in MODELS:
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


def read_results(path):
    """Righe di final_test.csv per modello: {modello: (valori, dev. std o None)}.

    Baseline e MLP: riga "media" e riga "std" sui seed; ridge: la sua unica riga.
    """
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    def numbers(row):
        return {k: float(v) for k, v in row.items()
                if k not in ("modello", "seed", "dettagli") and v != ""}

    return by_model(rows, [model for _, model, *_ in MODELS], numbers)


def bar_panel(ax, results, groups, pad):
    """Barre raggruppate: un gruppo per colonna di groups, una barra per modello.

    groups: [(etichetta del gruppo, colonna di final_test.csv)]. Valore scritto sopra ogni barra,
    a distanza pad (in unita' dell'asse y) dalla cima della barra di errore.
    """
    width = 0.19
    offsets = (np.arange(len(MODELS)) - (len(MODELS) - 1) / 2) * (width + 0.02)
    x = np.arange(len(groups))
    for (_, model, label, color, _), off in zip(MODELS, offsets):
        values, std = results[model]
        heights = [values[col] for _, col in groups]
        errors = [std[col] for _, col in groups] if std else None
        ax.bar(x + off, heights, width, color=color, label=label, zorder=2)
        if errors:
            ax.errorbar(x + off, heights, yerr=errors, fmt="none", ecolor=INK,
                        elinewidth=1, capsize=3, zorder=3)
        for xi, h, e in zip(x + off, heights, errors or [0] * len(heights)):
            ax.text(xi, h + e + pad, f"{h:.1f}", ha="center", va="bottom", fontsize=7.5,
                    color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([g for g, _ in groups])


def plot_bars(results, panels, title, out):
    """Due pannelli di barre (esatto, classe) con la stessa legenda dei modelli."""
    fig, axes = plt.subplots(1, len(panels), figsize=(13, 5.0), constrained_layout=True)
    for ax, (panel_title, groups) in zip(axes, panels):
        top = max(results[m][0][c] for _, m, *_ in MODELS for _, c in groups)
        bar_panel(ax, results, groups, pad=top * 0.012)
        ax.set_ylim(0, top * 1.12)                          # spazio per i valori sopra le barre
        style(ax, panel_title, "", "%" if ax is axes[0] else "", grid_axis="y")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=4,
               fontsize=9, frameon=False, labelcolor=INK)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=INK_MUTED)
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def select_matrix_clips(meta):
    """Indici delle clip della matrice, ordinati per classe: dipende solo dai metadati.

    Classi (verbo, nome) con almeno MATRIX_PER_CLASS frasi diverse; MATRIX_CLASSES classi a caso;
    in ognuna MATRIX_PER_CLASS frasi diverse a caso e, per ogni frase, una clip a caso.
    """
    rng = np.random.default_rng(MATRIX_SEED)
    by_class = {}
    for i, key in enumerate(zip(meta["verb_class"], meta["noun_class"])):
        by_class.setdefault(key, {}).setdefault(meta["narration"][i], []).append(i)
    eligible = sorted(k for k, phrases in by_class.items() if len(phrases) >= MATRIX_PER_CLASS)
    chosen = rng.choice(len(eligible), MATRIX_CLASSES, replace=False)
    idx, class_of = [], []
    for c, j in enumerate(sorted(chosen)):
        phrases = sorted(by_class[eligible[j]])
        for p in rng.choice(phrases, MATRIX_PER_CLASS, replace=False):
            idx.append(int(rng.choice(by_class[eligible[j]][p])))
            class_of.append(c)
    return np.array(idx), np.array(class_of)


def row_softmax(sim, tau=TAU):
    """Per ogni frase (riga), probabilita' di scegliere ciascun video: softmax(coseno / tau)."""
    z = sim / tau
    z = np.exp(z - z.max(axis=1, keepdims=True))            # stabile: si sottrae il massimo
    return z / z.sum(axis=1, keepdims=True)


def plot_matrix(probs, labels, class_of, out):
    """Figura 4: probabilita' frase -> video, un pannello per modello, stessa scala."""
    n = len(labels)
    vmax = max(p.max() for p in probs.values())
    fig, axes = plt.subplots(1, len(probs), figsize=(15, 7.6), constrained_layout=True)
    bounds = np.flatnonzero(np.diff(class_of)) + 0.5        # confini tra classi
    for ax, (title, prob) in zip(axes, probs.items()):
        image = ax.imshow(prob, cmap=BLUES, vmin=0, vmax=vmax, interpolation="nearest")
        for b in bounds:                                    # blocchi delle classi
            ax.axhline(b, color="white", linewidth=2)
            ax.axvline(b, color="white", linewidth=2)
        for i in range(n):                                  # coppia esatta: cella bordata
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor=INK_MUTED, linewidth=0.8, zorder=3))
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels if ax is axes[0] else [], fontsize=7.5, color=INK)
        ax.set_xticks(range(n))
        ax.set_xticklabels([str(i + 1) for i in range(n)], fontsize=6.5, color=INK_MUTED)
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xlabel("video delle stesse clip, stesso ordine delle frasi", color=INK_MUTED)
    fig.colorbar(image, ax=axes, shrink=0.6,
                 label=f"probabilità che la frase scelga il video (softmax della riga, τ={TAU})")
    fig.suptitle(f"Test: per ogni frase, quale video sceglie tra {n} clip ({MATRIX_CLASSES} "
                 f"classi verbo+nome × {MATRIX_PER_CLASS}, seed {MLP_SEED}). Cella bordata = "
                 "coppia esatta, quadrati bianchi = stessa classe",
                 x=0.01, ha="left", fontsize=10, color=INK_MUTED)
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def matrix_summary(prob, class_of):
    """Probabilita' media per frase: sul video esatto, sul resto della classe, fuori classe."""
    exact = np.eye(len(prob), dtype=bool)
    same = class_of[:, None] == class_of[None, :]
    return {"video esatto": prob[exact].mean(),
            "altri video della classe": (prob * (same & ~exact)).sum(axis=1).mean(),
            "video di altre classi": (prob * ~same).sum(axis=1).mean()}


def main():
    """Disegna le figure e stampa i numeri che le riassumono."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = np.load(OUT_RANKS)
    n_query = len(data["narration_id"])
    n_new = int(data["mask_combinazione_nuova"].sum())

    plot_recall_k(data, FIG_DIR / "p6_recall_k.png")
    print(f"salvato: {FIG_DIR / 'p6_recall_k.png'}")

    results = read_results(RESULTS["test"])
    plot_bars(results,
              [(f"giusto = {tag}", [(f"R@{k}", f"{tag}_R@{k}") for k in KS])
               for tag in ("esatto", "classe")],
              f"Test ({n_query} query): R@1/5/10 testo → video. Barre di errore: dev. std sui "
              "seed (baseline 10, MLP 3)", FIG_DIR / "p6_risultati.png")
    print(f"salvato: {FIG_DIR / 'p6_risultati.png'}")

    plot_bars(results,
              [(f"R@10, giusto = {tag}", [(f"tutte le query\n(n={n_query})", f"{tag}_R@10"),
                                         (f"combinazione mai vista\n(n={n_new})",
                                          f"nuove_{tag}_R@10")])
               for tag in ("esatto", "classe")],
              "Test, zero-shot: R@10 su tutte le query e su quelle con combinazione verbo+nome "
              "assente dal train", FIG_DIR / "p6_zero_shot.png")
    print(f"salvato: {FIG_DIR / 'p6_zero_shot.png'}")

    video, text, meta = load_pair(ENCODER, "test", normalize=False)
    idx, class_of = select_matrix_clips(meta)
    sims = {}
    for m in load_manifest():                               # verifica anche gli sha256
        if m["seed"] == MLP_SEED:
            sims[m["positives"]] = mlp_sim(m, video[idx], text[idx])
    probs = {"MLP, loss diag": row_softmax(sims["diag"]),
             "MLP, loss classe (finale)": row_softmax(sims["classe"])}
    labels = [str(meta["narration"][i]) for i in idx]
    plot_matrix(probs, labels, class_of, FIG_DIR / "p6_matrice_similarita.png")
    print("\nclip della matrice, una riga per classe:")
    for c in range(MATRIX_CLASSES):
        print(f"  {c + 1}. " + " | ".join(np.array(labels)[class_of == c]))
    print("probabilita' media per frase:")
    for title, prob in probs.items():
        summary = matrix_summary(prob, class_of)
        print(f"  {title}: " + ", ".join(f"{k} {v:.0%}" for k, v in summary.items()))
    print(f"salvato: {FIG_DIR / 'p6_matrice_similarita.png'}")


if __name__ == "__main__":
    main()
