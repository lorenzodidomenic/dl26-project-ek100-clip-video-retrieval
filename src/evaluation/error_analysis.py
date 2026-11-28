"""Analisi dei risultati sul test: per verbo, per frequenza, errori del top-1 (P7). Descrittiva.

Modelli: ridge lineare (λ=1000, riferimento) e MLP con loss classe (modello finale); in F3 anche
MLP con loss diag. MLP seed 0 come fissato in D4 di docs/roadmap_p6_test.md. Stessi colori di
final_figures.py.

Scelte fissate prima di guardare i risultati (dipendono solo dai metadati):
1. figures/analisi_recall_per_verbo.png: classe R@10 (% di query con almeno un video della stessa
   classe verbo+nome tra i primi 10) per i N_VERBS verbi piu' frequenti del test, ordinati per
   frequenza. Linea tratteggiata = classe R@10 del modello finale su tutte le query.
   Domanda: su quali azioni il modello va bene e su quali male?
2. figures/analisi_recall_per_frequenza.png: classe R@10 con le query divise per quante clip della
   loro classe verbo+nome ci sono nel train (FREQ_BINS: 0, 1-10, 11-100, oltre 100). Il gruppo
   "0" e' il sottoinsieme "combinazione nuova" di D3.
   Domanda: quanto conta aver visto la classe nel train?
   F1 e F2 non rilanciano nessun modello: leggono i ranghi per query salvati da evaluate_test.py
   (experiments/final/final_test_ranks.npz).
3. figures/analisi_tipi_di_errore.png: per ogni query, il primo video recuperato (top-1) confrontato
   con la query: classe giusta, verbo giusto ma nome sbagliato, nome giusto ma verbo sbagliato,
   tutti e due sbagliati. Una barra al 100% per modello (ridge, MLP diag, MLP classe).
   Domanda: quando il top-1 e' sbagliato, cosa sbaglia?
4. figures/analisi_coppie_confuse.png: MLP classe. Tra le query con top-1 di classe sbagliata, le
   N_CONFUSIONS coppie (classe della query -> classe del top-1) piu' frequenti, colorate con il
   tipo di errore di F3. Accanto, quante query della classe finiscono in quella confusione.
   Domanda: quali coppie verbo+nome il modello confonde di piu'? (obiettivo extra della traccia)
   F3 e F4 ricalcolano le similarita' dai checkpoint (mlp_sim e ridge_sim di evaluate_test.py).
   Il top-1 e' l'argmax della riga; a parita' di similarita' vince l'indice piu' basso.

Uso: python src/evaluation/error_analysis.py
"""
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
# pylint: disable=wrong-import-position
from evaluation.evaluate_test import (  # noqa: E402
    ENCODER, OUT_RANKS, load_manifest, mlp_sim, ridge_sim)
from evaluation.linear_alignment import best_val_lambdas  # noqa: E402
from evaluation.final_figures import style  # noqa: E402
from evaluation.plot_tsne import top_verbs, verb_names  # noqa: E402
from evaluation.plot_training import INK, INK_MUTED, plt  # noqa: E402
from utils.features import load_pair, load_split  # noqa: E402

FIG_DIR = ROOT / "figures"
NOUN_CLASSES = ROOT / "src/datasets/epic-kitchens-100-annotations/EPIC_100_noun_classes.csv"
MLP_SEED = 0                    # D4: le figure usano il seed 0
K = 10
N_VERBS = 12                    # verbi del test con almeno ~200 query
N_CONFUSIONS = 10
# (etichetta, minimo, massimo incluso; None = senza limite): clip della classe nel train
FREQ_BINS = [("mai vista", 0, 0), ("rara", 1, 10), ("media", 11, 100), ("frequente", 101, None)]
MODELS = [  # (chiave nel file dei ranghi, etichetta, colore): come in final_figures.py
    ("ridge", "ridge lineare (λ=1000)", "#1baf7a"),
    (f"mlp_classe_s{MLP_SEED}", "MLP, loss classe (finale)", "#2a78d6"),
]
FINAL_MODEL = MODELS[-1]
# F3: (chiave, etichetta), dall'alto in basso
ERROR_MODELS = [("ridge", "ridge lineare (λ=1000)"),
                (f"mlp_diag_s{MLP_SEED}", "MLP, loss diag"),
                (f"mlp_classe_s{MLP_SEED}", "MLP, loss classe (finale)")]
# tipi di errore del top-1, nell'ordine delle barre impilate: (etichetta, colore).
# Niente verde per gli errori: suggerisce "positivo"
ERROR_TYPES = [("classe giusta", "#2a78d6"),
               ("verbo giusto, nome sbagliato", "#eb6834"),
               ("nome giusto, verbo sbagliato", "#7b4fb8"),
               ("verbo e nome sbagliati", "#c9c8c4")]


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


def noun_names():
    """Nome di ogni classe di nome EPIC-100: {id: "tap", ...}."""
    with open(NOUN_CLASSES, newline="", encoding="utf-8") as f:
        return {int(r["id"]): r["key"] for r in csv.DictReader(f)}


def top1_of_models():
    """Indice del primo video recuperato per ogni query: {chiave di ERROR_MODELS: array (n,)}.

    Una matrice di similarita' alla volta (n x n float32, circa 370 MB), poi si tiene solo
    l'argmax di riga.
    """
    video, text, meta = load_pair(ENCODER, "test", normalize=False)
    top1 = {}
    lam = best_val_lambdas([ENCODER])[ENCODER]
    top1["ridge"] = ridge_sim("test", lam, ENCODER).argmax(1)
    for m in load_manifest():                               # verifica anche gli sha256
        if m["seed"] == MLP_SEED:
            top1[f"mlp_{m['positives']}_s{MLP_SEED}"] = mlp_sim(m, video, text).argmax(1)
    return top1, meta["narration_id"]


def error_type(verbs, nouns, top1):
    """Indice in ERROR_TYPES per ogni query, confrontando la query con il suo top-1."""
    same_verb, same_noun = verbs[top1] == verbs, nouns[top1] == nouns
    return np.select([same_verb & same_noun, same_verb, same_noun], [0, 1, 2], default=3)


def plot_error_types(types, out):
    """Figura 3: una barra impilata al 100% per modello, un segmento per tipo di errore."""
    fig, ax = plt.subplots(figsize=(11, 3.9), constrained_layout=True)
    y = np.arange(len(ERROR_MODELS))[::-1]                  # primo modello in alto
    table = {}
    for yi, (key, _) in zip(y, ERROR_MODELS):
        shares = np.bincount(types[key], minlength=len(ERROR_TYPES)) / len(types[key]) * 100
        table[key] = shares
        left = 0.0
        for share, (label, color) in zip(shares, ERROR_TYPES):
            ax.barh(yi, share, 0.6, left=left, color=color, label=label, zorder=2,
                    edgecolor="white", linewidth=1)
            if share >= 4:                                  # scritta solo se il segmento ci sta
                ax.text(left + share / 2, yi, f"{share:.0f}%", ha="center", va="center",
                        fontsize=9, color="white" if color != ERROR_TYPES[-1][1] else INK)
            left += share
    ax.set_yticks(y)
    ax.set_yticklabels([label for _, label in ERROR_MODELS], fontsize=9.5, color=INK)
    ax.set_xlim(0, 100)
    style(ax, "Il primo video recuperato (top-1) rispetto alla frase cercata: cosa ha di "
          "giusto e cosa di sbagliato", f"% delle {len(next(iter(types.values())))} query del "
          "test", "", grid_axis="x")
    handles, labels = ax.get_legend_handles_labels()
    keep = len(ERROR_TYPES)                                 # una voce per tipo, non per barra
    fig.legend(handles[:keep], labels[:keep], loc="outside lower center", ncol=keep,
               fontsize=9, frameon=False, labelcolor=INK)
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    return table


def top_confusions(verbs, nouns, top1, types):
    """Le N_CONFUSIONS coppie (classe query, classe top-1) piu' frequenti tra gli errori.

    Ritorna [(classe query, classe top-1, query nella confusione, query della classe, tipo)].
    A parita' di conteggio, ordine per classe (deterministico).
    """
    wrong = types != 0
    pairs = Counter(zip(zip(verbs[wrong], nouns[wrong]),
                        zip(verbs[top1[wrong]], nouns[top1[wrong]])))
    class_size = Counter(zip(verbs, nouns))
    ranked = sorted(pairs.items(), key=lambda item: (-item[1], item[0]))[:N_CONFUSIONS]
    out = []
    for (query, found), count in ranked:
        kind = 1 if query[0] == found[0] else (2 if query[1] == found[1] else 3)
        out.append((query, found, count, class_size[query], kind))
    return out


def plot_confusions(confusions, out):
    """Figura 4: barre orizzontali delle confusioni piu' frequenti, colore = tipo di errore."""
    verbs, nouns = verb_names(), noun_names()

    def name(c):
        return f"{verbs[c[0]]} {nouns[c[1]]}"

    fig, ax = plt.subplots(figsize=(11, 6.2), constrained_layout=True)
    y = np.arange(len(confusions))[::-1]                    # la piu' frequente in alto
    top = max(c[2] for c in confusions)
    for yi, (_, _, count, size, kind) in zip(y, confusions):
        ax.barh(yi, count, 0.65, color=ERROR_TYPES[kind][1], zorder=2)
        ax.text(count + top * 0.01, yi, f"{count} su {size} query della classe "
                f"({count / size:.0%})", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{name(q)}  →  {name(f)}" for q, f, *_ in confusions],
                       fontsize=9.5, color=INK)
    ax.set_xlim(0, top * 1.45)                              # spazio per le scritte
    style(ax, f"Le {len(confusions)} confusioni più frequenti del modello finale: frase cercata "
          "→ classe del primo video recuperato", "query del test con questa confusione", "",
          grid_axis="x")
    used = sorted({c[4] for c in confusions})
    handles = [plt.Rectangle((0, 0), 1, 1, color=ERROR_TYPES[k][1]) for k in used]
    fig.legend(handles, [ERROR_TYPES[k][0] for k in used], loc="outside lower center",
               ncol=len(used), fontsize=9, frameon=False, labelcolor=INK)
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


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

    out = FIG_DIR / "analisi_recall_per_verbo.png"
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
    out = FIG_DIR / "analisi_recall_per_frequenza.png"
    table = plot_per_frequency(data, masks, out)
    print_table(f"classe R@{K} per frequenza della classe nel train (test, seed {MLP_SEED})",
                [f"{name} (n={int(m.sum())})" for (name, *_), m in zip(FREQ_BINS, masks)],
                table)
    print(f"salvato: {out}")

    print("\ncalcolo del top-1 di ogni modello (similarita' dai checkpoint)...")
    top1, ids = top1_of_models()
    if not np.array_equal(ids, data["narration_id"]):
        raise SystemExit("ordine delle query diverso da final_test_ranks.npz")
    types = {key: error_type(verbs, nouns, top1[key]) for key, _ in ERROR_MODELS}
    # controllo: "classe giusta" al top-1 deve coincidere con rango classe 1 (salvo parita')
    for key, _ in ERROR_MODELS:
        diff = int(((types[key] == 0) != (data[f"{key}_classe"] == 1)).sum())
        print(f"  {key}: query in cui top-1 e rango classe 1 non coincidono: {diff}")
    out = FIG_DIR / "analisi_tipi_di_errore.png"
    table = plot_error_types(types, out)
    print_table("tipo del top-1, % delle query (test, seed 0)",
                [label for label, _ in ERROR_TYPES],
                {label: table[key] for key, label in ERROR_MODELS})
    print(f"salvato: {out}")

    key = FINAL_MODEL[0]
    confusions = top_confusions(verbs, nouns, top1[key], types[key])
    out = FIG_DIR / "analisi_coppie_confuse.png"
    plot_confusions(confusions, out)
    vn, nn = verb_names(), noun_names()
    print(f"\nconfusioni piu' frequenti, {key}:")
    for q, f, count, size, kind in confusions:
        print(f"  {vn[q[0]]} {nn[q[1]]:<14} -> {vn[f[0]]} {nn[f[1]]:<14} {count:4d} su {size:4d}"
              f"  ({ERROR_TYPES[kind][0]})")
    print(f"salvato: {out}")


if __name__ == "__main__":
    main()
