"""Figure t-SNE dello spazio appreso (P7, sul test). Solo descrittive: non ne segue nessuna scelta.

t-SNE porta i punti da 128 dimensioni a 2 cercando di conservare i vicini di ogni punto: si
leggono solo i vicinati (chi sta accanto a chi, se una classe forma un gruppo). Distanze tra
gruppi, dimensioni dei gruppi e assi non hanno significato; le coordinate di due fit diversi non
sono confrontabili, si confronta solo la struttura.

Scelte, fissate prima di guardare le figure:
- modelli: MLP diag e MLP classe, seed 0 (D4 di docs/roadmap_p6_test.md), sha256 verificato;
- clip: N_CLIPS clip a caso dello split (seed SAMPLE_SEED), cioe' la distribuzione su cui si
  calcola R@K;
- un solo t-SNE per modello con video e frasi insieme: lo spazio e' condiviso, e t-SNE non
  proietta punti nuovi su un fit esistente. Metrica coseno (quella del modello), init PCA,
  perplexity PERPLEXITY, random_state TSNE_SEED;
- colore = verbo, solo i N_VERBS verbi piu' frequenti dello split (dai metadati), gli altri in
  grigio: in uno scatter solo i primi 3 colori della palette restano distinguibili a coppie,
  anche con daltonismo. Marker: cerchio = video, triangolo = frase.

1. figures/tsne_spazio_appreso_<split>.png: due pannelli, MLP diag e MLP classe. Con diag ci si
   aspetta ogni frase vicina al suo video ma classi meno compatte; con classe gruppi per classe
   piu' netti. Se video e frasi formano due isole separate e' il modality gap.

2. figures/tsne_zero_shot_<split>.png: MLP classe, fit su clip a caso piu' tutte le clip con
   combinazione verbo+nome assente dal train (D3), evidenziate con bordo scuro; le altre fanno da
   contesto, sbiadite. Domanda: le combinazioni nuove cadono tra i video dello stesso verbo?
   Fit a parte: aggiungere le clip zero-shot al campione della figura 1 ne altererebbe la
   composizione.

3. figures/tsne_perplexity_<split>.png: MLP classe con perplexity 15, 30, 50 sulle stesse clip
   della figura 1. Se i gruppi restano al variare della perplexity la struttura e' reale.

Uso: python src/evaluation/plot_tsne.py [--split test]
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from matplotlib import patheffects
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
# pylint: disable=wrong-import-position
from evaluation.evaluate_test import ENCODER, load_manifest, mlp_embed  # noqa: E402
from evaluation.linear_alignment import unseen_mask  # noqa: E402
from evaluation.plot_training import GRID, INK, INK_MUTED, plt  # noqa: E402
from utils.features import load_pair  # noqa: E402

FIG_DIR = ROOT / "figures"
VERB_CLASSES = ROOT / "src/datasets/epic-kitchens-100-annotations/EPIC_100_verb_classes.csv"
MLP_SEED = 0                    # D4: le figure usano il seed 0
SAMPLE_SEED, N_CLIPS = 0, 2000  # clip a caso dello split
TSNE_SEED, PERPLEXITY = 0, 30
PERPLEXITIES = (15, 30, 50)     # figura 3
N_VERBS = 3
# slot 1-3 della palette di riferimento (gli stessi colori di final_figures.py), validati a coppie
VERB_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
OTHER = "#c9c8c4"               # altri verbi
MARKERS = {"video": "o", "frase": "^"}
MODEL_TITLES = {"diag": "MLP, loss diag", "classe": "MLP, loss classe (finale)"}


def verb_names():
    """Nome di ogni classe di verbo EPIC-100: {id: "take", ...}."""
    with open(VERB_CLASSES, newline="", encoding="utf-8") as f:
        return {int(r["id"]): r["key"] for r in csv.DictReader(f)}


def top_verbs(verbs, n):
    """Le n classi di verbo piu' frequenti dello split (a parita', id piu' basso)."""
    ids, counts = np.unique(verbs, return_counts=True)
    order = np.argsort(-counts, kind="stable")              # stabile: a parita' vince l'id
    return [int(v) for v in ids[order[:n]]]


def embed_models(video, text):
    """Embedding (z_v, z_t) di tutte le clip per i modelli seed MLP_SEED: {"diag": ..., ...}."""
    return {m["positives"]: mlp_embed(m, video, text)
            for m in load_manifest() if m["seed"] == MLP_SEED}   # verifica anche gli sha256


def joint_tsne(z_v, z_t, perplexity):
    """Un solo t-SNE su video e frasi insieme; ritorna le coordinate 2D (video, frasi)."""
    start = time.perf_counter()
    xy = TSNE(n_components=2, perplexity=perplexity, metric="cosine", init="pca",
              random_state=TSNE_SEED).fit_transform(np.vstack([z_v, z_t]))
    print(f"    t-SNE su {len(xy)} punti, perplexity {perplexity}: "
          f"{time.perf_counter() - start:.0f} s")
    return xy[:len(z_v)], xy[len(z_v):]


def draw(ax, coords, verbs, top, size=7, alpha=0.8, linewidths=0, **kwargs):
    """Video (cerchi) e frasi (triangoli): prima gli altri verbi in grigio, poi i verbi colorati.

    linewidths > 0 (con edgecolors) disegna il bordo, per evidenziare le clip zero-shot.
    """
    for modality, xy in zip(MARKERS, coords):
        other = ~np.isin(verbs, top)
        ax.scatter(*xy[other].T, s=size, c=OTHER, marker=MARKERS[modality], alpha=alpha,
                   linewidths=linewidths, **kwargs)
        for verb, color in zip(top, VERB_COLORS):
            mine = verbs == verb
            ax.scatter(*xy[mine].T, s=size, c=color, marker=MARKERS[modality], alpha=alpha,
                       linewidths=linewidths, **kwargs)


def label_verbs(ax, coords, verbs, top, names):
    """Nome del verbo sulla mediana dei suoi video e delle sue frasi (etichette dirette)."""
    halo = [patheffects.withStroke(linewidth=3, foreground="white")]
    for modality, xy in zip(MARKERS, coords):
        for verb in top:
            x, y = np.median(xy[verbs == verb], axis=0)
            ax.annotate(f"{names[verb]} ({modality})", (x, y), fontsize=8, color=INK,
                        ha="center", va="center", path_effects=halo, zorder=5)


def clean(ax, title):
    """Pannello t-SNE: niente tacche (gli assi non hanno significato), bordo discreto."""
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color(GRID)


def legend(fig, top, names, verbs, extra=(), ncol=None):
    """Legenda unica sotto i pannelli: verbi (con quota nello split), altri verbi, modalita'."""
    def dot(color, marker="o", **kwargs):
        return Line2D([], [], linestyle="", marker=marker, markersize=7, color=color, **kwargs)

    handles = [dot(c) for c in VERB_COLORS[:len(top)]] + [dot(OTHER)]
    labels = [f"{names[v]} ({(verbs == v).mean():.0%} dello split)" for v in top]
    labels.append("altri verbi")
    handles += [dot(INK_MUTED, MARKERS["video"]), dot(INK_MUTED, MARKERS["frase"])]
    labels += ["video", "frase"]
    for handle, label in extra:
        handles.append(handle)
        labels.append(label)
    fig.legend(handles, labels, loc="outside lower center", ncol=ncol or len(labels), fontsize=9,
               frameon=False, labelcolor=INK)


def save(fig, out):
    """Salva su sfondo bianco, come le altre figure, e chiude."""
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"salvato: {out}")


def plot_models(coords, verbs, top, names, suptitle, out):
    """Figura 1: un pannello per modello, stesse clip."""
    fig, axes = plt.subplots(1, len(coords), figsize=(15, 7.4), constrained_layout=True)
    for ax, (positives, xy) in zip(axes, coords.items()):
        draw(ax, xy, verbs, top)
        label_verbs(ax, xy, verbs, top, names)
        clean(ax, MODEL_TITLES[positives])
    legend(fig, top, names, verbs)
    fig.suptitle(suptitle, x=0.01, ha="left", fontsize=10, color=INK_MUTED)
    save(fig, out)


def plot_zero_shot(coords, verbs, new, top, names, suptitle, out):
    """Figura 2: clip zero-shot evidenziate (bordo scuro), le altre sbiadite come contesto."""
    fig, ax = plt.subplots(figsize=(10, 8.4), constrained_layout=True)
    context = [xy[~new] for xy in coords]
    draw(ax, context, verbs[~new], top, size=6, alpha=0.25)
    draw(ax, [xy[new] for xy in coords], verbs[new], top, size=24, alpha=1.0,
         linewidths=0.6, edgecolors=INK, zorder=3)
    clean(ax, MODEL_TITLES["classe"])
    ring = Line2D([], [], linestyle="", marker="o", markersize=7, markerfacecolor="white",
                  markeredgecolor=INK)
    legend(fig, top, names, verbs, extra=[(ring, f"combinazione mai vista (n={new.sum()})")],
           ncol=4)                                          # figura stretta: legenda su 2 righe
    fig.suptitle(suptitle, x=0.01, ha="left", fontsize=10, color=INK_MUTED)
    save(fig, out)


def plot_perplexity(coords, verbs, top, names, suptitle, out):
    """Figura 3: MLP classe, un pannello per perplexity, stesse clip."""
    fig, axes = plt.subplots(1, len(coords), figsize=(18, 6.6), constrained_layout=True)
    for ax, (perplexity, xy) in zip(axes, coords.items()):
        draw(ax, xy, verbs, top, size=5)
        clean(ax, f"perplexity {perplexity}")
    legend(fig, top, names, verbs)
    fig.suptitle(suptitle, x=0.01, ha="left", fontsize=10, color=INK_MUTED)
    save(fig, out)


def main():
    """Embedding dei due modelli, t-SNE e le tre figure."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    split = parser.parse_args().split
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    _, _, meta_tr = load_pair(ENCODER, "train")
    # feature non normalizzate: mlp_embed standardizza con le statistiche del train, come in P6
    video, text, meta = load_pair(ENCODER, split, normalize=False)
    verbs, names = meta["verb_class"], verb_names()
    top = top_verbs(verbs, N_VERBS)
    emb = embed_models(video, text)

    rng = np.random.default_rng(SAMPLE_SEED)
    idx = np.sort(rng.choice(len(video), N_CLIPS, replace=False))
    new = unseen_mask(meta_tr, meta)
    idx_zs = np.union1d(idx, np.flatnonzero(new))           # campione + tutte le zero-shot
    print(f"{split}: {len(video)} clip, campione di {len(idx)} (seed {SAMPLE_SEED}); verbi "
          f"colorati {', '.join(names[v] for v in top)}: "
          f"{np.isin(verbs[idx], top).mean():.0%} del campione; "
          f"zero-shot (combinazione nuova) {new.sum()}")

    what = f"{split.capitalize()}: t-SNE congiunto di video e frasi, seed {MLP_SEED}"
    note = "Assi senza significato: contano solo i vicini."

    print("figura 1: spazio appreso")
    coords = {p: joint_tsne(emb[p][0][idx], emb[p][1][idx], PERPLEXITY)
              for p in ("diag", "classe")}
    plot_models(coords, verbs[idx], top, names,
                f"{what}, {len(idx)} clip a caso, perplexity {PERPLEXITY}. {note}",
                FIG_DIR / f"tsne_spazio_appreso_{split}.png")

    print("figura 2: zero-shot")
    z_v, z_t = emb["classe"]
    plot_zero_shot(joint_tsne(z_v[idx_zs], z_t[idx_zs], PERPLEXITY), verbs[idx_zs],
                   new[idx_zs], top, names,
                   f"{what}: {len(idx)} clip a caso più tutte le {new.sum()} con combinazione "
                   f"verbo+nome assente dal train (bordate). {note}",
                   FIG_DIR / f"tsne_zero_shot_{split}.png")

    print("figura 3: perplexity")
    by_perplexity = {p: coords["classe"] if p == PERPLEXITY
                     else joint_tsne(z_v[idx], z_t[idx], p) for p in PERPLEXITIES}
    plot_perplexity(by_perplexity, verbs[idx], top, names,
                    f"{what}, MLP classe, {len(idx)} clip a caso (le stesse della figura "
                    f"spazio appreso) al variare della perplexity. {note}",
                    FIG_DIR / f"tsne_perplexity_{split}.png")


if __name__ == "__main__":
    main()
