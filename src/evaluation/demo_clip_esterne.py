"""Demo con clip esterne: il modello finale su video mai visti (Pexels), per le slide.

Le feature di EPIC sono gia' estratte; per le clip nuove lo script ripete la stessa catena:
    clip .mp4 -> 8 frame uniformi -> TimeSformer K400 (CLS, 768) -> testa video
    frase     -> MPNet (768)                                  -> testa testo
e confronta gli embedding con il coseno.

Scelte fissate prima di vedere i risultati (sezione "Demo con clip esterne" di
docs/roadmap_fase5.md):
- modello: MLP classe seed 0 (D4 di docs/roadmap_p6_test.md), sha256 verificato dal manifest;
- clip e frasi: experiments/demo/clips.csv, clip gia' tagliate in data/demo/clips/;
- video trattati come EPIC: lato corto 224 (frame_size di extract_features.py), N_FRAMES indici
  uniformi tra primo e ultimo frame (uniform_indices di cut_clips.py), processor TimeSformer,
  token CLS. Le due funzioni sono ricopiate qui: src/datasets non si puo' importare perche' il
  nome si scontra con il pacchetto HuggingFace "datasets";
- standardizzazione con media e dev. std del train salvate nel checkpoint (mlp_embed, come P6).

Uso:
    python src/evaluation/demo_clip_esterne.py                  # figura + matrice a schermo
    python src/evaluation/demo_clip_esterne.py --query "slice watermelon" --query "add cheese"

Output: figures/demo_clip_esterne.png. Le feature delle clip si salvano in
data/demo/clip_features.npz (fuori da git) e si riusano se le clip non sono cambiate.
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
# pylint: disable=wrong-import-position
from evaluation.evaluate_test import load_manifest, mlp_embed  # noqa: E402
from evaluation.p6_figures import BLUES, TAU, row_softmax  # noqa: E402
from evaluation.plot_training import INK, INK_MUTED, plt  # noqa: E402

CLIPS_CSV = ROOT / "experiments/demo/clips.csv"
CLIP_DIR = ROOT / "data/demo/clips"
CACHE = ROOT / "data/demo/clip_features.npz"
FIG = ROOT / "figures/demo_clip_esterne.png"
MODEL = ("classe", 0)           # (positives, seed) del modello finale usato nelle figure
N_FRAMES = 8                    # come le feature di EPIC (timesformer_k400_8f)
SHORT_SIDE = 224


def uniform_indices(n_frames, k):
    """k indici di frame distribuiti uniformemente nella clip (come cut_clips.py)."""
    if k == 1:
        return [(n_frames - 1) // 2]
    return [round(i * (n_frames - 1) / (k - 1)) for i in range(k)]


def frame_size(clip_path):
    """(larghezza, altezza) dopo lo scale del lato corto a 224 (come extract_features.py)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", str(clip_path)],
        capture_output=True, text=True, check=True).stdout
    stream = json.loads(out)["streams"][0]
    w, h = int(stream["width"]), int(stream["height"])
    if w >= h:
        return round(w * SHORT_SIDE / h), SHORT_SIDE
    return SHORT_SIDE, round(h * SHORT_SIDE / w)


def read_frames(clip_path):
    """Gli N_FRAMES frame uniformi della clip, come immagini PIL (lato corto 224)."""
    w, h = frame_size(clip_path)
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(clip_path),
           "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    return [Image.fromarray(frames[i]) for i in uniform_indices(len(frames), N_FRAMES)]


def read_clips():
    """Righe di experiments/demo/clips.csv; si ferma se manca una clip."""
    with open(CLIPS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    missing = [r["file"] for r in rows if not (CLIP_DIR / r["file"]).exists()]
    if missing:
        raise SystemExit(f"clip mancanti in {CLIP_DIR}: {missing}")
    return rows


def clip_features(files):
    """Feature TimeSformer (n, 768) e frame centrale di ogni clip; usa la cache se valida.

    La cache vale se contiene le stesse clip ed e' piu' recente di tutte.
    """
    paths = [CLIP_DIR / f for f in files]
    if CACHE.exists() and CACHE.stat().st_mtime > max(p.stat().st_mtime for p in paths):
        cached = np.load(CACHE)
        if list(cached["files"]) == list(files):
            print(f"feature delle clip dalla cache: {CACHE.name}")
            return cached["features"], cached["thumbs"]
    from models.timesformer import TimesformerEncoder  # pylint: disable=import-outside-toplevel
    encoder = TimesformerEncoder()
    features, thumbs = [], []
    for p in paths:
        frames = read_frames(p)
        features.append(encoder.encode([frames])[0])
        thumbs.append(square_thumb(frames[N_FRAMES // 2]))
        print(f"  {p.name}: {N_FRAMES} frame, feature {features[-1].shape}")
    features, thumbs = np.stack(features).astype(np.float32), np.stack(thumbs)
    np.savez(CACHE, features=features, thumbs=thumbs, files=np.array(files))
    return features, thumbs


def square_thumb(image, size=120):
    """Ritaglio quadrato centrale, per miniature tutte uguali nella figura."""
    w, h = image.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    return np.asarray(image.crop((left, top, left + side, top + side)).resize((size, size)))


def text_features(sentences):
    """Feature MPNet grezze delle frasi, come extract_text_features.py."""
    from models.text_encoders import MpnetEncoder  # pylint: disable=import-outside-toplevel
    return MpnetEncoder().encode(list(sentences)).astype(np.float32)


def final_model():
    """La voce del manifest del modello finale (MODEL); verifica anche gli sha256."""
    return next(m for m in load_manifest() if (m["positives"], m["seed"]) == MODEL)


def plot(prob, sentences, files, thumbs, out):
    """Matrice frasi x clip: probabilita' di riga, miniature in alto, diagonale bordata."""
    n = len(files)
    hits = int((prob.argmax(1) == np.arange(n)).sum())
    fig, ax = plt.subplots(figsize=(3.2 + 1.1 * n, 3.4 + 0.95 * n))
    ax.imshow(prob, cmap=BLUES, vmin=0, vmax=1, interpolation="nearest")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{prob[i, j]:.0%}", ha="center", va="center", fontsize=10,
                    color="white" if prob[i, j] > 0.5 else INK)
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK,
                                   linewidth=2, zorder=3))
    # miniature in un riquadro sopra la matrice, alto quanto una riga e con le stesse x:
    # ogni miniatura occupa esattamente la sua colonna
    top = make_axes_locatable(ax).append_axes("top", size=f"{100 / n:.4f}%", pad=0.08)
    for j, thumb in enumerate(thumbs):
        top.imshow(thumb, extent=(j - 0.5, j + 0.45, 0, 1), aspect="auto")
    top.set_xlim(ax.get_xlim())
    top.set_ylim(0, 1)
    top.axis("off")
    ax.set_xticks(range(n))
    ax.set_xticklabels([f.removesuffix(".mp4") for f in files], fontsize=8.5,
                       color=INK_MUTED, rotation=20, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"«{s}»" for s in sentences], fontsize=10, color=INK)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xlabel("clip mai viste (Pexels)", color=INK_MUTED)
    fig.suptitle(f"{hits} frasi su {n} trovano la loro clip al primo posto (a caso: 1 su {n})",
                 fontsize=12, color=INK, x=0.02, y=0.98, ha="left")
    # spiegazione in fondo: in alto coprirebbe le miniature
    fig.text(0.02, 0.015, f"Modello finale (MLP, loss classe, seed {MODEL[1]}). In ogni cella la "
             f"probabilità che la frase scelga la clip (softmax della riga, τ={TAU}).\n"
             "Bordo = coppia giusta. Clip da Pexels, mai viste in training né nel test di EPIC.",
             fontsize=8.5, color=INK_MUTED, ha="left", va="bottom")
    fig.subplots_adjust(top=0.92, left=0.27, bottom=0.17, right=0.97)
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    return hits


def main():
    """Figura e matrice delle clip della demo, oppure classifica per frasi libere (--query)."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--query", action="append",
                        help="frase libera: stampa la classifica delle clip (ripetibile)")
    args = parser.parse_args()

    rows = read_clips()
    files = [r["file"] for r in rows]
    video, thumbs = clip_features(files)
    model = final_model()

    if args.query:
        z_v, z_t = mlp_embed(model, video, text_features(args.query))
        sim = z_t @ z_v.T
        prob = row_softmax(sim)
        for q, s, p in zip(args.query, sim, prob):
            print(f"\n«{q}»")
            for rank, j in enumerate(np.argsort(-s, kind="stable"), start=1):
                print(f"  {rank}. {files[j]:<22} coseno {s[j]:+.3f}  probabilità {p[j]:.0%}")
        return

    sentences = [r["frase"] for r in rows]
    z_v, z_t = mlp_embed(model, video, text_features(sentences))
    sim = z_t @ z_v.T
    prob = row_softmax(sim)
    print("\nprobabilità frase -> clip (righe = frasi, colonne = clip nello stesso ordine)")
    for s, p in zip(sentences, prob):
        print(f"  {s:<26}" + " ".join(f"{v:5.0%}" for v in p))
    hits = plot(prob, sentences, files, thumbs, FIG)
    print(f"\n{hits} frasi su {len(files)} al primo posto la loro clip")
    print(f"salvato: {FIG}")


if __name__ == "__main__":
    main()
