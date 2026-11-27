"""Zero-shot sulle parole: frasi del test riscritte con sinonimi mai visti in training.

Domanda: se nella frase cercata una parola viene sostituita con un sinonimo che le teste non
hanno mai visto (es. "wash pan" -> "wash skillet"), il modello trova ancora i video giusti?
MPNet e' congelato e preaddestrato su testo generico, quindi conosce il sinonimo; le teste MLP e
il ridge lo vedono per la prima volta. Nessun training: le feature video del test sono gia'
estratte, si ricodificano solo le frasi.

Scelte fissate prima di guardare i risultati (sezione Zero-shot di docs/roadmap_fase5.md):
- sinonimi: NOUNS e VERBS. Ognuno e' assente da tutte le frasi di train e val (controllato
  all'avvio: se no lo script si ferma). Scartati quelli che contengono la parola originale
  (teakettle, stockpot), antiquati (icebox) o che rendono la frase sgrammaticata (deposit).
- regola: si sostituisce la parola intera (pan si', pans e pancake no), senza distinguere
  maiuscole; per i verbi solo se non e' seguita da una particella (PARTICLES: "take knife" ->
  "grasp knife", "take out knife" resta fuori). Ogni frase riscritta ha una sola parola cambiata:
  una query con due parole della lista da' due voci, una per sostituzione.
- confronto sulle stesse voci: frase originale contro frase riscritta, entrambe codificate ora
  con MPNet come in extract_text_features.py. Controllo: con le frasi originali i ranghi devono
  coincidere con quelli di P6 (experiments/final/final_test_ranks.npz).
- metriche come D2: candidati = tutti i video del test; classe R@10, rango mediano di classe,
  esatto R@10. Giusto = stessa classe verbo+nome della clip della query, come sempre.
- modelli: ridge (λ scelto sul val), MLP diag e MLP classe con i 3 seed del manifest (media ±
  dev. std campionaria sui seed, come D4).

Output:
- experiments/final/zero_shot_sinonimi.csv: per modello, seed e gruppo (nomi, verbi, ogni
  singola sostituzione) le metriche con frase originale (orig_) e con sinonimo (sin_);
- figures/zero_shot_sinonimi.png: classe R@10 originale contro sinonimo, pannelli nomi e verbi.

Uso: python src/evaluation/zero_shot_sinonimi.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
# pylint: disable=wrong-import-position
from evaluation.evaluate_test import (  # noqa: E402
    ENCODER, FINAL, OUT_RANKS, load_manifest, mlp_embed)
from evaluation.linear_alignment import best_val_lambdas, ridge, standardize  # noqa: E402
from evaluation.metrics import first_relevant_rank  # noqa: E402
from evaluation.p6_figures import style  # noqa: E402
from evaluation.plot_training import INK, plt  # noqa: E402
from models.text_encoders import MpnetEncoder  # noqa: E402
from utils.features import l2_normalize, load_pair, load_split, write_csv  # noqa: E402

OUT = FINAL / "zero_shot_sinonimi.csv"
FIG = ROOT / "figures/zero_shot_sinonimi.png"
K = 10
BATCH = 256
NOUNS = {"pan": "skillet", "plate": "platter", "tap": "faucet", "bag": "sack",
         "glass": "tumbler", "potato": "spud", "bin": "trashcan", "counter": "worktop",
         "hob": "stovetop"}
VERBS = {"take": "grasp", "wash": "cleanse", "pour": "decant", "move": "relocate"}
GROUPS = {"nomi": NOUNS, "verbi": VERBS}
PARTICLES = ("up", "down", "out", "off", "on", "in", "away")
# (modello, etichetta, colore) e (condizione, etichetta, colore) per la figura
MODELS = [("ridge", "ridge lineare"), ("mlp_diag", "MLP, loss diag"),
          ("mlp_classe", "MLP, loss classe (finale)")]
CONDITIONS = [("orig", "frase originale", "#2a78d6"),
              ("sin", "con il sinonimo mai visto", "#eb6834")]


def tokens(text):
    """Parole minuscole di una frase (solo lettere)."""
    return re.findall(r"[a-z]+", text.lower())


def check_unseen():
    """Si ferma se un sinonimo compare in una frase di train o val."""
    seen = {w for split in ("train", "val") for r in load_split(split)
            for w in tokens(r["narration"])}
    found = sorted(s for table in GROUPS.values() for s in table.values() if s in seen)
    if found:
        raise SystemExit(f"sinonimi presenti in train/val: {found}")
    print(f"vocabolario train+val: {len(seen)} parole, nessun sinonimo presente")


def pattern(word, verb):
    """Parola intera; per i verbi non seguita da una particella."""
    tail = rf"(?!\s+(?:{'|'.join(PARTICLES)})\b)" if verb else ""
    return re.compile(rf"\b{word}\b{tail}", re.IGNORECASE)


def rewrite(narrations):
    """Voci da valutare: [(gruppo, sostituzione, indice della query, frase riscritta)]."""
    items = []
    for group, table in GROUPS.items():
        for word, synonym in table.items():
            pat = pattern(word, group == "verbi")
            for q, text in enumerate(narrations):
                new = pat.sub(synonym, text)
                if new != text:
                    items.append((group, f"{word} → {synonym}", q, new))
    return items


def encode(texts):
    """Feature MPNet grezze, come extract_text_features.py: ogni frase unica una volta."""
    encoder = MpnetEncoder()
    unique = sorted(set(texts))
    vectors = np.vstack([encoder.encode(unique[i:i + BATCH])
                         for i in range(0, len(unique), BATCH)])
    lookup = dict(zip(unique, vectors))
    return np.stack([lookup[t] for t in texts]).astype(np.float32)


def ridge_sims(text_sets, lam):
    """Similarita' del ridge per ogni insieme di frasi, come ridge_sim di evaluate_test.py.

    Feature L2, standardizzazione con media e dev. del train, W dal train.
    """
    video_tr, text_tr, _ = load_pair(ENCODER, "train")
    video, _, _ = load_pair(ENCODER, "test")
    video_tr, video = standardize(video_tr, video)
    w = None
    sims = []
    for text in text_sets:
        text_tr_s, text_s = standardize(text_tr, l2_normalize(text))
        if w is None:
            w = ridge(text_tr_s, video_tr, lam)
        sims.append(l2_normalize(text_s @ w) @ l2_normalize(video).T)
    return sims


def mlp_sims(model_info, video, text_sets):
    """Similarita' coseno di un checkpoint MLP per ogni insieme di frasi (prodotto in torch)."""
    z_v, z_t = mlp_embed(model_info, video, np.vstack(text_sets))
    out, start = [], 0
    for text in text_sets:
        part = z_t[start:start + len(text)]
        out.append((torch.from_numpy(part) @ torch.from_numpy(z_v).T).numpy())
        start += len(text)
    return out


def ranks(sim, queries, verbs, nouns):
    """Ranghi (classe, esatto) del primo video giusto per ogni voce."""
    same_class = (verbs[queries][:, None] == verbs[None]) & (nouns[queries][:, None] == nouns[None])
    exact = np.zeros_like(same_class)
    exact[np.arange(len(queries)), queries] = True
    return first_relevant_rank(sim, same_class), first_relevant_rank(sim, exact)


def metrics(prefix, class_ranks, exact_ranks):
    """classe R@10, rango mediano di classe ed esatto R@10, con prefisso orig_ o sin_."""
    return {f"{prefix}_classe_R@{K}": round(float((class_ranks <= K).mean() * 100), 2),
            f"{prefix}_classe_rango_mediano": float(np.median(class_ranks)),
            f"{prefix}_esatto_R@{K}": round(float((exact_ranks <= K).mean() * 100), 2)}


def model_rows(model, seed, groups, sims, queries, verbs, nouns):
    """Righe del CSV di un modello: una per gruppo (nomi, verbi e ogni sostituzione)."""
    rank_sets = {c: ranks(s, queries, verbs, nouns) for c, s in zip(("orig", "sin"), sims)}
    rows = []
    for name, mask in groups:
        row = {"modello": model, "seed": seed, "gruppo": name, "n": int(mask.sum())}
        for c in ("orig", "sin"):
            row.update(metrics(c, rank_sets[c][0][mask], rank_sets[c][1][mask]))
        rows.append(row)
    return rows, rank_sets["orig"][0]


def aggregate_seeds(rows):
    """Righe media e dev. std campionaria (ddof=1) sui seed, per un modello e un gruppo."""
    fixed = ("modello", "gruppo", "n")
    out = []
    for stat, fn in (("media", np.mean), ("std", lambda v: np.std(v, ddof=1))):
        row = {k: rows[0][k] for k in fixed} | {"seed": stat}
        for k in rows[0]:
            if k not in fixed and k != "seed":
                row[k] = round(float(fn([r[k] for r in rows])), 2)
        out.append({k: row[k] for k in rows[0]})              # stesso ordine delle colonne
    return out


def plot(rows, out):
    """Classe R@10 originale contro sinonimo, per modello; pannelli nomi e verbi."""
    col = f"classe_R@{K}"
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True, constrained_layout=True)
    width = 0.36
    for ax, group in zip(axes, GROUPS):
        x = np.arange(len(MODELS))
        for j, (cond, label, color) in enumerate(CONDITIONS):
            values, errors = [], []
            for model, _ in MODELS:
                mine = {r["seed"]: r for r in rows if r["modello"] == model and r["gruppo"] == group}
                row = mine.get("media", next(iter(mine.values())))
                values.append(row[f"{cond}_{col}"])
                errors.append(mine["std"][f"{cond}_{col}"] if "std" in mine else 0)
            pos = x + (j - 0.5) * width
            ax.bar(pos, values, width, color=color, label=label, zorder=2)
            ax.errorbar(pos, values, yerr=errors, fmt="none", ecolor=INK, elinewidth=1,
                        capsize=3, zorder=3)
            for xi, v, e in zip(pos, values, errors):
                ax.text(xi, v + e + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5,
                        color=INK)
        n = next(r["n"] for r in rows if r["gruppo"] == group)
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in MODELS], fontsize=9.5, color=INK)
        ax.set_ylim(0, 105)
        style(ax, f"{group} sostituiti ({n} frasi del test)", "",
              f"classe R@{K} (%)" if ax is axes[0] else "", grid_axis="y")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2,
               fontsize=9, frameon=False, labelcolor=INK)
    fig.suptitle("Zero-shot sulle parole: la stessa frase del test con una parola sostituita da "
                 "un sinonimo mai visto in training. Barre di errore: dev. std sui 3 seed MLP",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def main():
    """Riscrive le frasi, le codifica, calcola i ranghi per tutti i modelli, salva CSV e figura."""
    check_unseen()
    video, text, meta = load_pair(ENCODER, "test", normalize=False)
    verbs, nouns = meta["verb_class"], meta["noun_class"]
    items = rewrite(list(meta["narration"]))
    queries = np.array([q for _, _, q, _ in items])
    group_of = np.array([g for g, *_ in items])
    swap_of = np.array([s for _, s, *_ in items])
    groups = [(g, group_of == g) for g in GROUPS]
    groups += [(s, swap_of == s) for g in GROUPS for s in
               (f"{w} → {syn}" for w, syn in GROUPS[g].items())]
    for name, mask in groups:
        print(f"  {name:<20} {int(mask.sum()):5d} voci")

    print("codifica MPNet delle frasi originali e riscritte...")
    orig = encode([str(meta["narration"][q]) for q in queries])
    new = encode([t for *_, t in items])
    diff = float(np.abs(orig - text[queries]).max())
    print(f"  frasi originali contro data/features/text/mpnet.npz: differenza massima {diff:.1e}")

    stored = np.load(OUT_RANKS)
    rows = []

    def add(model, seed, sims):
        model_r, orig_class = model_rows(model, seed, groups, sims, queries, verbs, nouns)
        key = model if seed == "" else f"{model}_s{seed}"
        mismatch = int((orig_class != stored[f"{key}_classe"][queries]).sum())
        print(f"  {key}: ranghi di classe delle frasi originali diversi da P6: {mismatch}")
        rows.extend(model_r)
        return model_r

    lam = best_val_lambdas([ENCODER])[ENCODER]
    add("ridge", "", ridge_sims([orig, new], lam))
    for positives in ("diag", "classe"):
        seed_rows = [add(f"mlp_{positives}", m["seed"], mlp_sims(m, video, [orig, new]))
                     for m in load_manifest() if m["positives"] == positives]
        for name, _ in groups:
            rows += aggregate_seeds([r for rs in seed_rows for r in rs if r["gruppo"] == name])
    write_csv(OUT, rows)
    print(f"salvato: {OUT}")

    col = f"classe_R@{K}"
    print(f"\n{col}: frase originale -> con sinonimo (MLP: media sui seed)")
    for name, _ in groups:
        line = []
        for model, _ in MODELS:
            mine = {r["seed"]: r for r in rows if r["modello"] == model and r["gruppo"] == name}
            r = mine.get("media", next(iter(mine.values())))
            line.append(f"{model} {r[f'orig_{col}']:5.1f} -> {r[f'sin_{col}']:5.1f}")
        n = next(r["n"] for r in rows if r["gruppo"] == name)
        print(f"  {name:<20} (n={n:4d})  " + " | ".join(line))

    plot(rows, FIG)
    print(f"salvato: {FIG}")


if __name__ == "__main__":
    main()
