"""Risultati della griglia di P4: tabella ordinata, effetti principali e heatmap delle interazioni.

Legge le run finite della griglia (stessa definizione di grid.py, seed 0). Per ognuna prende
l'epoca migliore con la regola di train.py: la prima con il massimo di classe_R@10, cioe'
quella salvata nel checkpoint. Le run non ancora finite si ignorano, quindi si puo' lanciare
anche con la griglia in corso.

Effetti principali: per ogni valore di un iperparametro, media e deviazione standard del
criterio su tutte le run con quel valore (24 per le leve a due valori, 16 per il batch).
Heatmap: in ogni cella la media del criterio sulle run con quella coppia di valori.

Seed (passo 5): le prime 5 config al seed 0 vengono rilanciate con i seed 1 e 2. Appena c'e'
almeno un seed in piu', si riassumono media e deviazione standard sui seed; con tutti i seed
finiti si applica la regola di scelta della roadmap:
1. la config con la media piu' alta e' la candidata;
2. ogni config con la media piu' bassa di meno di una deviazione standard (la maggiore delle
   due) e' alla pari; tra quelle alla pari vince la piu' semplice: d piu' piccolo, poi
   dropout piu' alto, poi la media.

Output:
- experiments/logs/p4_grid.csv: una riga per run al seed 0, ordinate per classe_R@10
- experiments/logs/p4_seed.csv: le prime 5, media e dev. std sui seed (se ci sono)
- figures/p4_effetti_principali.png, figures/p4_seed.png
- figures/p4_heatmap_batch_tau.png e figures/p4_heatmap_lr_dropout.png
A schermo: effetti principali, prime 10 run, comando per i seed, riassunto dei seed e scelta.
"""
import csv
import re
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
# pylint: disable=wrong-import-position
from evaluation.plot_training import GRID as GRID_LINE, INK, INK_MUTED, TRAIN as BLUE  # noqa: E402
from evaluation.plot_training import plt  # noqa: E402
from training import grid  # noqa: E402
from training.train import CRITERION, ridge_reference  # noqa: E402
from utils.features import write_csv  # noqa: E402

OUT = ROOT / "experiments/logs/p4_grid.csv"
OUT_SEED = ROOT / "experiments/logs/p4_seed.csv"
FIG_DIR = ROOT / "figures"
TOP = 5
SEEDS = (0, 1, 2)
# metriche riassunte sui seed, oltre al criterio
SEED_METRICS = (CRITERION, "classe_R@1", "esatto_R@10", "nuove_classe_R@10", "epoca_migliore")
# leve della griglia: colonna della tabella -> etichetta nei grafici
FACTORS = {"lr": "learning rate", "dropout": "dropout", "tau": "τ", "batch": "batch", "dim": "d"}
# scala sequenziale a una tinta (blu), dal chiaro (valori bassi) allo scuro (valori alti)
BLUES = LinearSegmentedColormap.from_list(
    "blu", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def best_row(name):
    """Riga dell'epoca migliore e numero di epoche del log di una run."""
    with open(grid.LOG_DIR / f"train_{name}.csv", newline="", encoding="utf-8") as f:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    # max ritorna il primo massimo: la stessa epoca che train.py ha salvato nel checkpoint
    return max(rows, key=lambda r: r[CRITERION]), len(rows)


def collect(seeds=(0,), only=None):
    """Una riga per run finita (config della griglia x seed), ordinate per criterio decrescente."""
    results = []
    for _, args in grid.plan(seeds, only):
        if not grid.is_complete(args):
            continue
        best, n_epochs = best_row(args.name)
        results.append({
            "run": args.name, "seed": args.seed, "lr": args.lr, "dropout": args.dropout,
            "tau": "fissa" if args.fixed_tau else "appresa", "batch": args.batch, "dim": args.dim,
            "epoca_migliore": int(best["epoca"]), "epoche": n_epochs,
            CRITERION: best[CRITERION], "classe_R@1": best["classe_R@1"],
            "esatto_R@10": best["esatto_R@10"], "nuove_classe_R@10": best["nuove_classe_R@10"],
            "classe_rango_mediano": best["classe_rango_mediano"], "tau_epoca_migliore": best["tau"],
        })
    return sorted(results, key=lambda r: r[CRITERION], reverse=True)


def levels(rows, key):
    """Valori distinti di una leva, in ordine crescente."""
    return sorted({r[key] for r in rows})


def stats(values):
    """Media, deviazione standard (campionaria) e numero di valori."""
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return float(np.mean(values)), std, len(values)


def label(key, value):
    """Valore di una leva come testo breve per assi e tabelle."""
    return f"{value:g}" if key == "lr" else str(value)


def style(ax, title):
    """Assi discreti e griglia orizzontale leggera, come in plot_training.py."""
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.grid(axis="y", color=GRID_LINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_LINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)


def plot_main_effects(rows, ridge, out):
    """Un pannello per leva: run singole in grigio, media ± deviazione standard in blu."""
    fig, axes = plt.subplots(1, len(FACTORS), figsize=(14, 4), sharey=True,
                             constrained_layout=True)
    rng = np.random.default_rng(0)
    for ax, (key, title) in zip(axes, FACTORS.items()):
        values = levels(rows, key)
        for x, value in enumerate(values):
            ys = [r[CRITERION] for r in rows if r[key] == value]
            mean, std, _ = stats(ys)
            # punti delle run singole, spostati a caso in orizzontale per non sovrapporsi
            # (non in ordine: le righe sono ordinate per criterio e disegnerebbero una diagonale)
            ax.scatter(x + rng.uniform(-0.15, 0.15, len(ys)), ys, s=12, color=INK_MUTED,
                       alpha=0.35, linewidths=0)
            ax.errorbar(x, mean, yerr=std, fmt="o", color=BLUE, markersize=8,
                        elinewidth=2, capsize=4)
            ax.annotate(f"{mean:.1f}", (x, mean), xytext=(10, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([f"{label(key, v)}\n(n={sum(r[key] == v for r in rows)})"
                            for v in values])
        ax.set_xlim(-0.6, len(values) - 0.4)
        style(ax, title)
        if ridge:
            ax.axhline(float(ridge[CRITERION]), color=INK_MUTED, linewidth=1, linestyle="--")
    axes[0].set_ylabel("R@10 classe sul val (%)", color=INK_MUTED)
    if ridge:
        axes[0].annotate(f"ridge λ={float(ridge['lambda']):g}", (0, float(ridge[CRITERION])),
                         xycoords=("axes fraction", "data"), xytext=(4, 4),
                         textcoords="offset points", fontsize=9, color=INK_MUTED)
    fig.suptitle(f"Effetti principali sul criterio: media ± dev. std su {len(rows)} run "
                 "(grigio: run singole)", x=0.01, ha="left", fontsize=10, color=INK_MUTED)
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def plot_heatmap(rows, row_key, col_key, out):
    """Media del criterio per ogni coppia di valori di due leve, con il numero di run."""
    rlev, clev = levels(rows, row_key), levels(rows, col_key)
    means = np.full((len(rlev), len(clev)), np.nan)
    counts = np.zeros_like(means, dtype=int)
    for i, rv in enumerate(rlev):
        for j, cv in enumerate(clev):
            ys = [r[CRITERION] for r in rows if r[row_key] == rv and r[col_key] == cv]
            if ys:
                means[i, j], _, counts[i, j] = stats(ys)

    fig, ax = plt.subplots(figsize=(1.6 * len(clev) + 2.4, 1.1 * len(rlev) + 1.4),
                           constrained_layout=True)
    image = ax.imshow(means, cmap=BLUES, aspect="auto")
    low, high = np.nanmin(means), np.nanmax(means)
    for i in range(len(rlev)):
        for j in range(len(clev)):
            if counts[i, j]:
                # testo bianco sulle celle scure, inchiostro sulle chiare
                dark = high > low and (means[i, j] - low) / (high - low) > 0.55
                ax.text(j, i, f"{means[i, j]:.1f}\n(n={counts[i, j]})", ha="center",
                        va="center", fontsize=9, color="white" if dark else INK)
    ax.set_xticks(range(len(clev)))
    ax.set_xticklabels([label(col_key, v) for v in clev])
    ax.set_yticks(range(len(rlev)))
    ax.set_yticklabels([label(row_key, v) for v in rlev])
    ax.set_xlabel(FACTORS[col_key], color=INK_MUTED)
    ax.set_ylabel(FACTORS[row_key], color=INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, shrink=0.9)
    bar.set_label("R@10 classe medio (%)", color=INK_MUTED, fontsize=9)
    bar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    bar.outline.set_visible(False)
    ax.set_title(f"{FACTORS[row_key]} × {FACTORS[col_key]}: R@10 classe sul val",
                 loc="left", fontsize=11, color=INK)
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def config_name(run):
    """Nome della config senza il seed, es. mpnet_h1024_d128_..._taufix."""
    return re.sub(r"_s\d+$", "", run)


def short(row):
    """Config in forma breve per grafici e tabelle a schermo."""
    return (f"d{row['dim']} · p{row['dropout']} · lr {row['lr']:g} · b{row['batch']} "
            f"· τ {row['tau']}")


def seed_summary(top):
    """Per ogni config di top: media e dev. std delle metriche sui SEEDS, e le run trovate."""
    runs = collect(SEEDS, [r["run"] for r in top])
    summary = []
    for r0 in top:
        group = [r for r in runs if config_name(r["run"]) == config_name(r0["run"])]
        row = {"config": config_name(r0["run"]), **{k: r0[k] for k in FACTORS},
               "seed_finiti": len(group)}
        for metric in SEED_METRICS:
            mean, std, _ = stats([r[metric] for r in group])
            row[f"{metric}_media"], row[f"{metric}_std"] = round(mean, 2), round(std, 2)
        summary.append(row)
    return sorted(summary, key=lambda s: s[f"{CRITERION}_media"], reverse=True), runs


def choose(summary):
    """Regola della roadmap: tra le config alla pari con la migliore vince la piu' semplice."""
    mean, std = f"{CRITERION}_media", f"{CRITERION}_std"
    best = summary[0]                                       # summary e' ordinato per media
    tied = [s for s in summary
            if s is best or best[mean] - s[mean] < max(best[std], s[std])]
    return min(tied, key=lambda s: (s["dim"], -s["dropout"], -s[mean])), tied


def plot_seeds(summary, runs, chosen, out):
    """Una riga per config: seed come punti grigi, media ± dev. std; la scelta in blu."""
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(summary) + 1.6), constrained_layout=True)
    for y, s in enumerate(summary):
        ys = [r[CRITERION] for r in runs if config_name(r["run"]) == s["config"]]
        color = BLUE if chosen is None or s is chosen else INK_MUTED
        ax.scatter(ys, [y] * len(ys), s=30, color=INK_MUTED, alpha=0.45, linewidths=0, zorder=2)
        ax.errorbar(s[f"{CRITERION}_media"], y, xerr=s[f"{CRITERION}_std"], fmt="o",
                    color=color, markersize=8, elinewidth=2, capsize=4, zorder=3)
        note = "  ← scelta" if s is chosen else ""
        ax.annotate(f"{s[f'{CRITERION}_media']:.2f} ± {s[f'{CRITERION}_std']:.2f} "
                    f"(n={s['seed_finiti']}){note}", (1, y), xycoords=("axes fraction", "data"),
                    xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(summary)))
    ax.set_yticklabels([short(s) for s in summary])
    ax.invert_yaxis()                                       # la migliore in alto
    ax.set_xlabel("R@10 classe sul val (%)", color=INK_MUTED)
    style(ax, f"Prime {len(summary)} config: seed {', '.join(map(str, SEEDS))} "
              "(grigio: seed singoli, barra: media ± dev. std)")
    ax.grid(axis="x", color=GRID_LINE, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def print_seeds(summary, runs, chosen, tied):
    """Riassunto dei seed e, se tutti finiti, la config scelta con il perche'."""
    print(f"\nseed sulle prime {len(summary)} ({len(runs)}/{len(summary) * len(SEEDS)} run finite):")
    for s in summary:
        print(f"  {short(s):45}{s[f'{CRITERION}_media']:>7.2f} ± {s[f'{CRITERION}_std']:<5.2f}"
              f"esatto R@10 {s['esatto_R@10_media']:>6.2f}   nuove {s['nuove_classe_R@10_media']:>6.2f}"
              f"   seed {s['seed_finiti']}/{len(SEEDS)}")
    if chosen is None:
        print("scelta: in attesa che tutti i seed siano finiti")
        return
    print(f"alla pari con la migliore (distanza < 1 dev. std): {len(tied)}")
    print(f"scelta: {chosen['config']}  ({short(chosen)})")


def print_summary(rows, n_total):
    """Effetti principali, prime 10 run e comando per i seed sulle prime TOP."""
    print(f"run finite: {len(rows)}/{n_total}")
    print("\neffetti principali (media ± dev. std del criterio, n run):")
    for key in FACTORS:
        parts = []
        for value in levels(rows, key):
            mean, std, n = stats([r[CRITERION] for r in rows if r[key] == value])
            parts.append(f"{label(key, value)}: {mean:.2f} ± {std:.2f} (n={n})")
        print(f"  {key:8}" + "   ".join(parts))

    print(f"\n{'#':>3} {'run':52}{CRITERION:>12}{'classe R@1':>12}{'esatto R@10':>13}"
          f"{'nuove R@10':>12}{'epoca':>7}")
    for i, r in enumerate(rows[:10], 1):
        print(f"{i:>3} {r['run']:52}{r[CRITERION]:>12.2f}{r['classe_R@1']:>12.2f}"
              f"{r['esatto_R@10']:>13.2f}{r['nuove_classe_R@10']:>12.2f}"
              f"{r['epoca_migliore']:>4}/{r['epoche']}")

    if len(rows) < n_total:
        print(f"\nattenzione: griglia incompleta, le prime {TOP} possono ancora cambiare")
    print(f"\npasso 5, seed 1 e 2 sulle prime {TOP}:")
    print("python src/training/grid.py --seeds 1 2 --only " +
          " ".join(r["run"] for r in rows[:TOP]))


def main():
    """Raccoglie le run finite, scrive la tabella, disegna i grafici e stampa il riepilogo."""
    rows = collect()
    if not rows:
        raise SystemExit("nessuna run della griglia ancora finita")
    write_csv(OUT, rows)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_main_effects(rows, ridge_reference("mpnet"), FIG_DIR / "p4_effetti_principali.png")
    plot_heatmap(rows, "batch", "tau", FIG_DIR / "p4_heatmap_batch_tau.png")
    plot_heatmap(rows, "lr", "dropout", FIG_DIR / "p4_heatmap_lr_dropout.png")
    print_summary(rows, len(grid.configs()))
    print(f"\nsalvati: {OUT}\n         {FIG_DIR}/p4_effetti_principali.png, p4_heatmap_*.png")

    # seed: solo se c'e' almeno una run oltre al seed 0
    summary, runs = seed_summary(rows[:TOP])
    if len(runs) > len(summary):
        complete = all(s["seed_finiti"] == len(SEEDS) for s in summary)
        chosen, tied = choose(summary) if complete else (None, [])
        write_csv(OUT_SEED, summary)
        plot_seeds(summary, runs, chosen, FIG_DIR / "p4_seed.png")
        print_seeds(summary, runs, chosen, tied)
        print(f"\nsalvati: {OUT_SEED}\n         {FIG_DIR}/p4_seed.png")


if __name__ == "__main__":
    main()
