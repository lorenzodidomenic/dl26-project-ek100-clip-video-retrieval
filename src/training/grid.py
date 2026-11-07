"""Griglia di iperparametri della Fase P4: lancia train.py una volta per configurazione.

Griglia completa, 48 config: lr x dropout x tau x batch x d. Gli altri iperparametri restano
ai default di train.py (MPNet, H=1024, weight decay 1e-2, patience 10, CPU).

Ogni run gira in un processo separato, cosi' parte sempre da zero (seed, memoria).
Una run e' fatta se il suo log mostra un training finito (early stopping o ultima epoca)
e il checkpoint esiste: quelle fatte si saltano, anche la run di P3 che e' una delle 48.
Si puo' interrompere (Ctrl+C) e rilanciare: riprende dalla prima run non finita.

Seed sulle migliori (passo 5): --only con i nomi al seed 0 delle config scelte, --seeds 1 2.

Uso:
    python src/training/grid.py --dry-run                 # elenca le run, non lancia niente
    caffeinate -i python src/training/grid.py             # tutta la griglia, seed 0
    python src/training/grid.py --seeds 1 2 --only <nome_s0> <nome_s0> ...
"""
import argparse
import csv
import itertools
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
# pylint: disable=wrong-import-position
from training.train import CKPT_DIR, CRITERION, LOG_DIR, parse_args  # noqa: E402

TRAIN = ROOT / "src/training/train.py"
GRID = {
    "lr": [1e-4, 1e-3],
    "dropout": [0.1, 0.3],
    "fixed_tau": [True, False],
    "batch": [128, 256, 512],
    "dim": [128, 256],
}


def configs():
    """Tutte le combinazioni della griglia, come dict iperparametro -> valore."""
    return [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())]


def to_argv(config, seed):
    """Argomenti di train.py per una config e un seed."""
    argv = ["--lr", f"{config['lr']:g}", "--dropout", str(config["dropout"]),
            "--batch", str(config["batch"]), "--dim", str(config["dim"]), "--seed", str(seed)]
    if config["fixed_tau"]:
        argv.append("--fixed-tau")
    return argv


def is_complete(args):
    """True se la run ha finito il training: early stopping raggiunto o ultima epoca."""
    log, ckpt = LOG_DIR / f"train_{args.name}.csv", CKPT_DIR / f"{args.name}.pt"
    if not (log.exists() and ckpt.exists()):
        return False
    with open(log, newline="", encoding="utf-8") as f:
        values = [float(r[CRITERION]) for r in csv.DictReader(f)]
    if len(values) >= args.epochs:
        return True
    # come in train.py: conta come miglioramento solo un valore strettamente maggiore,
    # quindi l'epoca migliore e' la prima con il massimo
    best = values.index(max(values))
    return len(values) - 1 - best >= args.patience


def plan(seeds, only):
    """Lista di (argv, args) delle run richieste, nell'ordine della griglia."""
    grid = configs()
    if only:
        by_name = {parse_args(to_argv(c, 0)).name: c for c in grid}
        unknown = [n for n in only if n not in by_name]
        if unknown:
            raise SystemExit(f"nomi non presenti nella griglia (servono quelli al seed 0): {unknown}")
        grid = [by_name[n] for n in only]
    runs = []
    for config, seed in itertools.product(grid, seeds):
        argv = to_argv(config, seed)
        runs.append((argv, parse_args(argv)))
    return runs


def main():
    """Lancia le run non ancora finite, una alla volta, e riassume com'e' andata."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--only", nargs="+", help="solo queste config (nomi delle run al seed 0)")
    parser.add_argument("--dry-run", action="store_true", help="elenca le run senza lanciarle")
    args = parser.parse_args()

    runs = plan(args.seeds, args.only)
    todo = [(argv, a) for argv, a in runs if not is_complete(a)]
    print(f"run richieste {len(runs)}: gia' finite {len(runs) - len(todo)}, da fare {len(todo)}")
    if args.dry_run:
        for _, a in runs:
            print(f"  {'fatta  ' if is_complete(a) else 'da fare'}  {a.name}")
        return

    failed, start = [], time.time()
    try:
        for i, (argv, a) in enumerate(todo, 1):
            print(f"\n=== [{i}/{len(todo)}] {a.name}  ({(time.time() - start) / 60:.1f} min) ===",
                  flush=True)
            if subprocess.run([sys.executable, str(TRAIN), *argv], check=False).returncode != 0:
                failed.append(a.name)
    except KeyboardInterrupt:
        print("\ninterrotto: rilancia lo stesso comando per riprendere dalla run non finita")
        sys.exit(130)

    print(f"\nfinito in {(time.time() - start) / 60:.1f} min: "
          f"{len(todo) - len(failed)} run ok, {len(failed)} fallite")
    for name in failed:
        print(f"  fallita: {name}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
