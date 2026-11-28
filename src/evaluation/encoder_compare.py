"""Confronto degli encoder di testo sul test (P5 fatto dopo P6): MPNet contro BERT.

Non valuta niente: legge i risultati di evaluate_test.py, prodotti con lo stesso codice per i
due encoder, e li mette in una tabella sola.
- MPNet: experiments/final/final_test.csv (P6);
- BERT:  experiments/final/final_test_bert.csv, da
         python src/evaluation/evaluate_test.py --encoder bert --split val    # verifica
         python src/evaluation/evaluate_test.py --encoder bert --split test

P5 e' stato fatto dopo P6: il test era gia' stato visto e il modello finale resta MPNet. Il
confronto e' informativo e va dichiarato come tale. BERT usa la config scelta per MPNet, senza
una griglia sua: confronto a parita' di config.

Per le MLP (media su 3 seed) si applica la regola "alla pari" della roadmap: due valori sono
alla pari se la differenza e' minore della dev. std maggiore delle due; altrimenti si indica
l'encoder migliore (per il rango mediano vince il piu' basso).

Uso: python src/evaluation/encoder_compare.py
Output: experiments/logs/encoder_compare.csv, una riga per modello e encoder (valore e dev. std).
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
# pylint: disable=wrong-import-position
from evaluation.evaluate_test import by_model, outputs  # noqa: E402
from utils.features import write_csv  # noqa: E402

ENCODERS = ("mpnet", "bert")
MODELS = ("caso", "baseline", "ridge", "mlp_diag", "mlp_classe")
METRICS = ("esatto_R@1", "esatto_R@5", "esatto_R@10", "esatto_rango_mediano", "classe_R@1",
           "classe_R@10", "nuove_classe_R@10")
LOWER_IS_BETTER = ("esatto_rango_mediano",)
OUT = ROOT / "experiments/logs/encoder_compare.csv"


def read_results(encoder):
    """{modello: (valori, dev. std o None)} dal CSV del test di un encoder.

    Baseline e MLP: righe "media" e "std" sui seed; caso e ridge: la loro unica riga.
    """
    path = outputs(encoder)["test"]
    if not path.exists():
        raise SystemExit(f"manca {path}: lancia evaluate_test.py --encoder {encoder} --split test")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    def numbers(row):
        return {m: float(row[m]) for m in METRICS}

    return by_model(rows, MODELS, numbers)


def verdict(metric, a, b):
    """Regola della roadmap su due (media, std): "alla pari" o l'encoder migliore."""
    (mean_a, std_a), (mean_b, std_b) = a, b
    if abs(mean_b - mean_a) < max(std_a, std_b):
        return "alla pari"
    b_better = (mean_b < mean_a) if metric in LOWER_IS_BETTER else (mean_b > mean_a)
    return ENCODERS[1] if b_better else ENCODERS[0]


def main():
    """Tabella MPNet contro BERT, differenze sulle MLP e CSV."""
    results = {e: read_results(e) for e in ENCODERS}

    rows = []
    print(f"{'modello':12}{'encoder':8}" + "".join(f"{m:>22}" for m in METRICS))
    for model in MODELS:
        for encoder in ENCODERS:
            values, std = results[encoder][model]
            cells = [f"{values[m]:.2f}" + (f" ± {std[m]:.2f}" if std else "") for m in METRICS]
            print(f"{model:12}{encoder:8}" + "".join(f"{c:>22}" for c in cells))
            row = {"modello": model, "encoder": encoder}
            for m in METRICS:
                row[m] = values[m]
                row[f"{m}_std"] = std[m] if std else ""
            rows.append(row)

    a, b = ENCODERS
    for model in ("mlp_diag", "mlp_classe"):
        print(f"\n{model}: {b} − {a} (regola: alla pari se |Δ| < dev. std maggiore)")
        for m in METRICS:
            (va, sa), (vb, sb) = results[a][model], results[b][model]
            v = verdict(m, (va[m], sa[m]), (vb[m], sb[m]))
            print(f"  {m:22} Δ {vb[m] - va[m]:+7.2f}   std {sa[m]:.2f} / {sb[m]:.2f}   -> {v}")

    write_csv(OUT, rows)
    print(f"\nsalvato: {OUT}")


if __name__ == "__main__":
    main()
