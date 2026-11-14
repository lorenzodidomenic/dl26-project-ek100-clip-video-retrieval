"""Callback di Lightning: storico delle metriche per epoca, CSV di log e riga a schermo.

MetricHistoryCallback, come nei laboratori, raccoglie a ogni epoca le metriche in un dizionario
history (nome -> lista di valori). In piu' riscrive il CSV experiments/logs/train_<nome>.csv
con le stesse colonne di prima del passaggio a Lightning (epoca, loss_train, loss_val, tau e le
metriche di val), che leggono grid.py, p4_grid.py e plot_training.py.

Si aggancia a on_train_epoch_end: in Lightning la validazione gira dentro l'epoca di training,
prima di questo hook, quindi qui ci sono sia la loss media di train sia le metriche di val.
"""
import time

import pytorch_lightning as pl

from utils.features import write_csv


class MetricHistoryCallback(pl.Callback):
    """Storico delle metriche: history[nome] = [valore epoca 1, epoca 2, ...].

    log_path: CSV riscritto a ogni epoca (None = nessun file).
    criterion: metrica di val (senza prefisso) per l'asterisco "migliorata" a schermo.
    """

    def __init__(self, log_path=None, criterion="classe_R@10"):
        super().__init__()
        self.log_path, self.criterion = log_path, criterion
        self.history, self.rows = {}, []
        self.best, self._start = None, None

    # pylint: disable=unused-argument
    # (gli hook di Lightning ricevono sempre trainer e pl_module, anche se non servono)

    def on_train_epoch_start(self, trainer, pl_module):
        """Fa partire il cronometro dell'epoca."""
        self._start = time.time()

    def on_fit_start(self, trainer, pl_module):
        """Intestazione della tabella a schermo."""
        print(f"{'epoca':>5}{'loss tr':>9}{'loss val':>10}{'tau':>7}"
              f"{'esatto R@10':>13}{'classe R@10':>13}{'s':>6}")

    def on_train_epoch_end(self, trainer, pl_module):
        """Aggiunge l'epoca a history, riscrive il CSV e stampa la riga."""
        m = {k: float(v) for k, v in trainer.callback_metrics.items()}
        # stesse colonne del CSV di prima: epoca da 1 (Lightning conta da 0), metriche senza "val/";
        # Lightning le tiene come float32 (13.15 -> 13.149999...), quindi si riarrotondano a 2 cifre
        row = {"epoca": trainer.current_epoch + 1, "loss_train": round(m["train/loss"], 4),
               "loss_val": round(m["val/loss"], 4), "tau": round(m["tau"], 4),
               **{k[len("val/"):]: round(v, 2) for k, v in m.items()
                  if k.startswith("val/") and k != "val/loss"}}
        self.rows.append(row)
        for k, v in row.items():
            self.history.setdefault(k, []).append(v)
        if self.log_path is not None:
            write_csv(self.log_path, self.rows)

        # stessa regola di EarlyStopping e ModelCheckpoint: migliora solo se strettamente maggiore
        improved = self.best is None or row[self.criterion] > self.best[self.criterion]
        if improved:
            self.best = row
        print(f"{row['epoca']:>5}{row['loss_train']:>9.4f}{row['loss_val']:>10.4f}"
              f"{row['tau']:>7.3f}{row['esatto_R@10']:>13.2f}{row['classe_R@10']:>13.2f}"
              f"{time.time() - self._start:>6.1f}{'  *' if improved else ''}", flush=True)
