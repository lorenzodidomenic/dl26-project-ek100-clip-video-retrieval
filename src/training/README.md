# Training

Training loops, custom loss functions, and optimizers.

- `loss.py`: InfoNCE simmetrica con temperatura apprendibile, anche multi-positivo.
- `task.py`: `RetrievalTask`, il `LightningModule` (teste + loss + AdamW). Le metriche di
  retrieval sul val si calcolano a fine epoca sull'intero val, dagli embedding accumulati.
  Le statistiche di standardizzazione del train sono buffer, quindi stanno nel checkpoint.
- `callbacks.py`: `MetricHistoryCallback`, storico delle metriche per epoca (`history`), CSV di
  log `experiments/logs/train_<nome>.csv` e riga a schermo.
- `train.py`: `pl.Trainer` con `EarlyStopping` e `ModelCheckpoint` su `val/classe_R@10`,
  `WandbLogger` (progetto `dl26-ek100-clip-retrieval`). Dati da `FeaturePairs` e `DataLoader`
  (`src/utils/dataset.py`). Checkpoint in `experiments/checkpoints/<nome>.ckpt`.
- `grid.py`: lancia `train.py` su tutte le combinazioni della griglia di iperparametri
  (`--group` e `--wandb-mode` passano a wandb).
- `overfit_check.py`: sanity check, le teste devono memorizzare N coppie del train.
