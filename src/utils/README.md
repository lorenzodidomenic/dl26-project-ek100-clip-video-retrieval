# Utils

Auxiliary functions, loggers, visualization scripts, and other helpers.

- `features.py`: lettura dei file di feature (`.npz`) e di `split.csv`; `load_pair` restituisce
  video e testo di uno split allineati per `narration_id`, piu' i metadati.
- `dataset.py`: `FeaturePairs`, il `torch.utils.data.Dataset` usato in training. Un elemento e'
  `(video, testo, etichetta dei positivi)`; le feature sono standardizzate con media e deviazione
  del solo train (val e test ricevono `stats=ds_train.stats`). Contiene anche `positive_labels`
  (diag / frase / classe) e `standardize`, usata pure da `evaluation/evaluate_test.py`.
