"""Dataset PyTorch delle coppie video-testo di uno split, gia' standardizzate.

Ogni elemento e' una clip: (feature video, feature testo, etichetta dei positivi).
Le feature sono gia' estratte (TimeSformer per il video, encoder di testo per le frasi) e
stanno tutte in memoria: il Dataset non legge file a ogni elemento, li carica una volta sola
con load_pair (che allinea video e testo per narration_id).

Standardizzazione: media e deviazione si calcolano solo sul train. Val e test le ricevono
dal train (stats=ds_train.stats), cosi' nessuna informazione di val o test entra nel modello.

Etichette dei positivi (vedi training/loss.py): due clip con la stessa etichetta sono positive
tra loro nella loss.
- diag:   ogni clip ha un'etichetta diversa (0..n-1), quindi e' positiva solo la diagonale
          e la loss coincide con la InfoNCE classica;
- frase:  stessa narrazione;
- classe: stesso verbo e stesso nome.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.features import load_pair

POSITIVES = ("diag", "frase", "classe")
EPS = 1e-8          # sommato alla deviazione: evita divisioni per zero su feature costanti


def positive_labels(meta, positives):
    """Un'etichetta intera per clip: due clip con la stessa etichetta sono positive tra loro.

    Le etichette servono solo a confrontare righe dello stesso batch, quindi possono essere
    numerate in modo indipendente per ogni split.
    """
    if positives not in POSITIVES:
        raise ValueError(f"positivi sconosciuti: {positives}, ammessi {POSITIVES}")
    n = len(meta["narration_id"])
    if positives == "diag":
        return torch.arange(n)
    if positives == "frase":
        # frase identica = stesso input all'encoder di testo = stesso embedding testuale
        keys = meta["narration"]
    else:
        keys = np.stack([meta["verb_class"], meta["noun_class"]], axis=1)
    _, labels = np.unique(keys, axis=0, return_inverse=True)
    return torch.from_numpy(labels.reshape(-1)).long()


def compute_stats(x):
    """Media e deviazione per colonna di x, (mean, std + EPS).

    correction=0: deviazione di popolazione, come np.std (torch.std di default divide per n-1).
    """
    return x.mean(0), x.std(0, correction=0) + EPS


def standardize(x, stats):
    """(x - media) / deviazione, con stats = (mean, std) del train."""
    mean, std = stats
    return (x - mean) / std


class FeaturePairs(Dataset):
    """Coppie (video, testo, etichetta) di uno split, standardizzate con le statistiche del train.

    encoder:   encoder di testo (bert, mpnet, clip);
    split:     train, val o test;
    stats:     None solo per il train (le calcola lui); per val e test passare ds_train.stats;
    positives: diag, frase o classe (vedi positive_labels).

    Attributi: video (n, 768), text (n, d), labels (n,), meta (dict di array numpy),
    stats {"video": (mean, std), "text": (mean, std)}.
    """

    def __init__(self, encoder, split, stats=None, positives="diag"):
        video, text, meta = load_pair(encoder, split, normalize=False)
        video, text = torch.from_numpy(video).float(), torch.from_numpy(text).float()
        if stats is None:
            if split != "train":
                raise ValueError(f"statistiche calcolate sul {split}: passa stats=ds_train.stats")
            stats = {"video": compute_stats(video), "text": compute_stats(text)}
        self.stats = stats
        self.video = standardize(video, stats["video"])
        self.text = standardize(text, stats["text"])
        self.meta = meta
        self.labels = positive_labels(meta, positives)

    def __len__(self):
        return len(self.video)

    def __getitem__(self, i):
        return self.video[i], self.text[i], self.labels[i]
