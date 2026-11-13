"""Loss InfoNCE simmetrica per l'allineamento video-testo.

S = Z_t · Z_v^T / tau   (forma B x B, riga = testo, colonna = video)

La diagonale contiene le coppie giuste; il resto sono negativi in-batch.
L = 1/2 · [ CE(S, diagonale) + CE(S^T, diagonale) ]

Variante multi-positivo (multi-instance): con un'etichetta per riga del batch, sono positive
tutte le coppie con la stessa etichetta (stessa frase, oppure stessa classe verbo/nome), non
solo la diagonale. Il target di ogni riga diventa la distribuzione uniforme sui suoi positivi:
    P[i, j] = 1 se label[i] == label[j]        (la diagonale e' sempre 1)
    target[i] = P[i] / somma(P[i])
Cosi' un video "giusto" nel batch non viene piu' spinto via come se fosse un negativo.
Se tutte le etichette sono diverse il target e' la diagonale e la loss e' identica a quella classica.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F

TAU_INIT = 0.07
TAU_MIN, TAU_MAX = 0.01, 0.5


def info_nce(z_v, z_t, tau, labels=None):
    """Loss InfoNCE simmetrica su un batch di coppie (z_v[i], z_t[i]) gia' normalizzate.

    labels None: positiva solo la diagonale (InfoNCE classica).
    labels tensore (B,): positive tutte le coppie con la stessa etichetta (multi-positivo).
    """
    sim = z_t @ z_v.T / tau
    if labels is None:
        target = torch.arange(len(sim), device=sim.device)
    else:
        # P e' simmetrica, quindi lo stesso target vale per testo->video e video->testo
        pos = (labels[:, None] == labels[None, :]).float()
        target = pos / pos.sum(1, keepdim=True)     # ogni riga somma a 1: uniforme sui positivi
    loss_t2v = F.cross_entropy(sim, target)         # per ogni testo, trova il suo video (la ground trouth è nella digonale)
    loss_v2t = F.cross_entropy(sim.T, target)       # per ogni video, trova il suo testo (la ground truth è nella diagonale)
    return 0.5 * (loss_t2v + loss_v2t)  # media tra le due loss


class InfoNCE(nn.Module):
    """InfoNCE con temperatura tau apprendibile, tenuta in [TAU_MIN, TAU_MAX]."""

    def __init__(self, learnable_tau=True):
        super().__init__()
        # si impara log(tau): cosi' tau resta sempre positiva
        log_tau = torch.tensor(math.log(TAU_INIT))
        self.log_tau = nn.Parameter(log_tau, requires_grad=learnable_tau)

    @property
    def tau(self):
        """Temperatura corrente, con clamp nell'intervallo ammesso."""
        return self.log_tau.exp().clamp(TAU_MIN, TAU_MAX)   # il clamp tiene la temperatura nell'intervallo di sicurezza

    def forward(self, z_v, z_t, labels=None):
        """Loss InfoNCE simmetrica con la temperatura corrente (multi-positivo se labels)."""
        return info_nce(z_v, z_t, self.tau, labels)
