"""Loss InfoNCE simmetrica per l'allineamento video-testo.

S = Z_t · Z_v^T / tau   (forma B x B, riga = testo, colonna = video)

La diagonale contiene le coppie giuste; il resto sono negativi in-batch.
L = 1/2 · [ CE(S, diagonale) + CE(S^T, diagonale) ]
"""
import math

import torch
from torch import nn
import torch.nn.functional as F

TAU_INIT = 0.07
TAU_MIN, TAU_MAX = 0.01, 0.5


def info_nce(z_v, z_t, tau):
    """Loss InfoNCE simmetrica su un batch di coppie (z_v[i], z_t[i]) gia' normalizzate."""
    sim = z_t @ z_v.T / tau
    target = torch.arange(len(sim), device=sim.device)
    loss_t2v = F.cross_entropy(sim, target)         # per ogni testo, trova il suo video
    loss_v2t = F.cross_entropy(sim.T, target)       # per ogni video, trova il suo testo
    return 0.5 * (loss_t2v + loss_v2t)


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
        return self.log_tau.exp().clamp(TAU_MIN, TAU_MAX)

    def forward(self, z_v, z_t):
        """Loss InfoNCE simmetrica con la temperatura corrente."""
        return info_nce(z_v, z_t, self.tau)
