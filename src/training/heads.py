"""Teste di proiezione: una MLP per modalita', che porta video e testo in uno spazio comune.

v -> Linear(D_v, H) -> GELU -> Dropout(p) -> Linear(H, d) -> L2-norm -> z_v
t -> Linear(D_t, H) -> GELU -> Dropout(p) -> Linear(H, d) -> L2-norm -> z_t

L2-norm: ogni vettore sta sulla sfera unitaria, quindi il prodotto scalare e' il coseno.
"""
import torch
from torch import nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """MLP a due strati con uscita normalizzata L2."""

    def __init__(self, in_dim, hidden=1024, out_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        """Ritorna le righe di x proiettate e normalizzate a norma 1."""
        return F.normalize(self.net(x), dim=-1)


class DualHead(nn.Module):
    """Due teste indipendenti, una per il video e una per il testo."""

    def __init__(self, video_dim, text_dim, hidden=1024, out_dim=256, dropout=0.1):
        super().__init__()
        self.video = ProjectionHead(video_dim, hidden, out_dim, dropout)
        self.text = ProjectionHead(text_dim, hidden, out_dim, dropout)

    def forward(self, video, text):
        """Ritorna (z_v, z_t), entrambi (n, out_dim) e normalizzati."""
        return self.video(video), self.text(text)

