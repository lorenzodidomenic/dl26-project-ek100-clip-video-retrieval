"""LightningModule del training: teste di proiezione + loss InfoNCE + metriche di retrieval.

RetrievalTask: tiene insieme modello (DualHead),
loss (InfoNCE, con tau) e ottimizzatore, e dice a Lightning cosa fare in ogni passo.
Il ciclo (epoche, backward, early stopping, checkpoint, device) lo gestisce pl.Trainer.

Train:  per ogni batch loss InfoNCE, loggata come "train/loss" (media pesata sull'epoca).
Val:    per ogni batch la loss ("val/loss") e gli embedding z_v, z_t, accumulati;
        a fine epoca la matrice frasi x video sull'INTERO val e le metriche di retrieval
        ("val/esatto_R@10", "val/classe_R@10", ...): ogni frase cerca tra tutti i video,
        quindi non si possono calcolare batch per batch.

Le statistiche di standardizzazione del train sono buffer del modulo: finiscono nel
checkpoint insieme ai pesi e servono a evaluate_test.py per standardizzare il test.
"""
import pytorch_lightning as pl
import torch

from evaluation.linear_alignment import evaluate
from models.heads import DualHead
from training.loss import InfoNCE


class RetrievalTask(pl.LightningModule):  # pylint: disable=too-many-ancestors
    """Allineamento video-testo con InfoNCE; metriche testo -> video sul val.

    stats:      {"video": (mean, std), "text": (mean, std)} del train (FeaturePairs.stats);
                None solo quando si ricarica da checkpoint (i valori arrivano dal state_dict).
    val_meta:   metadati del val NELL'ORDINE in cui li produce il val loader;
    val_unseen: maschera delle combinazioni verbo/nome nuove, nello stesso ordine.
    val_meta e val_unseen servono solo in training; per la valutazione si lasciano a None.
    """

    # lr e weight_decay non si usano qui: save_hyperparameters li salva in self.hparams,
    # e configure_optimizers li legge da li'
    # pylint: disable-next=too-many-arguments,too-many-positional-arguments,too-many-locals
    def __init__(self, video_dim, text_dim, hidden=1024, dim=256, dropout=0.1,
                 lr=1e-3, weight_decay=1e-2,  # pylint: disable=unused-argument
                 fixed_tau=False, stats=None, val_meta=None, val_unseen=None):
        super().__init__()
        # iperparametri nel checkpoint (per load_from_checkpoint) e nella config di wandb;
        # stats e val_* sono dati, non iperparametri
        self.save_hyperparameters(ignore=["stats", "val_meta", "val_unseen"])
        self.model = DualHead(video_dim, text_dim, hidden, dim, dropout)
        self.loss_fn = InfoNCE(learnable_tau=not fixed_tau)

        # buffer: salvati nel state_dict ma non allenati; zeri se si ricarica da checkpoint
        for kind, size in (("video", video_dim), ("text", text_dim)):
            mean, std = stats[kind] if stats else (torch.zeros(size), torch.ones(size))
            self.register_buffer(f"{kind}_mean", mean.clone())
            self.register_buffer(f"{kind}_std", std.clone())

        self.val_meta, self.val_unseen = val_meta, val_unseen
        self._val_z = []                                    # (z_v, z_t) dei batch di val

    def stats(self, kind):
        """(mean, std) del train per "video" o "text", come in FeaturePairs.stats."""
        return getattr(self, f"{kind}_mean"), getattr(self, f"{kind}_std")

    def forward(self, video, text):  # pylint: disable=arguments-differ
        """(z_v, z_t) normalizzati L2, da feature gia' standardizzate."""
        return self.model(video, text)

    def training_step(self, batch, batch_idx):  # pylint: disable=arguments-differ,unused-argument
        """Loss InfoNCE di un batch di train; Lightning fa backward e passo dell'ottimizzatore."""
        video, text, labels = batch
        z_v, z_t = self(video, text)
        loss = self.loss_fn(z_v, z_t, labels)
        # on_epoch: media sull'epoca pesata con batch_size, come total += loss * len(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, batch_size=len(video))
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=arguments-differ,unused-argument
        """Loss di un batch di val; gli embedding si tengono per le metriche di fine epoca."""
        video, text, labels = batch
        z_v, z_t = self(video, text)
        loss = self.loss_fn(z_v, z_t, labels)
        self.log("val/loss", loss, on_epoch=True, batch_size=len(video))
        self._val_z.append((z_v, z_t))

    def on_validation_epoch_end(self):
        """Metriche di retrieval sull'intero val, dagli embedding accumulati."""
        z_v = torch.cat([z for z, _ in self._val_z])
        z_t = torch.cat([z for _, z in self._val_z])
        self._val_z.clear()
        # il controllo iniziale di Lightning (sanity check) gira solo su 2 batch:
        # le metriche sull'intero val non hanno senso, quindi si saltano
        if self.trainer.sanity_checking:
            return
        sim = (z_t @ z_v.T).cpu().numpy()          # similarita' tra tutte le frasi e i video
        metrics = evaluate(sim, self.val_meta, self.val_unseen)
        self.log_dict({f"val/{k}": v for k, v in metrics.items()})
        self.log("tau", self.loss_fn.tau)                   # tau a fine epoca

    def configure_optimizers(self):
        """AdamW con due gruppi: le teste con weight decay, tau senza."""
        # niente weight decay su tau: la spingerebbe verso 1 (log tau verso 0) senza motivo
        return torch.optim.AdamW([
            {"params": self.model.parameters(), "weight_decay": self.hparams.weight_decay},
            {"params": self.loss_fn.parameters(), "weight_decay": 0.0},
        ], lr=self.hparams.lr)
