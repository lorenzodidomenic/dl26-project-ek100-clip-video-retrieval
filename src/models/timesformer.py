"""Encoder video congelato: TimeSformer (Kinetics-400). Un vettore da 768 dim per clip."""
# le dipendenze di deep learning non sono installate nel runner di CI (pylint)
# pylint: disable=import-error
import torch
from transformers import AutoImageProcessor, TimesformerModel

# Video Encoder scelto (è stato valutato anche EgoVLP ma poi si è svelta questa strada)
CHECKPOINT = "facebook/timesformer-base-finetuned-k400"


class TimesformerEncoder:  # pylint: disable=too-few-public-methods
    """Encoder TimeSformer che restituisce il token CLS per ogni clip."""

    def __init__(self, device=None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        # preprocessing del checkpoint
        self.processor = AutoImageProcessor.from_pretrained(CHECKPOINT)
        # video encoder senza la testa di classificazione (non ci serve la predizione).
        # In modalità eval disattiva i comportamenti di training; i pesi vanno sul device scelto
        self.model = TimesformerModel.from_pretrained(CHECKPOINT).eval().to(self.device)

    # dice a Pytorch di non costruire il grafo dei gradienti
    @torch.no_grad()
    def encode(self, clips):
        """Ritorna un array (n, 768) con il CLS di ogni clip."""
        # clips: lista di clip, ognuna lista di 8 immagini PIL. Ritorna array (n, 768).
        pixel_values = self.processor(clips, return_tensors="pt")["pixel_values"].to(self.device)
        # 1568 token di patch e uno [CLS]: noi prendiamo solo il CLS che riassume la clip
        cls = self.model(pixel_values=pixel_values).last_hidden_state[:, 0]
        return cls.cpu().numpy()
