"""Encoder video congelato: TimeSformer (Kinetics-400). Un vettore da 768 dim per clip."""
import torch
from transformers import AutoImageProcessor, TimesformerModel

# Video Encoder scelto (è stato valutato anche EgoVLP ma poi si è svelta questa strada)
CHECKPOINT = "facebook/timesformer-base-finetuned-k400"


class TimesformerEncoder:
    def __init__(self, device=None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(CHECKPOINT)     #carica il Preprocessing del checkpoint
        self.model = TimesformerModel.from_pretrained(CHECKPOINT).eval().to(self.device)    #carica il video encoder senza la testa della classificazione (non ci serve la predizione). Lo mette in modalità eval cosi da disattivae i comportamenti di training, spostando i pesi sul dispositivo scelto

    # dice a Pythorc di non costruire il grafo dei gradienti
    @torch.no_grad()
    def encode(self, clips):
        # clips: lista di clip, ognuna lista di 8 immagini PIL. Ritorna array (n, 768).
        pixel_values = self.processor(clips, return_tensors="pt")["pixel_values"].to(self.device)
        # l'encoder prende quello dal processor e ritorna un last hidden state (1568 token di patch e uno [CLS], noi prendiamo solo il CLS che riassume la clip)
        cls = self.model(pixel_values=pixel_values).last_hidden_state[:, 0]
        return cls.cpu().numpy()
