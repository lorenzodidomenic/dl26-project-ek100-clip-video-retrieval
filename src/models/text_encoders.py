"""Text encoder congelati da confrontare. Stessa interfaccia: encode(frasi) -> array (n, dim).

I vettori escono grezzi, senza normalizzazione: la normalizzazione L2 si fa in un solo
punto, al caricamento delle feature, uguale per tutti gli encoder.
"""
# le dipendenze di deep learning non sono installate nel runner di CI (pylint)
# pylint: disable=import-error
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, CLIPTextModelWithProjection, CLIPTokenizer


def default_device():
    """Sceglie MPS se disponibile, altrimenti CPU."""
    return "mps" if torch.backends.mps.is_available() else "cpu"


class BertEncoder:  # pylint: disable=too-few-public-methods
    """BERT generico (masked language modeling). Vettore di frase = media dei token."""

    name = "bert"
    checkpoint = "bert-base-uncased"

    def __init__(self, device=None):
        self.device = device or default_device()
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.model = AutoModel.from_pretrained(self.checkpoint).eval().to(self.device)

    @torch.no_grad()
    def encode(self, texts):
        """Ritorna un vettore per frase, (n, 768), con la media dei token senza padding."""
        tokens = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        hidden = self.model(**tokens).last_hidden_state           # (n, token, 768)
        # (n, token, 1): 1 sui token veri, 0 sul padding
        mask = tokens["attention_mask"].unsqueeze(-1).float()
        mean = (hidden * mask).sum(1) / mask.sum(1)               # media solo sui token veri
        return mean.cpu().numpy()


class MpnetEncoder:  # pylint: disable=too-few-public-methods
    """MPNet addestrato in modo contrastivo su coppie di frasi (confronto con il coseno)."""

    name = "mpnet"
    checkpoint = "sentence-transformers/all-mpnet-base-v2"

    def __init__(self, device=None):
        self.device = device or default_device()
        self.model = SentenceTransformer(self.checkpoint, device=self.device)

    @torch.no_grad()
    def encode(self, texts):
        """Ritorna un vettore per frase, con tokenizzazione e pooling gia' gestiti dal modello."""
        # il modello fa da se' tokenizzazione e mean pooling (e include gia' una normalizzazione L2)
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


class ClipTextEncoder:  # pylint: disable=too-few-public-methods
    """Torre testuale di CLIP, addestrata insieme alle immagini (vettore da 512 dim)."""

    name = "clip"
    checkpoint = "openai/clip-vit-base-patch16"

    def __init__(self, device=None):
        self.device = device or default_device()
        self.tokenizer = CLIPTokenizer.from_pretrained(self.checkpoint)
        self.model = CLIPTextModelWithProjection.from_pretrained(self.checkpoint).eval().to(
            self.device)

    @torch.no_grad()
    def encode(self, texts):
        """Ritorna un vettore per frase, (n, 512)."""
        tokens = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        return self.model(**tokens).text_embeds.cpu().numpy()     # (n, 512)


ENCODERS = {cls.name: cls for cls in [BertEncoder, MpnetEncoder, ClipTextEncoder]}
