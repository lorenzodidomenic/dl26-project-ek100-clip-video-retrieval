"""Test di overfit su un piccolo sottoinsieme di train (Fase P2).

Prende N coppie video-testo dal train e allena le teste su di esse, full batch.
Deve riuscire a memorizzarle: loss vicina a zero e R@1 vicino al 100% sul sottoinsieme.
Se non ci riesce, c'e' un bug (quasi sempre allineamento delle coppie o diagonale della loss).

Le coppie hanno frasi tutte diverse: due frasi identiche ("pick up cloth") danno lo stesso
vettore testo, e il modello non puo' sapere quale dei loro video e' quello giusto. Con frasi
ripetute R@1 non supera (frasi distinte / N) e la loss non scende sotto sum(k log k) / N.

Usa solo il train. Il dropout e' disattivato, altrimenti il test non puo' arrivare a zero.
Dati: FeaturePairs del train (standardizzato con le statistiche del train intero), poi un
Subset con le N coppie scelte e un DataLoader full batch (un solo batch con tutte le N coppie).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
# pylint: disable=wrong-import-position
from evaluation.metrics import exact_relevance, retrieval_metrics  # noqa: E402
from training.heads import DualHead  # noqa: E402
from training.loss import InfoNCE  # noqa: E402
from utils.dataset import FeaturePairs  # noqa: E402

LOSS_OK = 0.05      # loss finale sotto cui consideriamo il test superato
R1_OK = 95.0        # R@1 finale sul sottoinsieme sopra cui consideriamo il test superato


def distinct_sample(narrations, n, rng):
    """Indici di n coppie a caso, con frasi tutte diverse tra loro."""
    seen, idx = set(), []
    for i in rng.permutation(len(narrations)):
        if narrations[i] not in seen:
            seen.add(narrations[i])
            idx.append(i)
            if len(idx) == n:
                return np.array(idx)
    raise ValueError(f"solo {len(idx)} frasi distinte, meno di {n}")


@torch.no_grad()
def evaluate(model, loader):
    """R@1/5/10 esatto sul sottoinsieme: ogni testo deve trovare il suo video."""
    model.eval()
    video, text, _ = next(iter(loader))                     # full batch: tutto il sottoinsieme
    z_v, z_t = model(video, text)
    sim = (z_t @ z_v.T).numpy()
    return retrieval_metrics(sim, exact_relevance(len(sim)))


def main():
    """Allena le teste su N coppie di train e stampa loss e R@k ogni tot passi."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoder", default="mpnet", choices=["bert", "mpnet", "clip"])
    parser.add_argument("--n", type=int, default=256, help="coppie nel sottoinsieme")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # statistiche di normalizzazione dal train intero, poi si prende il sottoinsieme
    train = FeaturePairs(args.encoder, "train")
    idx = distinct_sample(train.meta["narration"], args.n, rng)
    loader = DataLoader(Subset(train, idx.tolist()), batch_size=args.n, shuffle=False)

    model = DualHead(train.video.shape[1], train.text.shape[1], dropout=0.0)
    loss_fn = InfoNCE()
    params = [*model.parameters(), *loss_fn.parameters()]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)

    print(f"encoder={args.encoder} n={args.n} steps={args.steps} lr={args.lr}")
    print(f"{'passo':>6}{'loss':>10}{'tau':>8}{'R@1':>8}{'R@10':>8}")

    for step in range(1, args.steps + 1):
        model.train()
        for video, text, _ in loader:                       # un solo batch (full batch)
            z_v, z_t = model(video, text)
            loss = loss_fn(z_v, z_t)                        # InfoNCE classica, solo la diagonale
            opt.zero_grad()
            loss.backward()
            opt.step()

        if step == 1 or step % 50 == 0:
            m = evaluate(model, loader)
            print(f"{step:>6}{loss.item():>10.4f}{loss_fn.tau.item():>8.3f}"
                  f"{m['R@1']:>8.1f}{m['R@10']:>8.1f}")

    m = evaluate(model, loader)
    passed = loss.item() < LOSS_OK and m["R@1"] >= R1_OK
    print(f"\nfinale: loss {loss.item():.4f}, R@1 {m['R@1']:.1f}%, R@10 {m['R@10']:.1f}%")
    print("overfit: SUPERATO" if passed else "overfit: NON superato, controlla allineamento e loss")


if __name__ == "__main__":
    main()
