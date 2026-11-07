"""Test di overfit su un piccolo sottoinsieme di train (Fase P2).

Prende N coppie video-testo dal train e allena le teste su di esse, full batch.
Deve riuscire a memorizzarle: loss vicina a zero e R@1 vicino al 100% sul sottoinsieme.
Se non ci riesce, c'e' un bug (quasi sempre allineamento delle coppie o diagonale della loss).

Le coppie hanno frasi tutte diverse: due frasi identiche ("pick up cloth") danno lo stesso
vettore testo, e il modello non puo' sapere quale dei loro video e' quello giusto. Con frasi
ripetute R@1 non supera (frasi distinte / N) e la loss non scende sotto sum(k log k) / N.

Usa solo il train. Il dropout e' disattivato, altrimenti il test non puo' arrivare a zero.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
# pylint: disable=wrong-import-position
from evaluation.metrics import exact_relevance, retrieval_metrics  # noqa: E402
from training.heads import DualHead  # noqa: E402
from training.loss import InfoNCE  # noqa: E402
from utils.features import load_pair  # noqa: E402

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


def standardize(train, other):
    """Standardizza entrambe le matrici con media e deviazione del solo train."""
    mean, std = train.mean(0), train.std(0) + 1e-8
    return (train - mean) / std, (other - mean) / std


@torch.no_grad()
def evaluate(model, video, text):
    """R@1/5/10 esatto sul sottoinsieme: ogni testo deve trovare il suo video."""
    model.eval()
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
    video_all, text_all, meta = load_pair(args.encoder, "train", normalize=False)
    video_all, _ = standardize(video_all, video_all)
    text_all, _ = standardize(text_all, text_all)
    idx = distinct_sample(meta["narration"], args.n, rng)
    video = torch.from_numpy(video_all[idx]).float()
    text = torch.from_numpy(text_all[idx]).float()

    model = DualHead(video.shape[1], text.shape[1], dropout=0.0)
    loss_fn = InfoNCE()
    params = [*model.parameters(), *loss_fn.parameters()]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)

    print(f"encoder={args.encoder} n={args.n} steps={args.steps} lr={args.lr}")
    print(f"{'passo':>6}{'loss':>10}{'tau':>8}{'R@1':>8}{'R@10':>8}")

    for step in range(1, args.steps + 1):
        model.train()
        z_v, z_t = model(video, text)
        loss = loss_fn(z_v, z_t)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step == 1 or step % 50 == 0:
            m = evaluate(model, video, text)
            print(f"{step:>6}{loss.item():>10.4f}{loss_fn.tau.item():>8.3f}"
                  f"{m['R@1']:>8.1f}{m['R@10']:>8.1f}")

    m = evaluate(model, video, text)
    passed = loss.item() < LOSS_OK and m["R@1"] >= R1_OK
    print(f"\nfinale: loss {loss.item():.4f}, R@1 {m['R@1']:.1f}%, R@10 {m['R@10']:.1f}%")
    print("overfit: SUPERATO" if passed else "overfit: NON superato, controlla allineamento e loss")


if __name__ == "__main__":
    main()
