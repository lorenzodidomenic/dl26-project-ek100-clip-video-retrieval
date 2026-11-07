"""Training delle teste di proiezione con loss InfoNCE.

Per ogni epoca:
1. mescola il train e lo divide in mini-batch;
2. per ogni batch: forward delle due teste, loss InfoNCE, backward, passo di AdamW;
3. sul val: loss (a batch della stessa dimensione del train) e metriche testo -> video,
   le stesse della baseline lineare (esatto, classe, combinazioni nuove);
4. se il criterio sul val migliora salva il checkpoint, altrimenti conta le epoche senza
   miglioramento e si ferma dopo --patience.

Normalizzazione: media e deviazione dal solo train, salvate nel checkpoint (servono al test).
Il test non viene mai usato qui.

Output: experiments/logs/train_<nome>.csv (una riga per epoca) e
experiments/checkpoints/<nome>.pt (epoca migliore sul val).
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


from evaluation.linear_alignment import CRITERION, OUT as RIDGE_LOG  # noqa: E402
from evaluation.linear_alignment import evaluate, unseen_mask  # noqa: E402
from training.heads import DualHead  # noqa: E402
from training.loss import InfoNCE  # noqa: E402
from utils.features import load_pair, write_csv  # noqa: E402

LOG_DIR = ROOT / "experiments/logs"
CKPT_DIR = ROOT / "experiments/checkpoints"


def load_data(encoder):
    """Train e val come tensori, standardizzati con media e deviazione del solo train."""
    video_tr, text_tr, meta_tr = load_pair(encoder, "train", normalize=False)   #prendiamo tensori training (nx768) e metadati
    video_va, text_va, meta_va = load_pair(encoder, "val", normalize=False) #prendiamo tensori validation (nx768) e metadati
    stats = {"video": (video_tr.mean(0), video_tr.std(0) + 1e-8),
             "text": (text_tr.mean(0), text_tr.std(0) + 1e-8)}      #prendiamo media e dev std video training e text training

    def norm(x, kind):
        mean, std = stats[kind]
        return torch.from_numpy((x - mean) / std).float()

    train = (norm(video_tr, "video"), norm(text_tr, "text"))
    val = (norm(video_va, "video"), norm(text_va, "text"))
    stats = {k: tuple(torch.from_numpy(s).float() for s in v) for k, v in stats.items()}
    return train, val, meta_va, unseen_mask(meta_tr, meta_va), stats    # unseen mask è un array boolenao lungo quanto il val. E' True per le clip di val la cui coppia 'verbo,nome' non compare nel train


def batches(n, size, rng):
    """Divide gli indici 0..n-1, mescolati con rng, in blocchi di al piu' size."""
    order = torch.from_numpy(rng.permutation(n))
    for start in range(0, n, size):
        yield order[start:start + size]


def train_epoch(model, loss_fn, opt, data, batch, rng):
    """Un passaggio sul train; ritorna la loss media per coppia."""
    model.train()
    video, text = data
    total = 0.0
    for idx in batches(len(video), batch, rng):
        z_v, z_t = model(video[idx], text[idx])
        loss = loss_fn(z_v, z_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * len(idx) # loss è la media sul batch. per avere la media sull'epoca bisogna pesare ogni batch per la sua dimensione
    return total / len(video)


@torch.no_grad()
def validate(model, loss_fn, data, batch, meta, unseen):  # pylint: disable=too-many-arguments
    """Loss e metriche di retrieval sul val, con il modello in modalita' eval."""
    model.eval()
    z_v, z_t = model(*data)
    # loss a batch della stessa dimensione del train, sempre con la stessa permutazione:
    # in ordine di split.csv i batch conterrebbero clip della stessa sessione (negativi piu' difficili)
    rng = np.random.default_rng(0)
    # batch da 256 come nel training con seed fisso: permutazione la stessa ogni epoca così che le variazioni della loss dipendono solo dal dal modello e non dal caso
    losses = [loss_fn(z_v[idx], z_t[idx]).item() * len(idx)
              for idx in batches(len(z_v), batch, rng)]
    sim = (z_t @ z_v.T).cpu().numpy()                       # similarità tra tutti frasi e video del val

    # evaluate è la funzione che ordina i video dal più simile al meno simile e calcola rango sia esatto che di classe
    # ogni matrice calcolata due volte: su tutte le query e solo sulle query unseen
    return sum(losses) / len(z_v), evaluate(sim, meta, unseen)  # ritorna loss media in validation e valutazione su validation set


def ridge_reference(encoder):
    """Legge il CSV della baseline lineare (ridge regression testo→video).
      Prende la riga migliore per lo stesso encoder, quella con il valore di classe_R@10 più alto. 
      Serve solo a stampare alla fine un confronto: "la MLP batte la baseline lineare?". Se il file non esiste restituisce None."""
    if not RIDGE_LOG.exists():
        return None
    with open(RIDGE_LOG, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["encoder"] == encoder]
    return max(rows, key=lambda r: float(r[CRITERION])) if rows else None


def parse_args():
    """Iperparametri della run; i default sono quelli della roadmap."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoder", default="mpnet", choices=["bert", "mpnet", "clip"])
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--fixed-tau", action="store_true", help="tau fissa a 0.07")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--name", help="nome della run (default: dagli iperparametri)")
    args = parser.parse_args()
    if args.name is None:
        tau = "taufix" if args.fixed_tau else "taulearn"
        args.name = (f"{args.encoder}_h{args.hidden}_d{args.dim}_p{args.dropout}"
                     f"_lr{args.lr:g}_b{args.batch}_{tau}_s{args.seed}")
    return args


def main():  # pylint: disable=too-many-locals
    """Training con early stopping sul criterio di val; salva log e checkpoint migliore."""
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    train, val, meta_va, unseen, stats = load_data(args.encoder)
    train = tuple(x.to(device) for x in train)
    val = tuple(x.to(device) for x in val)

    model = DualHead(train[0].shape[1], train[1].shape[1],
                     args.hidden, args.dim, args.dropout).to(device)
    loss_fn = InfoNCE(learnable_tau=not args.fixed_tau).to(device)
    # niente weight decay su tau: la spingerebbe verso 1 (log tau verso 0) senza motivo
    opt = torch.optim.AdamW([
        {"params": model.parameters(), "weight_decay": args.weight_decay},
        {"params": loss_fn.parameters(), "weight_decay": 0.0},
    ], lr=args.lr)

    log_path = LOG_DIR / f"train_{args.name}.csv"
    ckpt_path = CKPT_DIR / f"{args.name}.pt"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"run {args.name}: train {len(train[0])}, val {len(val[0])}, device {device}")
    print(f"{'epoca':>5}{'loss tr':>9}{'loss val':>10}{'tau':>7}"
          f"{'esatto R@10':>13}{'classe R@10':>13}{'s':>6}")

    rows, best, waited = [], None, 0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_epoch(model, loss_fn, opt, train, args.batch, rng)
        val_loss, metrics = validate(model, loss_fn, val, args.batch, meta_va, unseen)
        rows.append({"epoca": epoch, "loss_train": round(train_loss, 4),
                     "loss_val": round(val_loss, 4), "tau": round(loss_fn.tau.item(), 4),
                     **metrics})
        write_csv(log_path, rows)                           # riscritto a ogni epoca

        improved = best is None or metrics[CRITERION] > best[CRITERION]
        print(f"{epoch:>5}{train_loss:>9.4f}{val_loss:>10.4f}{loss_fn.tau.item():>7.3f}"
              f"{metrics['esatto_R@10']:>13.2f}{metrics['classe_R@10']:>13.2f}"
              f"{time.time() - start:>6.1f}{'  *' if improved else ''}")

        if improved:
            best, waited = rows[-1], 0
            torch.save({"model": model.state_dict(), "loss": loss_fn.state_dict(),
                        "stats": stats, "args": vars(args), "epoch": epoch,
                        "metrics": metrics}, ckpt_path)
        else:
            waited += 1
            if waited >= args.patience:
                print(f"nessun miglioramento di {CRITERION} da {args.patience} epoche, stop")
                break

    print(f"\nmigliore: epoca {best['epoca']}, {CRITERION} {best[CRITERION]:.2f}, "
          f"esatto R@1/5/10 {best['esatto_R@1']:.2f} / {best['esatto_R@5']:.2f} / "
          f"{best['esatto_R@10']:.2f}, nuove classe R@10 {best['nuove_classe_R@10']:.2f}")
    ref = ridge_reference(args.encoder)
    if ref:
        print(f"ridge (val, λ={float(ref['lambda']):g}): {CRITERION} {float(ref[CRITERION]):.2f}, "
              f"esatto R@10 {float(ref['esatto_R@10']):.2f}, "
              f"nuove classe R@10 {float(ref['nuove_classe_R@10']):.2f}")
    print(f"log: {log_path}\ncheckpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
