"""Training delle teste di proiezione con loss InfoNCE.

Dati: FeaturePairs (utils/dataset.py), un Dataset per split con feature gia' standardizzate,
e un DataLoader per il train (mescolato a ogni epoca) e uno per la loss di val.

Per ogni epoca:
1. il DataLoader del train mescola le clip e le divide in mini-batch;
2. per ogni batch: forward delle due teste, loss InfoNCE, backward, passo di AdamW;
   con --positives frase|classe la loss e' multi-positivo (vedi loss.py), default diag;
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

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


from evaluation.linear_alignment import CRITERION, OUT as RIDGE_LOG  # noqa: E402
from evaluation.linear_alignment import evaluate, unseen_mask  # noqa: E402
from training.heads import DualHead  # noqa: E402
from training.loss import InfoNCE  # noqa: E402
# POSITIVES: quali coppie del batch sono positive nella loss (vedi loss.py e dataset.py):
# diag = solo la propria (InfoNCE classica), frase = stessa narrazione, classe = stesso verbo e nome
from utils.dataset import POSITIVES, FeaturePairs  # noqa: E402
from utils.features import write_csv  # noqa: E402

LOG_DIR = ROOT / "experiments/logs"
CKPT_DIR = ROOT / "experiments/checkpoints"
VAL_SEED = 0        # seed della permutazione fissa dei batch della loss di val


def make_data(encoder, positives, batch, seed):
    """Dataset di train e val e i loro DataLoader.

    Train: mescolato a ogni epoca, con un generatore dedicato al seed della run.
    Val (solo per la loss): stessi batch a ogni epoca, presi da una permutazione fissa. In ordine
    di split.csv i batch conterrebbero clip della stessa sessione (negativi piu' difficili) e la
    loss non sarebbe confrontabile con quella di train; fissa, le sue variazioni tra un'epoca e
    l'altra dipendono solo dal modello e non dal caso.
    """
    ds_tr = FeaturePairs(encoder, "train", positives=positives)
    ds_va = FeaturePairs(encoder, "val", stats=ds_tr.stats, positives=positives)
    train_loader = DataLoader(ds_tr, batch_size=batch, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    perm = torch.randperm(len(ds_va), generator=torch.Generator().manual_seed(VAL_SEED))
    val_loader = DataLoader(Subset(ds_va, perm.tolist()), batch_size=batch, shuffle=False)
    return ds_tr, ds_va, train_loader, val_loader


def train_epoch(model, loss_fn, opt, loader, device):
    """Un passaggio sul train; ritorna la loss media per coppia.

    Ogni batch del loader e' (video, testo, etichette dei positivi), vedi utils/dataset.py.
    """
    model.train()
    total = 0.0
    for video, text, labels in loader:
        video, text, labels = video.to(device), text.to(device), labels.to(device)
        z_v, z_t = model(video, text)
        loss = loss_fn(z_v, z_t, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * len(video) # loss è la media sul batch. per avere la media sull'epoca bisogna pesare ogni batch per la sua dimensione
    return total / len(loader.dataset)


@torch.no_grad()
def validate(model, loss_fn, loader, dataset, unseen, device):  # pylint: disable=too-many-arguments
    """Loss e metriche di retrieval sul val, con il modello in modalita' eval.

    loader: batch della loss, della stessa dimensione del train e sempre uguali (make_data).
    dataset: il val intero, per le metriche.
    La loss di val usa gli stessi positivi del training, le metriche no: sono sempre
    le stesse (esatto, classe, nuove) per qualunque scelta dei positivi.
    """
    model.eval()
    total = 0.0
    for video, text, labels in loader:
        video, text, labels = video.to(device), text.to(device), labels.to(device)
        z_v, z_t = model(video, text)
        total += loss_fn(z_v, z_t, labels).item() * len(video)

    # metriche: ogni frase cerca tra TUTTI i video del val, quindi un solo passaggio sull'intero
    # split (a batch ogni frase vedrebbe solo i video del suo batch)
    z_v, z_t = model(dataset.video.to(device), dataset.text.to(device))
    sim = (z_t @ z_v.T).cpu().numpy()                       # similarità tra tutti frasi e video del val

    # evaluate è la funzione che ordina i video dal più simile al meno simile e calcola rango sia esatto che di classe
    # ogni matrice calcolata due volte: su tutte le query e solo sulle query unseen
    return total / len(loader.dataset), evaluate(sim, dataset.meta, unseen)  # ritorna loss media in validation e valutazione su validation set


def ridge_reference(encoder):
    """Legge il CSV della baseline lineare (ridge regression testo→video).
      Prende la riga migliore per lo stesso encoder, quella con il valore di classe_R@10 più alto. 
      Serve solo a stampare alla fine un confronto: "la MLP batte la baseline lineare?". Se il file non esiste restituisce None."""
    if not RIDGE_LOG.exists():
        return None
    with open(RIDGE_LOG, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["encoder"] == encoder]
    return max(rows, key=lambda r: float(r[CRITERION])) if rows else None

#funzione utile per il nome del run (utile nella griglia degli iperparametri)
def run_name(args):
    """Nome della run costruito dagli iperparametri; lo usa anche grid.py per le run gia' fatte."""
    tau = "taufix" if args.fixed_tau else "taulearn"
    # diag non aggiunge niente al nome: le run fatte prima della variante multi-positivo
    # (griglia di P4 e seed) restano riconosciute e non vengono sovrascritte
    pos = "" if args.positives == "diag" else f"_pos{args.positives}"
    return (f"{args.encoder}_h{args.hidden}_d{args.dim}_p{args.dropout}"
            f"_lr{args.lr:g}_b{args.batch}_{tau}{pos}_s{args.seed}")


def parse_args(argv=None):
    """Iperparametri della run; i default sono quelli della roadmap.

    argv None legge la riga di comando; una lista (es. da grid.py) viene letta al suo posto.
    """
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
    parser.add_argument("--positives", default="diag", choices=POSITIVES,
                        help="coppie positive nella loss: diag (classica), frase, classe")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--name", help="nome della run (default: dagli iperparametri)")
    args = parser.parse_args(argv)
    if args.name is None:
        args.name = run_name(args)
    return args


def main():  # pylint: disable=too-many-locals
    """Training con early stopping sul criterio di val; salva log e checkpoint migliore."""
    args = parse_args()
    torch.manual_seed(args.seed)                            # inizializzazione dei pesi e dropout
    device = torch.device(args.device)

    ds_tr, ds_va, train_loader, val_loader = make_data(
        args.encoder, args.positives, args.batch, args.seed)
    # True per le clip di val la cui coppia (verbo, nome) non compare nel train
    unseen = unseen_mask(ds_tr.meta, ds_va.meta)

    model = DualHead(ds_tr.video.shape[1], ds_tr.text.shape[1],
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
    print(f"run {args.name}: train {len(ds_tr)}, val {len(ds_va)}, device {device}")
    print(f"{'epoca':>5}{'loss tr':>9}{'loss val':>10}{'tau':>7}"
          f"{'esatto R@10':>13}{'classe R@10':>13}{'s':>6}")

    rows, best, waited = [], None, 0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_epoch(model, loss_fn, opt, train_loader, device)
        val_loss, metrics = validate(model, loss_fn, val_loader, ds_va, unseen, device)
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
                        "stats": ds_tr.stats, "args": vars(args), "epoch": epoch,
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
