"""Training delle teste di proiezione con loss InfoNCE, con PyTorch Lightning e wandb.

Pezzi (come nei laboratori):
- dati: FeaturePairs (utils/dataset.py) e due DataLoader, train mescolato a ogni epoca e val
  con batch sempre uguali (permutazione fissa) per la loss;
- RetrievalTask (training/task.py): il LightningModule con DualHead, InfoNCE e AdamW;
  con --positives frase|classe la loss e' multi-positivo (vedi loss.py), default diag;
- pl.Trainer con le callback:
  EarlyStopping   si ferma dopo --patience epoche senza miglioramento del criterio sul val,
  ModelCheckpoint salva l'epoca migliore sul val,
  MetricHistoryCallback (training/callbacks.py) storico delle metriche, CSV e riga a schermo;
- WandbLogger: curve, metriche e iperparametri su wandb (progetto WANDB_PROJECT).

Metriche sul val a ogni epoca: testo -> video, le stesse della baseline lineare
(esatto, classe, combinazioni nuove), sempre sull'intero val.
Normalizzazione: media e deviazione dal solo train, salvate nel checkpoint (servono al test).
Il test non viene mai usato qui.

Output: experiments/logs/train_<nome>.csv (una riga per epoca),
experiments/checkpoints/<nome>.ckpt (epoca migliore sul val) e la run su wandb.
"""
import argparse
import csv
import sys
import warnings
from pathlib import Path

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


from evaluation.linear_alignment import CRITERION, OUT as RIDGE_LOG  # noqa: E402
from evaluation.linear_alignment import unseen_mask  # noqa: E402
from training.callbacks import MetricHistoryCallback  # noqa: E402
from training.task import RetrievalTask  # noqa: E402
# POSITIVES: quali coppie del batch sono positive nella loss (vedi loss.py e dataset.py):
# diag = solo la propria (InfoNCE classica), frase = stessa narrazione, classe = stesso verbo e nome
from utils.dataset import POSITIVES, FeaturePairs  # noqa: E402

LOG_DIR = ROOT / "experiments/logs"
CKPT_DIR = ROOT / "experiments/checkpoints"
WANDB_DIR = ROOT / "experiments"     # wandb crea qui la cartella wandb/ (ignorata da git)
WANDB_PROJECT = "dl26-ek100-clip-retrieval"
VAL_SEED = 0        # seed della permutazione fissa dei batch della loss di val

# i dati sono gia' tutti in memoria: processi in piu' per caricarli non servono
warnings.filterwarnings("ignore", ".*does not have many workers.*")


def make_data(encoder, positives, batch, seed):
    """Dataset di train e val, i loro DataLoader e l'ordine in cui il val loader produce le clip.

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
    perm = torch.randperm(len(ds_va), generator=torch.Generator().manual_seed(VAL_SEED)).numpy()
    val_loader = DataLoader(Subset(ds_va, perm.tolist()), batch_size=batch, shuffle=False)
    return ds_tr, ds_va, train_loader, val_loader, perm


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
    # CPU di default (roadmap P4): con un modello cosi' piccolo MPS guadagna poco, e su MPS
    # lo stesso seed non garantisce lo stesso risultato
    parser.add_argument("--accelerator", default="cpu", choices=["auto", "cpu", "mps", "cuda"],
                        help="device di pl.Trainer (default cpu; auto: GPU/MPS se c'e')")
    parser.add_argument("--name", help="nome della run (default: dagli iperparametri)")
    parser.add_argument("--group",
                        help="gruppo della run su wandb (es. griglia, seed, loss, finale)")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"],
                        help="offline: salva in locale, poi 'wandb sync'; disabled: niente wandb")
    args = parser.parse_args(argv)
    if args.name is None:
        args.name = run_name(args)
    return args


def main():  # pylint: disable=too-many-locals
    """Training con early stopping sul criterio di val; salva log e checkpoint migliore."""
    args = parse_args()
    pl.seed_everything(args.seed)                           # inizializzazione dei pesi e dropout

    ds_tr, ds_va, train_loader, val_loader, perm = make_data(
        args.encoder, args.positives, args.batch, args.seed)
    # True per le clip di val la cui coppia (verbo, nome) non compare nel train
    unseen = unseen_mask(ds_tr.meta, ds_va.meta)
    # il task riceve gli embedding di val nell'ordine del val loader (perm): metadati e maschera
    # vanno riordinati allo stesso modo, cosi' la riga i della similarita' e' ancora la clip i
    task = RetrievalTask(ds_tr.video.shape[1], ds_tr.text.shape[1], args.hidden, args.dim,
                         args.dropout, args.lr, args.weight_decay, args.fixed_tau,
                         stats=ds_tr.stats,
                         val_meta={k: v[perm] for k, v in ds_va.meta.items()},
                         val_unseen=unseen[perm])

    monitor = f"val/{CRITERION}"
    # min_delta 0 e confronto stretto: migliora solo un valore strettamente maggiore,
    # quindi l'epoca salvata e' la prima con il massimo (come assume anche p4_grid.py)
    early_stop = EarlyStopping(monitor=monitor, mode="max", patience=args.patience)
    checkpoint = ModelCheckpoint(dirpath=CKPT_DIR, filename=args.name, monitor=monitor,
                                 mode="max", save_top_k=1, save_weights_only=True,
                                 enable_version_counter=False)   # sovrascrive, niente <nome>-v1
    history = MetricHistoryCallback(LOG_DIR / f"train_{args.name}.csv", CRITERION)

    logger = WandbLogger(project=WANDB_PROJECT, name=args.name, group=args.group,
                         save_dir=str(WANDB_DIR), mode=args.wandb_mode, config=vars(args))
    trainer = pl.Trainer(max_epochs=args.epochs, accelerator=args.accelerator, devices=1,
                         logger=logger, callbacks=[early_stop, checkpoint, history],
                         enable_progress_bar=False,         # una riga per epoca dalla callback
                         enable_model_summary=False)

    print(f"run {args.name}: train {len(ds_tr)}, val {len(ds_va)}, "
          f"device {trainer.accelerator.__class__.__name__}")
    trainer.fit(task, train_loader, val_loader)

    if early_stop.stopped_epoch > 0:
        print(f"nessun miglioramento di {CRITERION} da {args.patience} epoche, stop")
    best = history.best
    print(f"\nmigliore: epoca {best['epoca']}, {CRITERION} {best[CRITERION]:.2f}, "
          f"esatto R@1/5/10 {best['esatto_R@1']:.2f} / {best['esatto_R@5']:.2f} / "
          f"{best['esatto_R@10']:.2f}, nuove classe R@10 {best['nuove_classe_R@10']:.2f}")
    ref = ridge_reference(args.encoder)
    if ref:
        print(f"ridge (val, λ={float(ref['lambda']):g}): {CRITERION} {float(ref[CRITERION]):.2f}, "
              f"esatto R@10 {float(ref['esatto_R@10']):.2f}, "
              f"nuove classe R@10 {float(ref['nuove_classe_R@10']):.2f}")
    print(f"log: {history.log_path}\ncheckpoint: {checkpoint.best_model_path}")
    wandb.finish()


if __name__ == "__main__":
    main()
