# Roadmap P6: valutazione finale sul test

Fase P6 di [roadmap_fase5.md](roadmap_fase5.md). Si misurano **una volta sola**, sul test, i modelli
gia' scelti sul val, e si compila la tabella finale della relazione.

**Regola di fondo.** Dopo aver visto i numeri del test non si cambia piu' nulla: ne' modello, ne'
iperparametri, ne' scelta del seed. Rilanciare lo stesso script e ottenere gli stessi numeri non e'
"usare il test due volte"; lo diventa se, dopo averli visti, si modifica qualcosa e si rivaluta.

**Seconda esecuzione (pipeline v2).** P6 era gia' stato eseguito una volta con la pipeline
precedente (training senza DataLoader/Lightning; modello finale b512): risultati archiviati in
`experiments/archivio_v1/final/`. Dopo il refactor (Dataset/DataLoader, PyTorch Lightning, wandb)
griglia, seed, ablation della loss e mini-griglia sono stati rifatti da zero con le stesse regole
di scelta; questa e' la valutazione della nuova pipeline. Nella relazione va dichiarato che il
test era gia' stato visto una volta e che le regole di scelta non sono cambiate.

---

## Punto di partenza

| Cosa | Dove |
|---|---|
| Modello finale: MPNet, H=1024, d=128, dropout 0.3, lr 1e-4, batch 256, τ fissa, loss `classe` (3 seed) | `experiments/final/classe_s{0,1,2}.ckpt` |
| Confronto: stessa config, loss `diag` (InfoNCE classica, 3 seed) | `experiments/final/diag_s{0,1,2}.ckpt` |
| Storia della scelta, metriche val, sha256 dei file | `experiments/final/manifest.json` |
| Ridge sul val (dataset completo): λ = 1000 | `experiments/logs/linear_alignment.csv` |
| Baseline casuale sul test (solo metriche esatte) | `experiments/logs/baseline_zero_shot.csv` |

Valori sul val (media ± dev. std su 3 seed):

| Modello | esatto R@10 | classe R@10 | nuove classe R@10 |
|---|---|---|---|
| Ridge (λ=1000) | 6.09 | 52.89 | 8.27 |
| MLP `diag` | 15.89 ± 0.22 | 72.17 ± 0.56 | 17.38 ± 1.88 |
| **MLP `classe`** | **13.80 ± 0.05** | **75.51 ± 0.62** | **17.46 ± 1.04** |

---

## Decisioni fissate prima del test

Vanno scritte e committate **prima** del passo 4. Sono la "pre-registrazione" della valutazione.

**D1. Modelli valutati (righe della tabella).**

| # | Modello | Cosa rappresenta |
|---|---|---|
| 0 | Caso | pavimento: esatto analitico (R@K = K/N), classe con similarita' casuali |
| 1 | Baseline: proiezioni lineari casuali, non allenate | obiettivo minimo 1 della consegna |
| 2 | Ridge lineare, λ=1000 scelto sul val | riferimento lineare allenato |
| 3 | MLP, loss `diag`, seed 0/1/2 | loss contrastiva classica |
| 4 | **MLP, loss `classe`, seed 0/1/2** | **modello finale** |

Tutte le righe si calcolano nello **stesso script** (`evaluate_test.py`), con la stessa funzione di
metriche, cosi' la tabella e' omogenea. La baseline si ricalcola con la stessa logica di
`baseline_zero_shot.py` (seed 42..51, seed del testo = seed + 1 000 000, `proj_dim` 256) e si media
sui 10 seed. Rifarla e' lecito: non ha nulla da scegliere.

**D2. Metriche** (consegna, obiettivo 4). Direzione testo → video. R@1, R@5, R@10 e rango mediano,
in due versioni di "giusto":
- **esatto**: lo stesso `narration_id`;
- **classe**: stesso `verb_class` e stesso `noun_class` (lettura multi-instance).

Ogni query cerca sempre tra **tutti** i video del test.

**D3. Zero-shot** (consegna: "entirely unseen verbs/nouns"). "Mai visto" = assente dal **train**,
l'unico split da cui il modello ha imparato. Nome = `noun_class` (il nome principale).

| Sottoinsieme | Clip nel test | Uso |
|---|---|---|
| combinazione verbo+nome mai vista | 420 | **metrica zero-shot principale** (`nuove_*`) |
| nome mai visto | 24 | riportato, con l'avvertenza che e' troppo piccolo per essere affidabile |
| verbo mai visto | 2 | solo analisi qualitativa, query per query |

**D4. Aggregazione.** Per `diag` e `classe`: media ± deviazione standard campionaria (ddof=1) sui
3 seed. Mai il seed migliore sul test. Le figure usano il **seed 0**, fissato ora.

**D5. Criterio di successo** (gia' in roadmap §2). Il modello finale deve battere il ridge su
**esatto R@10** e su **classe R@10**, sul test. Se non ci riesce, si dichiara.

---

## Passi

### Passo 1 · Congelare le decisioni
- Rileggere la sezione "Decisioni fissate" e correggerla se serve.
- Commit di questo file e di `experiments/final/manifest.json` (prima esecuzione sul branch
  `eval/test-set-evaluation`; seconda, pipeline v2, sul branch `refactor/use-standard-tools`).

**Controllo:** il commit esiste ed e' precedente a qualsiasi file di risultati del test.
**Annotare:** hash del commit.

### Passo 2 · Scrivere `src/evaluation/evaluate_test.py`
Cosa fa, nell'ordine:
1. Legge `experiments/final/manifest.json` e verifica lo **sha256** di ogni checkpoint. Se non
   coincide si ferma.
2. Carica train e split da valutare (`--split val` oppure `test`) con `load_pair("mpnet", ...)`,
   allineati per `narration_id`.
3. Costruisce le maschere zero-shot rispetto al train: combinazione, nome, verbo (D3).
4. Calcola la matrice di similarita' (frasi × video) per ogni modello:
   - caso e baseline come in D1;
   - ridge: standardizzazione con media e dev. del train, `W` dal train con λ letto dal log di val;
   - MLP: normalizzazione con le statistiche del train **salvate nel checkpoint**, `model.eval()`
     (niente dropout), uscite gia' normalizzate L2 dalle teste, quindi il prodotto e' il coseno.
5. Per ogni matrice: metriche esatto e classe su tutte le query e sui sottoinsiemi zero-shot.
6. Scrive una riga per modello e seed, piu' le righe aggregate (media ± dev. std).
7. Salva anche i **ranghi per query** (esatto e classe) di ogni modello: servono alle figure e a P7.

Output:
- `--split val`: `experiments/final/check_val.csv` (solo verifica, passo 3);
- `--split test`: `experiments/final/final_test.csv` e `experiments/final/final_test_ranks.npz`.

Stanno in `experiments/final/`, che e' su git (tranne i checkpoint `.pt`/`.ckpt`): i risultati del test restano
versionati, a differenza di `experiments/logs/`.

**Controllo:** rilettura del codice insieme prima di lanciarlo.
**Annotare:** come vengono normalizzati i dati per ogni modello, e perche'.

### Passo 3 · Verifica della pipeline sul val
```bash
python src/evaluation/evaluate_test.py --split val
```
Il test non viene toccato. I numeri devono **riprodurre** quelli gia' noti:

| Riga | Deve coincidere con |
|---|---|
| MLP `diag` e `classe`, ogni seed | metriche `val` del `manifest.json` |
| Ridge | riga λ=1000 di `experiments/logs/linear_alignment.csv` |
| Baseline | circa il caso (esatto R@10 ≈ 10/N) |

**Controllo:** differenze nulle (al massimo arrotondamento a 0.01). Se qualcosa non coincide, si
corregge lo script e si ripete il passo 3. Il test resta chiuso finche' il val non torna.
**Annotare:** esito della verifica, riga per riga.

### Passo 4 · Valutazione sul test (una sola volta)
```bash
python src/evaluation/evaluate_test.py --split test
```
Subito dopo, commit di `final_test.csv` e `final_test_ranks.npz`. Lo script crea anche la run
wandb `test-finale` (gruppo `test`): tabella completa e medie dei modelli nel summary. E' solo
una copia per la dashboard; la fonte ufficiale resta `final_test.csv`.

**Controllo:** a schermo il numero di clip (9668) e delle query zero-shot (420 / 24 / 2).
**Annotare:** data e ora, hash del commit, tabella stampata.

### Passo 5 · Lettura dei risultati
Domande a cui rispondere, nell'ordine:
1. **Criterio di successo (D5):** `classe` batte il ridge su esatto R@10 e classe R@10?
2. **Ordine dei modelli:** caso < baseline < ridge < MLP, come sul val?
3. **Trade-off della loss:** `classe` contro `diag` su classe R@K (atteso +) ed esatto R@K (atteso −).
4. **Zero-shot:** `nuove_*` per tutti i modelli; nomi mai visti con l'avvertenza di D3.
5. **Scarto val → test:** di quanto cambiano le metriche? Uno scarto grande e sistematico va
   commentato (split diversi, test = partecipanti diversi?).

**Annotare:** una frase di risposta per ogni domanda. Diventano il testo della relazione.

### Passo 6 · Figure
Script `src/evaluation/final_figures.py`. Usa il seed 0 delle MLP (D4) e il seed 42 della baseline
(il primo dei suoi 10), fissati prima di disegnare.
- `figures/test_recall_k.png`: curva R@K al variare di K, da 1 a N (asse x logaritmico), pannelli
  esatto e classe: baseline casuale ("prima del training"), ridge, MLP `diag`, MLP `classe`
  ("dopo"). Scelta al posto dell'istogramma dei ranghi: a K = 1, 5, 10 la curva vale le R@K della
  tabella, e le curve non si coprono. Usa i ranghi salvati al passo 4.
- `figures/test_risultati.png`: la tabella finale come barre. Pannelli esatto e classe; per R@1, R@5,
  R@10 una barra per modello (baseline, ridge, MLP `diag`, MLP `classe`), barre di errore = dev.
  std sui seed (D4). Legge `final_test.csv`.
- `figures/test_combinazioni_nuove.png`: R@10 (esatto e classe) su tutte le query e sulle 420 con combinazione
  verbo+nome mai vista nel train (D3). Nomi mai visti (24) solo in tabella.
- `figures/test_matrice_similarita.png`: matrice frasi × video "stile CLIP" su 30 clip del test, MLP
  `diag` e `classe` (seed 0). In ogni cella la probabilita' che la frase scelga quel video tra i 30
  (softmax della riga, τ = 0.07, la τ fissa dei modelli). Mostra il trade-off della loss: `diag`
  concentra la probabilita' sulla diagonale (coppia esatta), `classe` la distribuisce nel blocco
  della classe. **Scelta delle clip fissata ora, solo dai metadati:** classi verbo+nome del test
  con almeno 5 frasi diverse; 6 classi a caso (seed 0); in ognuna 5 frasi diverse a caso e una
  clip per frase. Nessuna clip si sceglie guardando i risultati.

Sostituisce l'istogramma dei coseni (`p6_coseni.png` della prima esecuzione), poco leggibile.

**Annotare:** cosa mostra ogni figura, in una frase.

### Passo 7 · Chiusura
- Compilare la tabella di [roadmap_fase5.md §2](roadmap_fase5.md) con i numeri del test e
  sostituire la riga del ridge (oggi e' quella del subset, non confrontabile).
- Spuntare P6 nella checklist di `roadmap_fase5.md`.
- Riportare tabella e figure in `docs/REPORT.md`, con le definizioni di D2 e D3.
- Commit e PR verso `main`.

**Controllo:** ogni numero in roadmap e report si ritrova in `final_test.csv`.

---

## Errori da evitare

| Errore | Conseguenza |
|---|---|
| Normalizzare con media/dev. del test | il test influenza il modello: risultato non valido |
| Allineare video e testo per posizione | coppie sbagliate, metriche senza senso |
| Dimenticare `model.eval()` | dropout attivo: metriche peggiori e non riproducibili |
| Riportare il seed migliore sul test | scelta fatta sul test: risultato ottimista |
| Definire lo zero-shot dopo aver visto i numeri | definizione scelta a posteriori |
| Confrontare con numeri del subset | pool di candidati diverso: R@K non confrontabili |
