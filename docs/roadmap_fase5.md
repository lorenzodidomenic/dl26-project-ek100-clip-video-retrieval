# Roadmap Fase 5: allineamento contrastivo video–testo

Documento di lavoro per il progetto EK100 (traccia 15, Vision-Language Alignment with CLIP for Video).
Da seguire passo dopo passo. Ogni passo ha un obiettivo, le visualizzazioni da produrre e il criterio per passare al successivo.

---

## 0. Mappa delle fasi nel progetto

| Fase | Cosa | Stato | File |
|---|---|---|---|
| 1–2 | Estrazione feature (TimeSformer, MPNet, BERT, CLIP) e split | fatto (dataset completo: 633 sessioni; testo solo MPNet e BERT) | `src/datasets/`, `data/features/` |
| 3 | Baseline zero-shot: proiezione casuale, il "pavimento" | fatto (seed separati per testo e video) | `src/evaluation/baseline_zero_shot.py` |
| 4 | Allineamento lineare (ridge in forma chiusa), riferimento da battere | fatto (val e test, dataset completo) | `src/evaluation/linear_alignment.py` |
| 5 | Modello allenato: teste MLP + loss InfoNCE | **fatto** sul dataset completo (P0–P6; pipeline v2 con Dataset/DataLoader, PyTorch Lightning e wandb). Studio pilota sul subset di 4 partecipanti archiviato in `subset4p` | `src/training/` |
| 6 | Metriche riusabili, analisi e estensioni | fatto (multi-positivo, P7 analisi qualitativa) | `src/evaluation/` |

---

## 1. Il training, spiegato

### 1.1 Input

Per ogni clip e ogni frase ci sono già vettori fissi prodotti da encoder congelati:
- `v_i`: video, 768 dimensioni (TimeSformer K400, 8 frame)
- `t_i`: testo, 768 dimensioni (MPNet, BERT) o 512 (CLIP)

Il training non modifica gli encoder. Allena solo le teste.

### 1.2 Teste di proiezione

Due MLP, uno per modalità:

```
v_i → Linear(768, H) → GELU → Dropout(p) → Linear(H, d) → L2-norm → z^v_i
t_i → Linear(D_t, H) → GELU → Dropout(p) → Linear(H, d) → L2-norm → z^t_i
```

Valori di partenza: `H = 1024`, `d = 256`, `p = 0.1`.

Servono perché le due modalità hanno dimensioni diverse (768 vs 512/768) e perché non sono allineate: lo spazio comune va imparato.

Dopo la normalizzazione L2 ogni vettore sta sulla sfera unitaria, quindi il prodotto scalare è il coseno.

### 1.3 Similarità in un batch

Con un batch di `B` coppie (es. `B = 256`):

```
S = Z^t · (Z^v)^T / τ        # forma B × B
```

- La diagonale contiene le coppie giuste.
- Il resto sono negativi "in-batch".
- `τ` è la temperatura: piccola concentra la loss sui negativi difficili. Valore iniziale 0.07, apprendibile con clamp in `[0.01, 0.5]`.

### 1.4 Loss InfoNCE simmetrica

```
L = ½ · [ CE(S, diagonale) + CE(S^T, diagonale) ]
```

- La prima termine: per ogni testo, trova il suo video tra i `B` candidati.
- Il secondo: per ogni video, trova il suo testo.
- La cross-entropy spinge la diagonale a essere il valore più alto della riga.
- In pratica la loss è una misura di retrieval dentro il batch: più il modello ordina bene, più scende.

### 1.5 Ciclo di training

1. Estrai un batch casuale di coppie dal train (mescolando a ogni epoca).
2. Forward delle due teste, calcolo di `S` e della loss.
3. Backward e passo di AdamW (`lr ≈ 1e-3`, `weight_decay ≈ 1e-2`). Aggiorna `τ` se apprendibile.
4. Fine epoca: loss di validazione e R@1/5/10 sul val, con la stessa definizione di `retrieval_metrics`.
5. Early stopping: fermati se R@10 sul val non migliora per 10 epoche. Salva il miglior checkpoint in `experiments/checkpoints/`.

**Implementazione attuale (pipeline v2).** I dati passano da `FeaturePairs` e `DataLoader`
(`src/utils/dataset.py`); teste, loss e AdamW stanno in un `LightningModule` (`task.py`), il ciclo
e' un `pl.Trainer` con `EarlyStopping` e `ModelCheckpoint` su `val/classe_R@10` (`train.py`).
Le statistiche di standardizzazione del train sono buffer del modello, quindi finiscono nel
checkpoint. Log per epoca in CSV (`callbacks.py`) e su wandb (progetto `dl26-ek100-clip-retrieval`).
Device di default: CPU.

**Cosa aspettarsi:**
- La loss di train scende.
- La loss di val scende, poi risale quando il modello memorizza.
- R@10 sul val sale e poi si stabilizza.
- Se la loss di train va a zero in pochi passi, il modello sta memorizzando: aumenta il dropout o riduci `H`.

**Trappola: falsi negativi.** Più frasi possono descrivere lo stesso verbo e nome. Un "negativo" nel batch può essere una risposta giusta. Per questo conviene confrontare InfoNCE classico e la variante multi-positivo (positivi tutti i video con stesso verbo/nome).

**Normalizzazione dell'input.** Standardizza ogni feature con media e deviazione standard calcolate sul **train**. Con MPNet e TimeSformer spesso conta più degli iperparametri.

### 1.6 PyTorch, non NumPy

Il training va scritto in PyTorch:
- **Gradienti automatici**: `loss.backward()` calcola le derivate; in NumPy le scriveresti a mano.
- **Moduli e ottimizzatori pronti**: `nn.Linear`, `nn.GELU`, `nn.Dropout`, `AdamW`.
- **Dispositivo**: stesso codice su CPU, MPS (Mac) o CUDA (cloud).

PyTorch è già nel progetto (`environment.yml`, venv, `src/models/`). Conversione al caricamento:

```python
import torch
Xv, Xt, meta = load_pair("mpnet", "train")
Xv = torch.from_numpy(Xv).float().to(device)
Xt = torch.from_numpy(Xt).float().to(device)
```

Per le metriche, riporta a NumPy con `.detach().cpu().numpy()`, così `retrieval_metrics` resta invariata. Fasi 3 e 4 restano in NumPy perché sono algebra lineare senza gradienti.

---

## 2. Obiettivo finale

Un modello che, su **test**, fa text→video meglio della baseline lineare.

**Risultati sul test** (dataset completo, 9668 clip; P6 con la pipeline v2,
`experiments/final/final_test.csv`). R@1/5/10 e rango mediano sono *esatti* (giusto = stesso
`narration_id`); "nuove combinazioni" = query con coppia verbo/nome assente dal train (420),
giusto = stessa classe. Valori in %; media ± dev. std sui seed (baseline 10, MLP 3).

| Modello | R@1 | R@5 | R@10 | rango mediano | classe R@10 | nuove combinazioni R@10 |
|---|---|---|---|---|---|---|
| Caso (K/N) | 0.01 | 0.05 | 0.10 | 4834.5 (≈N/2) | 5.14 | 0.71 |
| Baseline: proiezioni casuali non allenate | 0.01 ± 0.01 | 0.05 ± 0.02 | 0.10 ± 0.04 | 4773 ± 208 | 4.22 ± 1.53 | 0.81 ± 1.69 |
| Ridge lineare (MPNet, λ=1000 scelto sul val) | 1.07 | 3.85 | 6.33 | 491 | 50.16 | 10.48 |
| Contrastivo (MLP), loss `diag` | 2.94 ± 0.06 | 10.24 ± 0.47 | **15.95 ± 0.26** | **116 ± 3** | 70.56 ± 0.45 | 14.13 ± 0.96 |
| **Contrastivo (MLP), loss `classe` (finale)** | 2.57 ± 0.08 | 9.14 ± 0.15 | 14.27 ± 0.27 | 138 ± 6 | **73.68 ± 0.40** | **14.76 ± 0.95** |

Modello finale: MPNet, H=1024, d=128, dropout 0.3, lr 1e-4, batch 256, τ fissa, loss multi-positivo
`classe` (vedi `docs/roadmap_p6_test.md` e `experiments/final/manifest.json`). Trade-off rispetto a
`diag`: +3.1 su classe R@10 (+4.6 su classe R@1: 37.34 contro 32.77), −1.7 su esatto R@10. Sulle
combinazioni nuove le due loss sono alla pari (differenza < 1 dev. std).

Prima esecuzione di P6 (pipeline v1, senza DataLoader/Lightning, modello finale con batch 512):
classe R@10 72.89 ± 1.38, esatto R@10 13.61 ± 0.50. Archiviata in `experiments/archivio_v1/final/`;
le regole di scelta sono rimaste le stesse.

Valori del subset a 4 partecipanti (ridge test: esatto R@10 16.06%, classe R@10 57.38%, in
`experiments/logs/subset4p/`): **non confrontabili** con questi, perche' il numero di video
candidati per query e' molto piu' piccolo.

**Criterio di successo:** supera il ridge su R@10 esatto e R@10 classe, sul val (per la scelta) e sul test (una volta sola).
**Esito (test): superato.** Esatto R@10 14.27 contro 6.33 del ridge, classe R@10 73.68 contro 50.16.
Anche sul val: esatto R@10 13.80 contro 6.09, classe R@10 75.51 contro 52.89.

Se non lo supera, il report resta valido ma il messaggio cambia: la loss contrastiva non aggiunge nulla rispetto a una proiezione lineare. Va dichiarato.

---

## 3. Dati e split

- Dataset completo: 633 sessioni, 34 partecipanti. Feature video
  `data/features/timesformer_k400_8f/timesformer_k400_8f.npz` (76.885 × 768, esattamente le
  annotazioni di retrieval: 67.217 train + 9.668 test); testo in `data/features/text/{mpnet,bert}.npz`.
  I video grezzi sono stati cancellati dopo l'estrazione.
- Split (`data/splits/split.csv`, da `make_split.py`): train 56.430, val 10.787 (78 sessioni, 15%),
  test 9.668 (il test ufficiale delle annotazioni).
- Combinazioni verbo+nome assenti dal train: 399 clip nel val, 420 nel test. Verbi e nomi mai
  visti sono quasi assenti (test: 2 clip con verbo nuovo, 24 con nome nuovo).
- Il val è separato per sessioni (`split_info.json`), quindi non c'è leakage tra sessioni.
- Studio pilota sul subset di 4 partecipanti (P22, P24, P27, P30; train 14.772, val 2.778,
  test 2.248): archiviato in `data/splits/subset4p/` e nelle altre cartelle `subset4p`.
- `load_pair(encoder, split)` allinea per `narration_id`.

Regole:
- Il test si guarda una sola volta, alla fine.
- Ogni scelta (config, epoca, encoder) si fa sul val.
- Statistiche di normalizzazione solo dal train.
- Ogni numero nel report ha seed e config tracciabili.

---

## 4. Roadmap passo per passo

### P0 · Sanity dei dati
**Obiettivo:** essere sicuri che i dati siano allineati e completi.

- Conta le coppie per split; verifica che video e testo abbiano gli stessi `narration_id`.
- Conta verbi, nomi e combinazioni per split. Quante combinazioni del test non compaiono nel train? Serve per la metrica "nuove".
- Fissa il seed; registra le versioni di torch e sklearn.

**Visualizzazioni:**
- Istogrammi di verbi e nomi per split.
- Barra "combinazioni nuove nel test".

**Criterio per passare:** coppie allineate e conteggi documentati.

### P1 · Riferimenti
**Obiettivo:** avere i numeri di confronto sul test.

- Riesegui `linear_alignment` anche sul test (oggi è valutato solo sul val).
- Verifica che `baseline_zero_shot.csv` contenga i valori attesi (già fatto).

**Visualizzazioni:**
- t-SNE delle feature grezze, video e testo in grafici separati, colorati per verbo (val, sottocampione stratificato). È la fotografia del "prima".

**Criterio per passare:** tabella di riferimento completa per val e test.

### P2 · Modello, loss e test di correttezza
**Obiettivo:** codice corretto prima di qualsiasi training vero.

- `src/models/heads.py`: le teste.
- `src/training/loss.py`: InfoNCE simmetrica.
- Funzione di metriche che riusa `retrieval_metrics`.
- **Test di overfit** su 256 coppie di train: la loss deve andare quasi a zero e R@1 sul sottoinsieme ~100%. Se no, c'è un bug (quasi sempre allineamento o diagonale).

**Visualizzazioni:**
- Curva di loss del sottoinsieme.
- Matrice `S` 256×256 prima e dopo: la diagonale deve accendersi.

**Criterio per passare:** overfit riuscito.

### P3 · Primo training completo (MPNet)
**Obiettivo:** una run di riferimento funzionante.

- Config: `H=1024, d=256, p=0.1, lr=1e-3, τ=0.07 apprendibile, batch=256`.
- Log per epoca in `experiments/logs/train_<nome>.csv`: loss train, loss val, R@1/5/10 val, `τ`.

**Visualizzazioni:**
- Curve di loss (train/val) ed R@10 val per epoca.
- Andamento di `τ`.

**Criterio per passare:** il val migliora rispetto al ridge, o si capisce il motivo se non succede.

**Esito:** run `mpnet_h1024_d256_p0.1_lr0.001_b256_taulearn_s0`, epoca migliore 4 di 14. Sul val: classe R@10 68.5 (ridge 59.6), esatto R@10 20.5 (ridge 14.5). Overfitting già dall'epoca 2: la loss di train scende, la loss val sale e τ scende da 0.07 a 0.04.

### P4 · Ricerca iperparametri (sul val)
**Obiettivo:** scegliere la config migliore.

- **Griglia completa, 48 config** (seed 0): `lr ∈ {1e-4, 1e-3}` × `p ∈ {0.1, 0.3}` × `τ` fissa/apprendibile × `batch ∈ {128, 256, 512}` × `d ∈ {128, 256}`.
  Completa e non a stadi: le leve interagiscono (batch × τ, batch × lr), e con 24 run per valore gli effetti principali sono medie robuste.
- Fissi: MPNet, `H=1024`, weight decay 1e-2, max 200 epoche, patience 10.
- La run di P3 è una delle 48 celle e viene riusata: le run nuove sono 47.
- **Device CPU**: con un modello così piccolo MPS guadagna al massimo circa il 25%, e su MPS lo stesso seed non garantisce lo stesso risultato. Costo misurato: circa 2.5–3.5 s per epoca, quindi circa 1–2 h per la griglia.
- **Seed sulle prime 5**: dopo la griglia, le prime 5 config si rilanciano con seed 1 e 2 (10 run). Si sceglie sulla media dei 3 seed, perché il massimo di 48 misure rumorose (±3 punti tra epoche) è ottimista.
- **Regola di scelta**, fissata prima di vedere i risultati: vince la media più alta; se due config distano meno di una deviazione standard (la maggiore delle due), vince la più semplice (d più piccolo, poi dropout più alto, poi la media). Applicata in automatico da `grid_results.py` quando tutti i seed sono finiti.
- Criterio: R@10 classe sul val, come in `linear_alignment`. Non usare il test.
- Restano fuori da P4 la loss multi-positivo e il rumore sull'input: diventano ablation sulla config scelta (vedi P8).

**Implementazione** (un passo alla volta, ognuno validato prima del successivo):
1. `train.py`: estrarre il nome della run in una funzione `run_name(args)`, condivisa con lo script della griglia.
2. `src/training/grid.py`: genera le 48 config e lancia `train.py` in un processo separato per ciascuna. Una run conta come fatta se il suo log mostra un training finito (early stopping o ultima epoca) e il checkpoint esiste; si può interrompere e rilanciare. Opzioni `--dry-run`, `--seeds` e `--only`.
3. Esecuzione: `caffeinate -i python src/training/grid.py`.
4. `src/evaluation/grid_results.py`: run e iperparametri presi dalla definizione della griglia in `grid.py` (solo le run finite), epoca migliore dal log con la regola di `train.py`; tabella e grafici. Si può lanciare anche con la griglia in corso.
5. Seed 1 e 2 sulle prime 5, tabella con media ± deviazione standard.
6. Chiusura: config scelta documentata qui e nel report.

**Output:**
- `experiments/logs/p4_grid.csv`: 48 run ordinate per criterio.
- `experiments/logs/p4_seed.csv`: prime 5, media ± deviazione standard su 3 seed.

**Visualizzazioni:**
- Effetti principali: per ogni iperparametro, media ± deviazione standard di R@10 classe per valore, con il ridge come riferimento (`figures/p4_effetti_principali.png`).
- Heatmap delle interazioni `batch × τ` e `lr × dropout` (`figures/p4_heatmap_*.png`).
- Tabella config ordinate.

**Criterio per passare:** una config scelta con la regola sopra e documentata, con la sua variabilità sui seed.

**Esito (subset di 4 partecipanti, val):**
- **Config scelta: `d=128, dropout=0.3, lr=1e-4, batch=256, τ fissa`** (con `H=1024`, weight decay 1e-2). Sui 3 seed: classe R@10 **69.31 ± 1.42**, classe R@1 32.75 ± 1.15, esatto R@10 22.43 ± 0.72, nuove classe R@10 24.70 ± 0.92; epoca migliore 44.7 ± 14.4.
- Le prime 5 sono tutte alla pari (distanza dalla migliore < 1 dev. std, valori tra 68.08 e 69.31). La scelta ha sia la media più alta sia la forma più semplice (d più piccolo, dropout più alto), quindi la regola non è ambigua. Ha anche il risultato più stabile sulle combinazioni nuove (± 0.92 contro 1.9–6.0 delle altre).
- Effetto dei seed: la prima al seed 0 valeva 69.98, la media sui 3 seed è 69.31 (i seed vanno da 67.7 a 70.3). La classifica al seed 0 è rumore entro circa ±1.5 punti, come previsto.
- Effetti principali (48 run): d 256 > 128 (+1.0), τ fissa > appresa (+0.8), lr 1e-4 > 1e-3 (+0.7); dropout (+0.2) e batch (≤ 0.2) quasi ininfluenti. Tutte le 48 run battono il ridge (59.6): il minimo è 65.5.
- Nota: sulla singola leva d=256 è meglio, ma la config scelta ha d=128. Gli effetti principali sono medie su tutte le altre leve, mentre la scelta riguarda una combinazione precisa; con il rumore osservato non c'è contraddizione.
- Il guadagno rispetto alla config di partenza di P3 (68.5, un solo seed) è circa +0.8: con questi dati gli iperparametri contano poco rispetto al salto dal ridge (circa +9).
- File: `experiments/logs/p4_grid.csv`, `p4_seed.csv`; figure `p4_effetti_principali.png`, `p4_heatmap_*.png`, `p4_seed.png`. Archiviati in `subset4p` (vedi sezione 5).

**Esito (dataset completo, val, pipeline v2).** Griglia e seed rifatti da zero con le stesse
regole (dettagli in `experiments/final/manifest.json`):
- Griglia 48 config con loss `diag`, seed 1 e 2 sulle prime 5: vince di nuovo
  **`d=128, dropout=0.3, lr=1e-4, batch=256, τ fissa`**, classe R@10 72.17 ± 0.56.
- Ablation della loss sulla config scelta (anticipata da P8), 3 seed: `classe` 75.51 ± 0.62,
  `diag` 72.17 ± 0.56, `frase` 72.07 ± 0.59 (equivalente a `diag`). Vince `classe`.
- Mini-griglia con loss `classe`, batch {128, 256, 512} × τ {fissa, appresa}: b256 τ fissa ha la
  media più alta (75.51 ± 0.62); b256 τ appresa, b512 τ appresa e b512 τ fissa sono alla pari, b128
  no. Resta b256 τ fissa, anche la più semplice tra quelle alla pari.
- Figure aggiornate sul dataset completo: `figures/p4_*.png`.

### P5 · Confronto encoder di testo
**Obiettivo:** scegliere l'encoder di testo.

- Config migliore con MPNet, BERT e CLIP, tre seed ciascuno.

**Visualizzazioni:**
- Barre con deviazione standard sui seed, per encoder, con il ridge a confronto.

**Criterio per passare:** encoder scelto con variabilità nota.

**Esito (fatto dopo P6, quindi solo informativo).** Sul dataset completo è stato scelto MPNet
senza griglia sugli encoder (CLIP non estratto). Il confronto con BERT è stato fatto dopo la
valutazione sul test, a parità di config (quella scelta per MPNet, senza una griglia per BERT),
con le loss `diag` e `classe` × 3 seed; verifica sul val prima del test
(`experiments/final/check_val_bert.csv`). Il modello finale resta MPNet e il confronto va
dichiarato come informativo.

| Test, MLP `classe` | esatto R@10 | rango mediano | classe R@10 | nuove classe R@10 |
|---|---|---|---|---|
| MPNet | **14.27 ± 0.27** | **138 ± 6** | **73.68 ± 0.40** | **14.76 ± 0.95** |
| BERT | 13.03 ± 0.09 | 153 ± 5 | 70.27 ± 0.61 | 9.76 ± 1.91 |

MPNet è migliore su tutte le metriche e per tutti i modelli allenati (anche ridge e MLP `diag`).
Lo scarto più grande è sulle combinazioni nuove (+5 punti). File: `src/evaluation/encoder_compare.py`,
`experiments/final/final_test_bert.csv`, `experiments/logs/encoder_compare.csv`.

### P6 · Valutazione finale sul test
**Obiettivo:** il numero del report.

- Un solo run, checkpoint scelto sul val.
- Calcola: R@1/5/10 esatto, R@1/5/10 classe, rango mediano, metriche sulle combinazioni nuove.
- Salva in `experiments/logs/final_test.csv`.

**Visualizzazioni:**
- Istogramma dei ranghi del giusto, prima e dopo il training.
- Distribuzione del coseno positivo vs negativo.

**Criterio per passare:** tabella finale del punto 2 compilata.

**Esito: fatto** (piano completo e decisioni pre-registrate in `docs/roadmap_p6_test.md`). Script
`src/evaluation/evaluate_test.py` (verifica sha256 dei checkpoint, prima `--split val`, poi
`--split test` una volta) e `src/evaluation/final_figures.py`. Output in `experiments/final/`
(`final_test.csv`, `final_test_ranks.npz`, anche su wandb come run `test-finale`). Figure
`p6_recall_k.png`, `p6_risultati.png`, `p6_zero_shot.png`, `p6_matrice_similarita.png`, spiegate in
`figures/p6_figure_spiegate.md`. Eseguito due volte: pipeline v1 (archiviata in
`experiments/archivio_v1/final/`) e pipeline v2 (quella in tabella al punto 2).

### P7 · Analisi qualitativa
**Obiettivo:** capire perché il modello sbaglia.

- t-SNE/UMAP dello spazio appreso (val, colorati per verbo), video e testo con marker diversi.
- Distanza tra centroidi di video e testo (*modality gap*).
- Matrice di confusione dei verbi: per ogni verbo della query, quale verbo compare più spesso nel top-1.
- Barre R@10 per verbo: i più deboli e i più forti.
- Esempi di fallimento: frase, primi 5 video recuperati, posizione del giusto.
- Seen vs unseen: barre affiancate.

**Criterio per passare:** analisi pronta per il report.

**Esito: fatto** (sul test, modelli seed 0 come fissato in D4; figure solo descrittive, nessuna
scelta ne segue).

**t-SNE dello spazio appreso** (`src/evaluation/plot_tsne.py`, 2000 clip a caso, video e frasi in un
solo fit, metrica coseno), con note di lettura nei `.md` accanto alle figure:
- `figures/p7_tsne_spazio_appreso_test.png` (MLP `diag` e `classe`): nessun modality gap visibile,
  video e frasi mescolati; wash ha una regione sua, take e put sono sparsi; lo spazio è organizzato
  in piccoli gruppi per verbo+nome; a scala di verbo le due loss si somigliano.
- `figures/p7_tsne_zero_shot_test.png` (MLP `classe` più le 420 clip con combinazione nuova): le
  clip zero-shot hanno quasi tutte verbi poco frequenti (insert, mix, scoop), si concentrano in una
  parte della mappa e spesso frasi e video stanno in gruppi separati, coerente con il 15.7% di
  classe R@10 (seed 0).
- `figures/p7_tsne_perplexity_test.png`: perplexity 15, 30, 50, controllo di robustezza.

**Analisi per verbo, per frequenza ed errori** (`src/evaluation/error_analysis.py`). Ridge contro MLP
`classe`; in F3 anche MLP `diag`. Scelte (verbi, gruppi di frequenza) fissate solo dai metadati.
- **F1** `figures/p7_recall_per_verbo.png`: classe R@10 per i 12 verbi più frequenti del test.
  L'MLP batte il ridge su tutti e 12. I più deboli sono insert (41.9) e remove (39.9), gesti di
  spostamento dove conta l'oggetto; i più forti hanno un contesto visivo riconoscibile: open
  (91.4), turn-on (89.4), close (87.2), wash (86.4). Media su tutte le query: 74.0.
- **F2** `figures/p7_recall_per_frequenza.png`: classe R@10 per numero di clip della classe
  verbo+nome nel train. MLP: mai vista (420 query) 15.7, 1–10 (1165) 33.6, 11–100 (3033) 62.9,
  oltre 100 (5050) 94.8; ridge 10.5 / 14.1 / 31.8 / 72.8. Il risultato dipende molto dalla
  frequenza; il guadagno sul ridge è massimo sulle classi viste da 1 a 100 volte e minimo sulle
  combinazioni mai viste (+5). Sostituisce le barre "seen vs unseen".
- **F3** `figures/p7_tipi_di_errore.png`: il primo video recuperato rispetto alla frase. Classe
  giusta: ridge 16.7%, `diag` 31.4%, `classe` 38.4% (= classe R@1). Verbo e nome entrambi
  sbagliati: ridge 50.4%, `classe` 25.5%. Quando l'MLP sbaglia, sbaglia più spesso il verbo
  (22.6%) che il nome (13.5%): gli oggetti si riconoscono meglio delle azioni.
- **F4** `figures/p7_coppie_confuse.png`: le 10 confusioni più frequenti del modello finale
  (classe della frase → classe del top-1). 7 su 10 sono azioni opposte sullo stesso oggetto
  (turn-off tap → turn-on tap, 143 query su 154; take plate → put plate; put spoon → take spoon),
  le altre oggetti simili (take fork → take spoon, take oil → take bottle). Risponde all'obiettivo
  extra della traccia sulle coppie verbo/nome confuse. Si riferisce al solo top-1: per turn-off
  la classe giusta resta spesso tra i primi 10 (R@10 86.7 in F1).

Controllo: "classe giusta" al top-1 coincide query per query con il rango classe 1 di
`final_test_ranks.npz` (0 differenze per i tre modelli).

Non fatti, per scelta: matrice di confusione completa dei verbi (78 verbi, poco leggibile:
sostituita da F3 e F4), distanza tra centroidi (il t-SNE non mostra modality gap), esempi
qualitativi di retrieval (provati e scartati).

### Zero-shot (obiettivo minimo 4 della consegna)
**Richiesta:** *"Explicitly define and test zero-shot capabilities acting on entirely unseen
verbs/nouns"*: interrogare il modello con verbi o nomi che non ha mai visto in training.

**Definizione** (D3 di `docs/roadmap_p6_test.md`, fissata prima del test). "Mai visto" = assente
dal **train**, l'unico split da cui le teste hanno imparato; verbo = `verb_class`, nome =
`noun_class`. Tre sottoinsiemi di query del test, valutati come tutte le altre (candidati = tutti
i 9668 video, giusto = stessa classe verbo+nome):

| Sottoinsieme | Query | Cosa misura |
|---|---|---|
| combinazione verbo+nome mai vista | 420 | verbo e nome visti, mai insieme: composizione |
| nome mai visto | 24 (4 nomi) | zero-shot vero sui nomi |
| verbo mai visto | 2 (1 frase) | zero-shot vero sui verbi |

**Risultati sul test** (`experiments/final/final_test.csv`; MLP media ± dev. std su 3 seed):

| Modello | combinazione nuova: classe R@10 | rango mediano | nome nuovo: classe R@10 | rango mediano |
|---|---|---|---|---|
| Caso | 0.71 | 2743 | 0.00 | 1072 |
| Baseline casuale | 0.81 ± 1.69 | 3522 ± 422 | 0.00 | 3482 ± 1707 |
| Ridge (λ=1000) | 10.48 | 475 | 0.00 | **66.5** |
| MLP `diag` | 14.13 ± 0.96 | **157 ± 20** | 0.00 | 702 ± 636 |
| MLP `classe` (finale) | **14.76 ± 0.95** | 192 ± 27 | 4.17 ± 4.17 | 343 ± 278 |

**Lettura:**
- **Combinazioni nuove:** la sola misura con abbastanza query. Gli MLP battono il ridge (14.8
  contro 10.5 di classe R@10, rango mediano circa 2.5 volte migliore), ma restano molto sotto le
  query di classi viste (73.7): il modello compone poco. Vedi anche F2 di P7 e `p6_zero_shot.png`.
- **Nomi mai visti:** R@10 quasi nullo per tutti, ma sul rango mediano il **ridge è migliore**
  degli MLP (66.5 contro 343 e 702). Un'ipotesi coerente: il ridge, lineare e molto regolarizzato,
  conserva di più la geometria di MPNet, mentre gli MLP si specializzano sulle classi del train.
  Va presa con cautela: sono 24 clip di soli 4 nomi (quorn 14, asparagus 6, computer 2,
  door:kitchen 2) e la dev. std sui seed è enorme. Due casi sono quasi sinonimi di classi viste:
  "move mouse" (classe computer) e "open/close kitchen door" (classe door:kitchen, distinta da door).
- **Verbo mai visto:** 2 clip con la stessa frase, "scoop potatoes onto baking tray", annotata con
  il verbo *bake*. La parola "scoop" è nel train: è un caso di etichetta, non una parola nuova.
  Nessun modello la trova nei primi 10 (rango classe migliore: MLP `diag`, 235).

**Limite da dichiarare:** il test contiene troppe poche parole del tutto nuove per una misura
affidabile dello zero-shot vero sulle classi. Per questo si aggiunge lo zero-shot sulle parole,
qui sotto.

#### Zero-shot sulle parole: sinonimi mai visti
**Domanda:** se nella frase cercata una parola viene sostituita con un sinonimo che le teste non
hanno mai visto (es. "wash pan" → "wash skillet"), il modello trova ancora i video giusti? MPNet,
congelato e preaddestrato, conosce il sinonimo; le teste MLP e il ridge no. Nessun training: le
feature video del test sono già estratte, si ricodificano solo le frasi.

**Scelte fissate prima di vedere i risultati** (`src/evaluation/zero_shot_sinonimi.py`):
- 9 nomi e 4 verbi, con sinonimo assente da tutte le frasi di train e val (1615 parole; lo script
  lo controlla e si ferma se non è vero): pan → skillet, plate → platter, tap → faucet,
  bag → sack, glass → tumbler, potato → spud, bin → trashcan, counter → worktop,
  hob → stovetop; take → grasp, wash → cleanse, pour → decant, move → relocate. Scartati i
  sinonimi già nel train (place, grab, rinse, cabinet, mug, stir…), quelli che contengono la
  parola originale (teakettle, stockpot), antiquati (icebox) o sgrammaticati (deposit).
- Si sostituisce la parola intera (pan sì, pans no); per i verbi solo se non è seguita da una
  particella ("take knife" → "grasp knife", "take out knife" escluso). Una parola cambiata per
  frase: 1886 frasi con un nome sostituito, 1671 con un verbo.
- Stesse frasi, originale contro riscritta, codificate ora con MPNet; metriche come D2
  (candidati = tutti i 9668 video del test, giusto = stessa classe verbo+nome); ridge, MLP
  `diag` e `classe` con 3 seed.

**Controlli:** frasi originali ricodificate uguali a `mpnet.npz` (differenza massima 2e-7);
ranghi di classe delle frasi originali identici a P6 per tutti i 6 MLP (ridge: 1 su 3557,
pareggio invertito dall'arrotondamento).

**Risultati** (classe R@10; MLP media ± dev. std su 3 seed; `experiments/final/zero_shot_sinonimi.csv`,
`figures/zero_shot_sinonimi.png`):

| Modello | nomi: originale | nomi: sinonimo | verbi: originale | verbi: sinonimo |
|---|---|---|---|---|
| Ridge (λ=1000) | 52.0 | 49.1 | 50.9 | 31.6 |
| MLP `diag` | 77.0 ± 1.1 | 56.1 ± 3.6 | 70.4 ± 1.8 | 59.9 ± 4.1 |
| **MLP `classe` (finale)** | **79.9 ± 1.8** | **72.0 ± 3.2** | **73.5 ± 1.0** | **66.2 ± 3.1** |

**Lettura:**
- **Il modello finale regge le parole nuove:** perde 7–8 punti, ma con il sinonimo resta molto
  sopra il ridge con la parola originale (72 contro 52). Sfrutta ciò che MPNet sa delle parole,
  non solo il vocabolario del train.
- **La loss `classe` generalizza meglio di `diag`:** sui nomi `diag` perde 21 punti, `classe` 8.
  Ipotesi: `classe` avvicina frasi diverse della stessa classe ("take pan", "pick up pan"), quindi
  impara il significato più che la formulazione esatta.
- **Il ridge crolla sui verbi** (−19), molto meno sui nomi (−3).
- **Per parola** (MLP finale): quasi invariate tap → faucet (95.5 → 91.6), pan → skillet
  (73.7 → 73.0), move → relocate (33.3 → 33.1); in forte calo glass → tumbler (90.6 → 57.5) e
  potato → spud (75.8 → 54.3), parole più rare o informali (tumbler è anche ambiguo).

**Limite:** qui è nuova la parola, non l'oggetto: il video mostra sempre una classe vista in
training. È uno zero-shot lessicale, complementare a quello sui nomi mai visti (sopra), dove su
24 query il ridge aveva il rango mediano migliore: misurano cose diverse.

#### Demo con clip esterne
**Obiettivo:** far vedere, su video che il modello non ha mai visto (né nel train né nel test di
EPIC), che una frase scritta da noi trova la clip giusta. È una **dimostrazione per le slide, non
una misura**: poche clip, da un dominio diverso (video stock, non GoPro in testa).

**Dati (fatto).** 6 clip di Pexels (licenza Pexels), tagliate sull'azione, lato corto 480 pixel,
senza audio, in `data/demo/clips/` (fuori da git: i video non si ridistribuiscono). Frasi, tagli,
file originali e link in `experiments/demo/clips.csv` (su git):

| Clip | Frase | Nota |
|---|---|---|
| `put_mince.mp4` | put minced meat in pan | |
| `crack_egg.mp4` | crack egg into pan | |
| `pour_cream.mp4` | pour cream into pan | |
| `open_bag_cheese.mp4` | open bag of cheese | |
| `pour_milk.mp4` | pour milk into bowl | |
| `cut_watermelon.mp4` | cut watermelon | *watermelon* assente dal train: video e parola nuovi |

Scartate le anteprime iStock (a pagamento, con filigrana) e le clip non in prima persona o con
più azioni montate.

**Scelte fissate prima di vedere i risultati:**
- modello: MLP `classe` seed 0 (D4), lo stesso delle figure di P6 e P7;
- frasi: quelle del CSV, scritte prima di lanciare il modello. Se dopo si cambia una frase o una
  clip, va dichiarato;
- video trattati come EPIC: 8 frame a distanza uniforme tra primo e ultimo
  (`uniform_indices` di `cut_clips.py`), preprocessing del processor TimeSformer, token CLS.

**Passi:**
1. **Script `src/evaluation/demo_clip_esterne.py`.** Per ogni clip: 8 frame → TimeSformer
   (768) → testa video del modello finale; per ogni frase: MPNet (768) → testa testo. La
   standardizzazione usa media e dev. std del train salvate nel checkpoint, come in P6. Ne esce
   la matrice di coseno frasi × clip (6 × 6).
2. **Figura `figures/demo_clip_esterne.png`.** In alto una miniatura per clip, a sinistra le
   frasi; in ogni cella la probabilità che la frase scelga quella clip (softmax della riga,
   τ = 0.07, come `p6_matrice_similarita.png`); diagonale bordata = coppia giusta. Nel titolo:
   quante frasi su 6 trovano la loro clip al primo posto (caso: 1 su 6).
3. **Opzione `--query "frase"`.** Stampa la classifica delle 6 clip per una frase libera, per
   provare dal vivo all'esame anche parole mai viste (es. "slice watermelon", "add parmesan").
4. *(Facoltativo)* dashboard Gradio con la stessa logica: solo se avanza tempo.
5. **Chiusura:** esito e limiti qui e nelle slide.

Passi 1–3 fatti. Le feature delle clip sono salvate in `data/demo/clip_features.npz` (fuori da
git) e si ricalcolano solo se le clip cambiano. Spiegazione della figura e della formula in
`figures/demo_clip_esterne.md`.

**Esito: 4 frasi su 6** trovano la loro clip al primo posto (a caso: 1 su 6).

| Frase | Primo posto | Probabilità della clip giusta |
|---|---|---|
| open bag of cheese | ✔ sua clip | 98% |
| pour milk into bowl | ✔ sua clip | 64% |
| put minced meat in pan | ✔ sua clip | 60% |
| cut watermelon (parola mai vista) | ✔ sua clip | 56% |
| crack egg into pan | ✘ `pour_cream` (55%) | 30% (seconda) |
| pour cream into pan | ✘ `put_mince` (42%) | 18% (terza) |

**Lettura:**
- Lo zero-shot completo funziona: clip e parola *watermelon* sono entrambe nuove.
- I due errori sono tra clip "in padella" (uova, panna, carne): a scena simile il modello
  confonde, coerente con F3 e F4 di P7. La clip con lo sfondo più diverso (sacchetto) è la più
  netta.
- Frasi libere (`--query`): «slice watermelon» → `cut_watermelon` (45%); «add parmesan to rice»
  → `open_bag_cheese` (87%); «break an egg» → `crack_egg`, ma di poco (36% contro 33%).

**Limiti da dichiarare:** 6 clip non sono una misura (un errore vale 17 punti); 4 su 6 sono "in
padella", quindi il modello deve distinguere l'oggetto nella stessa scena; video stock diversi da
EPIC (stabilizzati, inquadratura diversa). Le percentuali dipendono da quante clip ci sono:
con più clip si abbassano.

### P8 · Estensioni (opzionali)
- Loss multi-positivo: **fatta**, ed entrata nella scelta del modello finale (vedi esito di P4 sul
  dataset completo). Opzione `positives` di `train.py`: `diag` (InfoNCE classica), `frase` (positivi
  = stessa frase), `classe` (positivi = stesso verbo e nome). Motivazione: nella run di P3, nel 59%
  delle righe di un batch da 256 c'era un falso negativo (stesso verbo e nome) e nel 38% la stessa
  frase identica.
- Rumore o dropout sulle feature di input, come augmentation (ablation sulla config di P4).
- Fine-tuning parziale di MPNet (ultimi layer). Richiede di rieseguire l'encoder sui token.
- Fine-tuning video: richiede GPU e clip ritagliate dai video grezzi, che sono stati cancellati dopo l'estrazione delle feature (andrebbero riscaricati). Non è necessario per l'obiettivo minimo.

---

## 5. Struttura dei file

| Percorso | Contenuto |
|---|---|
| `src/utils/dataset.py` | `FeaturePairs` (Dataset) e DataLoader delle coppie video–testo |
| `src/models/heads.py` | teste MLP |
| `src/training/loss.py` | InfoNCE simmetrica, anche multi-positivo (`diag`, `frase`, `classe`) |
| `src/training/task.py` | `RetrievalTask`, il `LightningModule` (teste + loss + AdamW, metriche val a fine epoca) |
| `src/training/callbacks.py` | storico delle metriche per epoca e CSV di log |
| `src/training/train.py` | `pl.Trainer` con early stopping, checkpoint e logger wandb |
| `src/training/grid.py` | lancio della griglia di P4 e dei seed |
| `src/evaluation/grid_results.py` | tabella, effetti principali, heatmap e seed di P4, con la scelta della config |
| `src/evaluation/evaluate_test.py` | valutazione finale (P6), `--split val` per la verifica, `--encoder` per BERT |
| `src/evaluation/final_figures.py` | figure di P6 |
| `src/evaluation/encoder_compare.py` | confronto MPNet–BERT sul test (P5) |
| `src/evaluation/plot_tsne.py` | t-SNE dello spazio appreso (P7) |
| `experiments/final/` | checkpoint scelti, `manifest.json` (scelta e sha256), risultati di P6 (su git, tranne i checkpoint) |
| `experiments/archivio_v1/` | checkpoint, log e risultati del test della pipeline v1 |
| `*/subset4p/`, `figures/figures4p/` | archivio dello studio pilota sul subset di 4 partecipanti (split, feature testo, log, checkpoint, figure) |
| `src/evaluation/error_analysis.py` | R@10 per verbo e per frequenza, tipi di errore e coppie confuse del top-1 (P7) |
| `src/evaluation/zero_shot_sinonimi.py` | zero-shot sulle parole: frasi del test con sinonimi mai visti in training |
| `src/evaluation/demo_clip_esterne.py` | demo su clip esterne (Pexels): figura frasi × clip e `--query` per frasi libere |
| `experiments/demo/clips.csv` | clip della demo: frase, taglio, file originale, link e licenza (i video stanno in `data/demo/clips/`) |
| `experiments/configs/*.yaml` | non usato: la config di ogni run sta nel suo nome, nel checkpoint e su wandb |
| `experiments/checkpoints/` | pesi migliori (fuori da git) |
| `experiments/logs/` | un CSV per run, più i CSV di riepilogo (P4, ridge, baseline, encoder) |
| `figures/` | grafici, con nome che indica fase, config e seed; note di lettura nei `.md` |

---

## 6. Checklist rapida

- [x] P0 dati allineati e conteggi documentati
- [x] P1 ridge calcolato anche sul test
- [x] P2 test di overfit superato
- [x] P3 prima run completa su MPNet
- [x] P4 config scelta sul val (subset, poi rifatta sul dataset completo con ablation della loss e mini-griglia)
- [x] P5 confronto encoder con 3 seed (MPNet contro BERT, fatto dopo P6: solo informativo)
- [x] P6 valutazione finale sul test, una sola volta per pipeline (`docs/roadmap_p6_test.md`)
- [x] P7 analisi qualitativa e grafici (t-SNE, R@10 per verbo e per frequenza, tipi di errore, coppie confuse)
- [x] Zero-shot: definizione e risultati sui sottoinsiemi del test (D3)
- [x] Zero-shot: frasi con parole mai viste (sinonimi)
- [x] Zero-shot: demo con clip esterne (6 clip Pexels, 4 su 6 al primo posto)
- [x] P8 loss multi-positivo
- [ ] Tabella finale e figure nella relazione
