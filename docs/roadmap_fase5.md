# Roadmap Fase 5: allineamento contrastivo video–testo

Documento di lavoro per il progetto EK100 (traccia 15, Vision-Language Alignment with CLIP for Video).
Da seguire passo dopo passo. Ogni passo ha un obiettivo, le visualizzazioni da produrre e il criterio per passare al successivo.

---

## 0. Mappa delle fasi nel progetto

| Fase | Cosa | Stato | File |
|---|---|---|---|
| 1–2 | Estrazione feature (TimeSformer, MPNet, BERT, CLIP) e split | fatto | `src/datasets/`, `data/features/` |
| 3 | Baseline zero-shot: proiezione casuale, il "pavimento" | fatto | `src/evaluation/baseline_zero_shot.py` |
| 4 | Allineamento lineare (ridge in forma chiusa), riferimento da battere | fatto (val e test) | `src/evaluation/linear_alignment.py` |
| 5 | Modello allenato: teste MLP + loss InfoNCE | **da fare** | `src/training/` |
| 6 | Metriche riusabili, analisi e estensioni | parziale | `src/evaluation/metrics.py` |

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

| Modello | R@1 | R@5 | R@10 | rango mediano | classe R@10 | nuove combinazioni R@10 |
|---|---|---|---|---|---|---|
| Caso (K/N) | 0.04% | 0.22% | 0.44% | ~N/2 | — | — |
| Ridge lineare (MPNet, λ=100) | 3.07% | 10.45% | 16.06% | 96 | 57.38% | 28.35% |
| Contrastivo (MLP) | | | | | | |

Riferimenti già noti:
- **Baseline casuale (fase 3), test**: R@1 0.044%, R@5 0.133%, R@10 0.222%.
- **Ridge lineare (fase 4), MPNet λ=100**: val R@10 esatto 14.5%, R@10 classe 59.6%; test R@10 esatto 16.06%, R@10 classe 57.38% (`experiments/logs/linear_alignment_test.csv`).

**Criterio di successo:** supera il ridge su R@10 esatto e R@10 classe, sul val (per la scelta) e sul test (una volta sola).

Se non lo supera, il report resta valido ma il messaggio cambia: la loss contrastiva non aggiunge nulla rispetto a una proiezione lineare. Va dichiarato.

---

## 3. Dati e split

- Feature: `data/features/timesformer_k400_8f/timesformer_k400_8f.npz` (19.798 × 768); testo in `data/features/text/{mpnet,bert,clip}.npz`.
- Split (`data/splits/split.csv`): train 14.772, val 2.778, test 2.248.
- Il val è separato per sessioni (`split_info.json`), quindi non c'è leakage tra persone.
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

- `src/training/heads.py`: le teste.
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

### P4 · Ricerca iperparametri (sul val)
**Obiettivo:** scegliere la config migliore.

- Griglia di 20–30 config: `lr ∈ {1e-4, 1e-3}`, `p ∈ {0.1, 0.3}`, `τ` fisso vs apprendibile, `batch ∈ {128, 256, 512}`, `d ∈ {128, 256}`.
- Criterio: R@10 classe sul val, come in `linear_alignment`. Non usare il test.

**Visualizzazioni:**
- Heatmap `lr × τ` su R@10 val.
- Tabella config ordinate.

**Criterio per passare:** una config scelta e documentata.

### P5 · Confronto encoder di testo
**Obiettivo:** scegliere l'encoder di testo.

- Config migliore con MPNet, BERT e CLIP, tre seed ciascuno.

**Visualizzazioni:**
- Barre con deviazione standard sui seed, per encoder, con il ridge a confronto.

**Criterio per passare:** encoder scelto con variabilità nota.

### P6 · Valutazione finale sul test
**Obiettivo:** il numero del report.

- Un solo run, checkpoint scelto sul val.
- Calcola: R@1/5/10 esatto, R@1/5/10 classe, rango mediano, metriche sulle combinazioni nuove.
- Salva in `experiments/logs/final_test.csv`.

**Visualizzazioni:**
- Istogramma dei ranghi del giusto, prima e dopo il training.
- Distribuzione del coseno positivo vs negativo.

**Criterio per passare:** tabella finale del punto 2 compilata.

### P7 · Analisi qualitativa
**Obiettivo:** capire perché il modello sbaglia.

- t-SNE/UMAP dello spazio appreso (val, colorati per verbo), video e testo con marker diversi.
- Distanza tra centroidi di video e testo (*modality gap*).
- Matrice di confusione dei verbi: per ogni verbo della query, quale verbo compare più spesso nel top-1.
- Barre R@10 per verbo: i più deboli e i più forti.
- Esempi di fallimento: frase, primi 5 video recuperati, posizione del giusto.
- Seen vs unseen: barre affiancate.

**Criterio per passare:** analisi pronta per il report.

### P8 · Estensioni (opzionali)
- Loss multi-positivo.
- Fine-tuning parziale di MPNet (ultimi layer). Richiede di rieseguire l'encoder sui token.
- Fine-tuning video: richiede GPU e clip ritagliate dai video grezzi (`data/EPIC-KITCHENS`, 193 GB). Non è necessario per l'obiettivo minimo.

---

## 5. Struttura dei file

| Percorso | Contenuto |
|---|---|
| `src/training/heads.py` | teste MLP |
| `src/training/loss.py` | InfoNCE simmetrica |
| `src/training/train.py` | ciclo di training |
| `src/evaluation/analysis.py` | grafici e metriche per verbo, fallimenti, centroidi |
| `experiments/configs/*.yaml` | una config per run |
| `experiments/checkpoints/` | pesi migliori (fuori da git) |
| `experiments/logs/` | un CSV per run, più `final_test.csv` |
| `figures/` | grafici, con nome che indica config e seed |

---

## 6. Checklist rapida

- [x] P0 dati allineati e conteggi documentati
- [x] P1 ridge calcolato anche sul test
- [ ] P2 test di overfit superato
- [ ] P3 prima run completa su MPNet
- [ ] P4 config scelta sul val
- [ ] P5 encoder di testo scelto con 3 seed
- [ ] P6 valutazione finale sul test, una sola volta
- [ ] P7 analisi qualitativa e grafici
- [ ] Tabella finale nel report (`docs/REPORT.md`)
