# Assignment 1: Unfair Clause Detection

Multi-label detection of potentially unfair clauses in online Terms of Service, on the ToS-100 corpus. Each clause can carry any of five unfairness categories, or none:

| Label | Category |
|---|---|
| `A` | Arbitration |
| `CH` | Unilateral change |
| `CR` | Content removal |
| `LTD` | Limitation of liability |
| `TER` | Unilateral termination |

The corpus is heavily imbalanced: 92.3% of the clauses have no label, and `A` has only 75 positive training clauses. The main metric is the macro F1 over the five categories.

## Approach

- **Preprocessing:** removal of clause numbering, restoration of brackets, quotes and clitics, POS-aware lemmatization.
- **Encoding:** GloVe (`glove-wiki-gigaword-100`). Training tokens missing from GloVe are added to the vocabulary, composed from their parts when possible; validation and test OOV tokens map to a static `<UNK>`.
- **Models:**
  - Baseline: Embedding, Bidirectional LSTM, Dense;
  - Stacked: same with a second Bidirectional LSTM layer;
  - LEGAL-BERT (`nlpaueb/legal-bert-base-uncased`), fine-tuned with the Hugging Face `Trainer`.
  
  The LSTM models encode a clause as the element-wise maximum of the hidden states over its tokens, and output one sigmoid per category.
- **Class imbalance:** `A` clauses oversampled ×4 in the training set, per-category `pos_weight` in `BCEWithLogitsLoss`, per-category decision thresholds tuned on the validation set only.
- **Evaluation:** 3 seeds (42, 1337, 2026), mean ± standard deviation, best epoch and model chosen on the validation macro F1.

## Results

Macro F1, mean ± standard deviation over 3 seeds:

| Model | Validation, threshold 0.5 | Validation, tuned thresholds | Test, threshold 0.5 | Test, tuned thresholds |
|---|---:|---:|---:|---:|
| Baseline | 0.546 ± 0.006 | 0.593 ± 0.007 | 0.618 ± 0.002 | 0.649 ± 0.014 |
| Stacked | 0.547 ± 0.017 | 0.574 ± 0.016 | 0.604 ± 0.029 | 0.612 ± 0.025 |
| LEGAL-BERT | 0.646 ± 0.012 | 0.664 ± 0.008 | 0.698 ± 0.017 | 0.704 ± 0.001 |

Per-category scores, learning curves, error analysis and the additional experiments are in the notebook.

## Repository structure

```
NLP-Assigment-2026_1/
├── Deliverable.ipynb          # main notebook: Tasks 1-7 and additional experiments
├── requirements.txt
└── Experiments/
    ├── embedding_frozen.py    # frozen vs trainable embeddings, A oversampling, sentence encoding
    ├── bonus_lstm.py          # imbalance strategies and Law2Vec embeddings (LSTM models)
    ├── legalbert_epochs.py    # LEGAL-BERT with more epochs, bert-base-uncased
    ├── summarise_bonus.py     # summary tables of the additional experiments
    ├── notebooks/             # single-factor checks on Task 5 (epochs, MAX_LEN, cascade, ...)
    └── results/               # saved runs (.pkl), metrics (.json) and training logs
```

Model weights are not included (each LSTM checkpoint is about 160 MB).

## How to run

1. Create the environment (Python 3.13):

   ```bash
   python -m venv NLP
   NLP\Scripts\activate          # Windows; on Linux/macOS: source NLP/bin/activate
   pip install -r requirements.txt
   ```

2. The dataset is not stored here. Set `DOWNLOAD_DATA = True` in the first cell of `Deliverable.ipynb` on the first run: it downloads ToS-100 from [Zenodo](https://doi.org/10.5281/zenodo.22691656) into `Data/ToS-100/`.
3. Run `Deliverable.ipynb` from this folder. With `TRAIN_MODELS = False` and `TRAIN_BERT = False` (the default) the notebook loads the saved runs from `Experiments/results/` instead of training. Set them to `True` to retrain; a CUDA GPU is recommended.
4. The model graph in Task 4 is drawn with `torchview`, which needs the [Graphviz](https://graphviz.org/download/) executables on the `PATH`.
5. Scripts in `Experiments/` are run from this folder, for example `python Experiments/bonus_lstm.py --help`. The notebooks in `Experiments/notebooks/` find the data and results relative to this folder.

## References

- Lippi et al., 2019. *CLAUDETTE: an Automated Detector of Potentially Unfair Clauses in Online Terms of Service*. Artificial Intelligence and Law.
- Ruggeri et al., 2022. *Detecting and Explaining Unfairness in Consumer Contracts through Memory Networks*. Artificial Intelligence and Law.
- Pennington et al., 2014. *GloVe: Global Vectors for Word Representation*. EMNLP.
- Chalkidis et al., 2020. *LEGAL-BERT: The Muppets straight out of Law School*. Findings of EMNLP.
