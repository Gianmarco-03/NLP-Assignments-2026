"""
Experiment: LEGAL-BERT fine-tuned for more epochs than the Deliverable (Task 6), or another checkpoint (--model).

With 4 epochs the best validation epoch is 4, 4 and 3, and the validation macro F1 is still rising for two
seeds. This script checks whether a longer run helps. Everything else (text preprocessing, oversampling,
pos_weight, Trainer arguments, seeds, metrics) replicates the Deliverable. The linear learning-rate schedule
spans all the epochs, so epoch 4 of a longer run is not the same as epoch 4 of the 4-epoch run.

Run from the project root:
    NLP/Scripts/python Experiments/legalbert_epochs.py [--epochs N] [--model CHECKPOINT]
Results are written to Experiments/results/task6_runs_[<model>_]<N>epochs.pkl, with the same structure as task6_runs.pkl.
"""
import argparse
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=8)
parser.add_argument('--model', default='nlpaueb/legal-bert-base-uncased',
                    help='HuggingFace checkpoint; bert-base-uncased gives the general-domain comparison (bonus)')
args_cli = parser.parse_args()

ROOT = Path.cwd()
DATA = ROOT / 'Data' / 'ToS-100'
BERT_NAME = args_cli.model
MODEL_LABEL = 'LEGAL-BERT' if BERT_NAME == 'nlpaueb/legal-bert-base-uncased' else BERT_NAME.split('/')[-1]
MODEL_TAG = '' if MODEL_LABEL == 'LEGAL-BERT' else f'{MODEL_LABEL}_'
RUNS_FILE = ROOT / 'Experiments' / 'results' / f'task6_runs_{MODEL_TAG}{args_cli.epochs}epochs.pkl'
BERT_DIR = ROOT / 'Experiments' / 'legalbert' / f'{MODEL_TAG}{args_cli.epochs}epochs'

SEEDS = [42, 1337, 2026]
A_FACTOR = 4
BERT_MAX_LEN = 256
THRESHOLD_GRID = np.arange(0.05, 0.96, 0.05)

# ---------------------------------------------------------------- data (Task 1)

label_cols = ['A', 'CH', 'CR', 'LTD', 'TER']
keep = ['document_ID', 'text'] + label_cols
training_set = pd.read_csv(DATA / 'train.csv')[keep]
val_set = pd.read_csv(DATA / 'validation.csv')[keep]
test_set = pd.read_csv(DATA / 'test.csv')[keep]
y_val = val_set[label_cols].to_numpy(dtype=np.float32)
y_test = test_set[label_cols].to_numpy(dtype=np.float32)

# ---------------------------------------------------------------- text (Task 2a steps kept for BERT)

ENUMERATOR = r"(?:\d+|[a-z]|[ivx]+)"
NUMBERING_RE = re.compile(
    rf"^(?:-l[rs]b- {ENUMERATOR} -r[rs]b-"
    rf"|{ENUMERATOR} -r[rs]b-"
    rf"|\d+(?:\.\d+)+\.?"
    rf"|\d+ [.°]"
    rf"|[•·●*-])\s+"
)
TOKEN_MAP = {
    '-lrb-': '(', '-rrb-': ')', '-lsb-': '[', '-rsb-': ']', '-lcb-': '{', '-rcb-': '}',
    '``': '"', "''": '"', '`': "'",
}
CLITICS = {"'s", "'re", "'ve", "'ll", "'d", "'m", "n't"}


def strip_numbering(text: str) -> str:
    while (stripped := NUMBERING_RE.sub('', text, count=1)) != text:
        text = stripped
    return text


def reattach_clitics(tokens: list[str]) -> list[str]:
    merged = []
    for token in tokens:
        if token in CLITICS and merged:
            merged[-1] += token
        else:
            merged.append(token)
    return merged


def bert_text(text: str) -> str:
    tokens = [TOKEN_MAP.get(token, token) for token in strip_numbering(text).split()]
    return ' '.join(reattach_clitics(tokens))


for df in (training_set, val_set, test_set):
    df['bert_text'] = df['text'].apply(bert_text)

# ---------------------------------------------------------------- metrics (Task 5)


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, thresholds=0.5) -> dict:
    y_pred = (probs >= thresholds).astype(int)
    per_category = f1_score(y_true, y_pred, average=None, zero_division=0)
    return {
        'macro F1': per_category.mean(),
        'macro precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'macro recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        **{f'F1 {label}': score for label, score in zip(label_cols, per_category)},
    }


def tune_thresholds(y_true: np.ndarray, probs: np.ndarray) -> np.ndarray:
    thresholds = []
    for c in range(y_true.shape[1]):
        scores = [f1_score(y_true[:, c], probs[:, c] >= t, zero_division=0) for t in THRESHOLD_GRID]
        thresholds.append(THRESHOLD_GRID[np.argmax(scores)])
    return np.array(thresholds)

# ---------------------------------------------------------------- model and datasets (Task 6)


tokenizer = AutoTokenizer.from_pretrained(BERT_NAME)


def load_bert():
    return AutoModelForSequenceClassification.from_pretrained(
        BERT_NAME, num_labels=len(label_cols), problem_type='multi_label_classification',
        id2label=dict(enumerate(label_cols)), label2id={label: i for i, label in enumerate(label_cols)},
    )


class BertClauseDataset(Dataset):
    def __init__(self, texts: pd.Series, labels: np.ndarray):
        self.encodings = tokenizer(list(texts), truncation=True, max_length=BERT_MAX_LEN)
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {**{key: values[index] for key, values in self.encodings.items()}, 'labels': self.labels[index]}


def bert_collate(batch):
    labels = torch.stack([item.pop('labels') for item in batch])
    return {**tokenizer.pad(batch, return_tensors='pt'), 'labels': labels}


bert_train_resampled = pd.concat([training_set] + [training_set[training_set['A'] == 1]] * (A_FACTOR - 1),
                                 ignore_index=True)
y_bert_train = bert_train_resampled[label_cols].to_numpy(dtype=np.float32)

bert_train = BertClauseDataset(bert_train_resampled['bert_text'], y_bert_train)
bert_val = BertClauseDataset(val_set['bert_text'], y_val)
bert_test = BertClauseDataset(test_set['bert_text'], y_test)

n_pos = y_bert_train.sum(axis=0)
bert_pos_weight = torch.tensor((len(y_bert_train) - n_pos) / n_pos)


def weighted_bce(outputs, labels, num_items_in_batch=None):
    logits = outputs.logits
    return nn.functional.binary_cross_entropy_with_logits(logits, labels.to(logits.dtype),
                                                          pos_weight=bert_pos_weight.to(logits.device))


def trainer_metrics(eval_pred) -> dict:
    probs = torch.sigmoid(torch.tensor(eval_pred.predictions)).numpy()
    return {'macro_f1': compute_metrics(eval_pred.label_ids, probs)['macro F1']}


def train_bert(seed: int) -> dict:
    args = TrainingArguments(
        output_dir=str(BERT_DIR / f'seed_{seed}'), seed=seed, num_train_epochs=args_cli.epochs,
        learning_rate=2e-5, warmup_steps=0.1, weight_decay=0.01,
        per_device_train_batch_size=16, per_device_eval_batch_size=64,
        fp16=True,
        eval_strategy='epoch', save_strategy='epoch', logging_strategy='epoch',
        load_best_model_at_end=True, metric_for_best_model='macro_f1',
        save_total_limit=1, save_only_model=True, report_to='none', disable_tqdm=True,
    )
    trainer = Trainer(model_init=load_bert, args=args, train_dataset=bert_train, eval_dataset=bert_val,
                      data_collator=bert_collate, compute_loss_func=weighted_bce, compute_metrics=trainer_metrics)
    trainer.train()

    def predict(dataset):
        return torch.sigmoid(torch.tensor(trainer.predict(dataset).predictions)).numpy()

    val_probs = predict(bert_val)
    history = [entry for entry in trainer.state.log_history if 'eval_macro_f1' in entry]
    return {
        'seed': seed,
        'best_epoch': int(round(max(history, key=lambda e: e['eval_macro_f1'])['epoch'])),
        'history': history,
        'val_probs': val_probs,
        'test_probs': predict(bert_test),
        'thresholds': tune_thresholds(y_val, val_probs),
    }


bert_runs = []
for seed in SEEDS:
    start = time.time()
    run = train_bert(seed)
    bert_runs.append(run)
    curve = ' '.join(f"{entry['eval_macro_f1']:.3f}" for entry in run['history'])
    print(f"{MODEL_LABEL} | {args_cli.epochs} epochs | seed {seed} | best epoch {run['best_epoch']} | "
          f"val macro F1 per epoch {curve} | {time.time() - start:.0f} s", flush=True)
    pd.to_pickle({MODEL_LABEL: bert_runs}, RUNS_FILE)
shutil.rmtree(BERT_DIR, ignore_errors=True)

for split, y_true, key in (('validation', y_val, 'val_probs'), ('test', y_test, 'test_probs')):
    for name, thresholds in (('0.5', None), ('tuned', 'thresholds')):
        scores = [compute_metrics(y_true, run[key], 0.5 if thresholds is None else run[thresholds])['macro F1']
                  for run in bert_runs]
        print(f"{split:>10} | threshold {name:>5} | macro F1 {np.mean(scores):.3f} ± {np.std(scores, ddof=1):.3f}")
