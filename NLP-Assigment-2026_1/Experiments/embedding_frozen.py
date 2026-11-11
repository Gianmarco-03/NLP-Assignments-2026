"""
Experiment: Baseline (Bi-LSTM + Dense) with a frozen or trainable GloVe Embedding layer,
with or without oversampling of the A clauses.

The settings are compared on validation macro F1 (Task 5). Everything else (cleaning, vocabulary,
embedding matrix, pos_weight, hyper-parameters) replicates the Deliverable.

Run from the project root:
    NLP/Scripts/python Experiments/embedding_frozen.py [--trainable] [--a-factor N]
Results are written to Experiments/results/baseline_<frozen|trainable>_a<N>.json.
"""
import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import gensim.downloader as gloader
import nltk
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------- configuration

parser = argparse.ArgumentParser()
parser.add_argument('--trainable', action='store_true', help='train the Embedding layer instead of freezing it')
parser.add_argument('--a-factor', type=int, default=4,
                    help='each A clause appears this many times in the train set (1 = no oversampling)')
parser.add_argument('--pooling', choices=['final', 'max'], default='final',
                    help='sentence encoding: final forward and backward states, or element-wise max of all states')
args = parser.parse_args()

FREEZE_EMBEDDING = not args.trainable
SEEDS = [42, 1337, 2026]

MAX_LEN = 150           # truncation length, about p99 of the train clause lengths
A_FACTOR = args.a_factor
HIDDEN_SIZE = 128
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3    # Adam
THRESHOLD = 0.5         # per-category tuning is done in the Deliverable

ROOT = Path(__file__).resolve().parent.parent
DATA_FOLDER = ROOT / 'Data' / 'ToS-100'
RESULTS_FILE = (Path(__file__).resolve().parent / 'results'
                / f"baseline_{'frozen' if FREEZE_EMBEDDING else 'trainable'}_a{A_FACTOR}"
                  f"{'_max' if args.pooling == 'max' else ''}.json")

LABEL_COLS = ['A', 'CH', 'CR', 'LTD', 'TER']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------- data loading

keep = ['document_ID', 'text'] + LABEL_COLS
training_set = pd.read_csv(DATA_FOLDER / 'train.csv')[keep]
val_set = pd.read_csv(DATA_FOLDER / 'validation.csv')[keep]
test_set = pd.read_csv(DATA_FOLDER / 'test.csv')[keep]

# ---------------------------------------------------------------- Task 2a - cleaning (same as Deliverable)

nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

ENUMERATOR = r"(?:\d+|[a-z]|[ivx]+)"
NUMBERING_RE = re.compile(
    rf"^(?:-l[rs]b- {ENUMERATOR} -r[rs]b-"   # (a)  (ii)  (1)  [a]
    rf"|{ENUMERATOR} -r[rs]b-"               # a)  ii)  1)
    rf"|\d+(?:\.\d+)+\.?"                    # 1.1  2.3.1
    rf"|\d+ [.°]"                            # 1 .  3 °
    rf"|[•·●*-])\s+"                         # bullets
)
TOKEN_MAP = {
    '-lrb-': '(', '-rrb-': ')', '-lsb-': '[', '-rsb-': ']', '-lcb-': '{', '-rcb-': '}',
    '``': '"', "''": '"', '`': "'",
}
CLITICS = {"'s", "'re", "'ve", "'ll", "'d", "'m", "n't"}
SPECIAL_CHARS_RE = re.compile(r"[^a-z0-9\s.,;:!?()\[\]{}\"'/$%&@-]")

lemmatizer = WordNetLemmatizer()


def get_wordnet_key(tag: str) -> str | None:
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    return None


def strip_numbering(text: str) -> str:
    while (stripped := NUMBERING_RE.sub('', text, count=1)) != text:
        text = stripped
    return text


def lemmatize(tokens: list[str]) -> list[str]:
    lemmas = []
    for i, (word, tag) in enumerate(pos_tag(tokens)):
        key = get_wordnet_key(tag)
        before_negation = i + 1 < len(tokens) and tokens[i + 1] == "n't"
        lemmas.append(lemmatizer.lemmatize(word, key) if key and not before_negation else word)
    return lemmas


def reattach_clitics(tokens: list[str]) -> list[str]:
    merged = []
    for token in tokens:
        if token in CLITICS and merged:
            merged[-1] += token
        else:
            merged.append(token)
    return merged


def clean_text(text: str) -> str:
    text = strip_numbering(text)
    tokens = [TOKEN_MAP.get(token, token) for token in text.split()]
    tokens = reattach_clitics(lemmatize(tokens))
    text = SPECIAL_CHARS_RE.sub(' ', ' '.join(tokens))
    return ' '.join(text.split())


for df in (training_set, val_set, test_set):
    df['clean_text'] = df['text'].apply(clean_text)

# ---------------------------------------------------------------- Task 3 - vocabulary and embedding matrix (same as Deliverable)

glove = gloader.load('glove-wiki-gigaword-100')

PAD, UNK = '<PAD>', '<UNK>'
PAD_IDX, UNK_IDX = 0, 1

train_counts = Counter(token for text in training_set['clean_text'] for token in text.split())
train_oov = [token for token in train_counts if token not in glove.key_to_index]

idx_to_word = [PAD, UNK] + glove.index_to_key + train_oov
word_to_idx = {word: idx for idx, word in enumerate(idx_to_word)}


def compose_vector(token: str) -> np.ndarray | None:
    clitic = next((c for c in CLITICS if token.endswith(c) and token != c), None)
    stem = token[:-len(clitic)] if clitic else token
    parts = re.split(r'[-/]', stem) + ([clitic] if clitic else [])
    vectors = [glove[part] for part in parts if part in glove.key_to_index]
    return np.mean(vectors, axis=0) if vectors else None


rng = np.random.default_rng(42)
embedding_dim = glove.vector_size

embedding_matrix = np.zeros((len(idx_to_word), embedding_dim), dtype=np.float32)
embedding_matrix[UNK_IDX] = rng.uniform(-0.05, 0.05, embedding_dim)
embedding_matrix[2:2 + len(glove.index_to_key)] = glove.vectors
for token in train_oov:
    vector = compose_vector(token)
    embedding_matrix[word_to_idx[token]] = vector if vector is not None else rng.uniform(-0.05, 0.05, embedding_dim)


def encode(text: str) -> list[int]:
    return [word_to_idx.get(token, UNK_IDX) for token in text.split()]


for df in (training_set, val_set, test_set):
    df['token_ids'] = df['clean_text'].apply(encode)

# ---------------------------------------------------------------- Task 2b - imbalance strategy

# oversample A clauses only (A never co-occurs with other labels), train set only
a_clauses = training_set[training_set['A'] == 1]
train_resampled = pd.concat([training_set] + [a_clauses] * (A_FACTOR - 1), ignore_index=True)

y_train = train_resampled[LABEL_COLS].to_numpy(dtype=np.float32)
y_val = val_set[LABEL_COLS].to_numpy(dtype=np.float32)
y_test = test_set[LABEL_COLS].to_numpy(dtype=np.float32)

# pos_weight = N_neg / N_pos per category, on the train set actually used for training
n_pos = y_train.sum(axis=0)
pos_weight = torch.tensor((len(y_train) - n_pos) / n_pos, dtype=torch.float32)

# ---------------------------------------------------------------- dataset and padding


class ClauseDataset(Dataset):
    def __init__(self, token_ids: pd.Series, labels: np.ndarray):
        # empty clauses become a single <UNK> so that every sequence has length >= 1
        self.sequences = [torch.tensor((ids or [UNK_IDX])[:MAX_LEN]) for ids in token_ids]
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def collate(batch):
    """Dynamic padding: each batch is padded with <PAD> to its longest clause."""
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded = pad_sequence(sequences, batch_first=True, padding_value=PAD_IDX)
    return padded, lengths, torch.stack(labels)


# ---------------------------------------------------------------- Task 4 - Baseline model


class BaselineModel(nn.Module):
    """Embedding -> Bi-LSTM -> concatenation of the final forward and backward states -> Dense."""

    def __init__(self, embedding_matrix: np.ndarray, hidden_size: int, n_labels: int, freeze: bool):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.from_numpy(embedding_matrix), freeze=freeze, padding_idx=PAD_IDX
        )
        self.lstm = nn.LSTM(embedding_matrix.shape[1], hidden_size, batch_first=True, bidirectional=True)
        self.dense = nn.Linear(2 * hidden_size, n_labels)

    def forward(self, token_ids, lengths):
        embedded = self.embedding(token_ids)
        # packing makes the LSTM skip <PAD>, so the final states belong to the real last tokens
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        outputs, (hidden, _) = self.lstm(packed)
        if args.pooling == 'max':
            # <PAD> positions are excluded from the max
            states, _ = pad_packed_sequence(outputs, batch_first=True)
            mask = torch.arange(states.size(1), device=states.device)[None, :] >= lengths.to(states.device)[:, None]
            encoding = states.masked_fill(mask.unsqueeze(-1), float('-inf')).max(dim=1).values
        else:
            encoding = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.dense(encoding)   # logits, sigmoid is inside BCEWithLogitsLoss


# ---------------------------------------------------------------- Task 5 - training and evaluation


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_proba(model, loader) -> np.ndarray:
    model.eval()
    probs = []
    for token_ids, lengths, _ in loader:
        logits = model(token_ids.to(DEVICE), lengths)
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def compute_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict:
    y_pred = (probs >= THRESHOLD).astype(int)
    metrics = {
        'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
    }
    per_category = f1_score(y_true, y_pred, average=None, zero_division=0)
    metrics.update({f'f1_{label}': score for label, score in zip(LABEL_COLS, per_category)})
    return {name: float(value) for name, value in metrics.items()}


def run(seed: int) -> dict:
    set_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(ClauseDataset(train_resampled['token_ids'], y_train), batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate, generator=generator)
    val_loader = DataLoader(ClauseDataset(val_set['token_ids'], y_val), batch_size=256, collate_fn=collate)
    test_loader = DataLoader(ClauseDataset(test_set['token_ids'], y_test), batch_size=256, collate_fn=collate)

    model = BaselineModel(embedding_matrix, HIDDEN_SIZE, len(LABEL_COLS), FREEZE_EMBEDDING).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=LEARNING_RATE)

    best = {'macro_f1': -1.0}
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for token_ids, lengths, labels in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(token_ids.to(DEVICE), lengths), labels.to(DEVICE))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(labels)
        epoch_loss /= len(train_loader.dataset)

        val_metrics = compute_metrics(y_val, predict_proba(model, val_loader))
        history.append({'epoch': epoch, 'train_loss': epoch_loss, 'val_macro_f1': val_metrics['macro_f1']})
        print(f"  seed {seed} | epoch {epoch:2d} | loss {epoch_loss:.4f} | val macro F1 {val_metrics['macro_f1']:.4f}")

        # best epoch chosen on validation macro F1
        if val_metrics['macro_f1'] > best['macro_f1']:
            best = {'epoch': epoch, **val_metrics,
                    'test': compute_metrics(y_test, predict_proba(model, test_loader))}

    return {'seed': seed, 'best_epoch': best.pop('epoch'), 'test': best.pop('test'), 'validation': best,
            'history': history}


def summarise(runs: list[dict], split: str) -> pd.DataFrame:
    table = pd.DataFrame([r[split] for r in runs])
    return table.agg(['mean', 'std']).T


if __name__ == '__main__':
    truncated = sum(len(ids) > MAX_LEN for ids in training_set['token_ids'])
    print(f"Device: {DEVICE} | freeze embedding: {FREEZE_EMBEDDING} | vocabulary: {len(idx_to_word)}")
    print(f"Train clauses after A oversampling: {len(train_resampled)} | truncated at {MAX_LEN} tokens: {truncated}")
    print(f"pos_weight: {dict(zip(LABEL_COLS, pos_weight.round(decimals=1).tolist()))}")

    start = time.time()
    runs = []
    for seed in SEEDS:
        print(f"\nSeed {seed}")
        runs.append(run(seed))

    print(f"\nValidation (best epoch per seed, threshold {THRESHOLD}):")
    print(summarise(runs, 'validation').round(4))
    print("\nTest (same epochs):")
    print(summarise(runs, 'test').round(4))
    print(f"\nTotal time: {time.time() - start:.0f} s")

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    config = {'freeze_embedding': FREEZE_EMBEDDING, 'seeds': SEEDS, 'max_len': MAX_LEN, 'a_factor': A_FACTOR,
              'hidden_size': HIDDEN_SIZE, 'batch_size': BATCH_SIZE, 'epochs': EPOCHS,
              'learning_rate': LEARNING_RATE, 'threshold': THRESHOLD}
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump({'config': config, 'runs': runs}, f, indent=2)
    print(f"Results saved to {RESULTS_FILE}")
