"""
Bonus experiments on the Baseline or the Stacked model (Task 5 setup), outside the Deliverable.

Two bonus suggestions of the assignment are covered:
- comparison of imbalance strategies with the same architecture: pos_weight on or off, and three resampling
  options (none, A clauses x4, every positive clause x3); threshold tuning is evaluated on every run;
- GloVe swapped for Law2Vec legal word embeddings, with the OOV rate of both.

Everything else (cleaning, OOV handling, BiLSTMClassifier with max pooling, trainable Embedding, Adam, seeds,
20 epochs, best epoch on validation macro F1 at threshold 0.5, thresholds tuned on validation) replicates
Deliverable.ipynb. Law2Vec was trained with every digit replaced by 'D', so in that setting the cleaned text of all
splits gets the same replacement before the vocabulary is built.

Run from the project root:
    NLP/Scripts/python Experiments/bonus_lstm.py [--no-pos-weight] [--resample none|a4|pos3]
                                                 [--embeddings glove|law2vec] [--pw-power P] [--layers 1|2]
Runs are written to Experiments/results/bonus_[stacked_]<embeddings>_<resample>_<pw|pw<P>|nopw>.pkl, with the same
structure as task5_runs.pkl ({'Baseline' or 'Stacked': [run per seed]}).
"""
import argparse
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
from gensim.models import KeyedVectors
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--pos-weight', action=argparse.BooleanOptionalAction, default=True,
                    help='per-category pos_weight in BCEWithLogitsLoss')
parser.add_argument('--resample', choices=['none', 'a4', 'pos3'], default='a4',
                    help='none, A clauses x4 (Deliverable), or every positive clause x3')
parser.add_argument('--embeddings', choices=['glove', 'law2vec'], default='glove')
parser.add_argument('--pw-power', type=float, default=1.0,
                    help='softened pos_weight = (N_neg / N_pos) ** power; 0.5 is the square root of N_neg / N_pos')
parser.add_argument('--layers', type=int, choices=[1, 2], default=1, help='1 = Baseline, 2 = Stacked')
args = parser.parse_args()
MODEL_NAME = 'Baseline' if args.layers == 1 else 'Stacked'

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'Data' / 'ToS-100'
LAW2VEC_FILE = ROOT / 'Experiments' / 'law2vec' / 'Law2Vec.100d.txt'
PW_TAG = ('pw' if args.pw_power == 1.0 else f'pw{args.pw_power:g}') if args.pos_weight else 'nopw'
TAG = f"{'' if args.layers == 1 else 'stacked_'}{args.embeddings}_{args.resample}_{PW_TAG}"
RUNS_FILE = ROOT / 'Experiments' / 'results' / f'bonus_{TAG}.pkl'

SEEDS = [42, 1337, 2026]
MAX_LEN = 150
HIDDEN_SIZE = 128
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
THRESHOLD_GRID = np.arange(0.05, 0.96, 0.05)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

label_cols = ['A', 'CH', 'CR', 'LTD', 'TER']
keep = ['document_ID', 'text'] + label_cols
training_set = pd.read_csv(DATA / 'train.csv')[keep]
val_set = pd.read_csv(DATA / 'validation.csv')[keep]
test_set = pd.read_csv(DATA / 'test.csv')[keep]
y_val = val_set[label_cols].to_numpy(dtype=np.float32)
y_test = test_set[label_cols].to_numpy(dtype=np.float32)

# ---------------------------------------------------------------- Task 2a, same as the Deliverable

nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

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
    if args.embeddings == 'law2vec':
        df['clean_text'] = df['clean_text'].str.replace(r'\d', 'D', regex=True)

# ---------------------------------------------------------------- Task 3 with the chosen word embeddings

if args.embeddings == 'glove':
    vectors = gloader.load('glove-wiki-gigaword-100')
else:
    vectors = KeyedVectors.load_word2vec_format(str(LAW2VEC_FILE), binary=False)

PAD, UNK = '<PAD>', '<UNK>'
PAD_IDX, UNK_IDX = 0, 1

train_counts = Counter(token for text in training_set['clean_text'] for token in text.split())
train_oov = [token for token in train_counts if token not in vectors.key_to_index]
idx_to_word = [PAD, UNK] + vectors.index_to_key + train_oov
word_to_idx = {word: idx for idx, word in enumerate(idx_to_word)}


def compose_vector(token: str) -> np.ndarray | None:
    clitic = next((c for c in CLITICS if token.endswith(c) and token != c), None)
    stem = token[:-len(clitic)] if clitic else token
    parts = re.split(r'[-/]', stem) + ([clitic] if clitic else [])
    found = [vectors[part] for part in parts if part in vectors.key_to_index]
    return np.mean(found, axis=0) if found else None


rng = np.random.default_rng(42)
embedding_dim = vectors.vector_size
embedding_matrix = np.zeros((len(idx_to_word), embedding_dim), dtype=np.float32)
embedding_matrix[UNK_IDX] = rng.uniform(-0.05, 0.05, embedding_dim)
embedding_matrix[2:2 + len(vectors.index_to_key)] = vectors.vectors
n_composed = 0
for token in train_oov:
    vector = compose_vector(token)
    if vector is None:
        vector = rng.uniform(-0.05, 0.05, embedding_dim)
    else:
        n_composed += 1
    embedding_matrix[word_to_idx[token]] = vector


def oov_report() -> pd.DataFrame:
    report = {}
    for name, df in (('train', training_set), ('validation', val_set), ('test', test_set)):
        counts = Counter(token for text in df['clean_text'] for token in text.split())
        # train: tokens missing from the pre-trained embeddings; validation/test: tokens mapped to <UNK>
        known = vectors.key_to_index if name == 'train' else word_to_idx
        oov = [token for token in counts if token not in known]
        report[name] = {
            'token types': len(counts),
            'OOV types': len(oov),
            'OOV types (%)': round(100 * len(oov) / len(counts), 2),
            'OOV occurrences (%)': round(100 * sum(counts[t] for t in oov) / counts.total(), 2),
        }
    return pd.DataFrame.from_dict(report, orient='index')


def encode(text: str) -> list[int]:
    return [word_to_idx.get(token, UNK_IDX) for token in text.split()]


for df in (training_set, val_set, test_set):
    df['token_ids'] = df['clean_text'].apply(encode)

# ---------------------------------------------------------------- Task 2b strategy under test

if args.resample == 'a4':
    extra = [training_set[training_set['A'] == 1]] * 3
elif args.resample == 'pos3':
    extra = [training_set[training_set[label_cols].any(axis=1)]] * 2
else:
    extra = []
train_resampled = pd.concat([training_set] + extra, ignore_index=True)
y_resampled = train_resampled[label_cols].to_numpy(dtype=np.float32)
n_pos = y_resampled.sum(axis=0)
pos_weight = torch.tensor(((len(y_resampled) - n_pos) / n_pos) ** args.pw_power) if args.pos_weight else None

# ---------------------------------------------------------------- Tasks 4-5, same as the Deliverable


class ClauseDataset(Dataset):
    def __init__(self, token_ids: pd.Series, labels: np.ndarray):
        self.sequences = [torch.tensor((ids or [UNK_IDX])[:MAX_LEN]) for ids in token_ids]
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def collate(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded = pad_sequence(sequences, batch_first=True, padding_value=PAD_IDX)
    return padded, lengths, torch.stack(labels)


class BiLSTMClassifier(nn.Module):
    def __init__(self, embedding_matrix: np.ndarray, hidden_size: int, num_layers: int, n_labels: int = len(label_cols),
                 freeze_embedding: bool = False):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix), freeze=freeze_embedding, padding_idx=PAD_IDX
        )
        self.lstm = nn.LSTM(embedding_matrix.shape[1], hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=True)
        self.dense = nn.Linear(2 * hidden_size, n_labels)

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(self.embedding(token_ids), lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        outputs, _ = self.lstm(packed)
        states, _ = pad_packed_sequence(outputs, batch_first=True)
        padding = torch.arange(states.size(1), device=states.device) >= lengths.to(states.device).unsqueeze(1)
        return self.dense(states.masked_fill(padding.unsqueeze(-1), float('-inf')).max(dim=1).values)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, thresholds: float | np.ndarray = 0.5) -> dict:
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


train_dataset = ClauseDataset(train_resampled['token_ids'], y_resampled)
val_loader = DataLoader(ClauseDataset(val_set['token_ids'], y_val), batch_size=256, collate_fn=collate)
test_loader = DataLoader(ClauseDataset(test_set['token_ids'], y_test), batch_size=256, collate_fn=collate)


@torch.no_grad()
def predict_proba(model: nn.Module, loader: DataLoader) -> np.ndarray:
    model.eval()
    return np.concatenate([torch.sigmoid(model(ids.to(DEVICE), lengths)).cpu().numpy() for ids, lengths, _ in loader])


def train_model(seed: int) -> dict:
    set_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate,
                              generator=torch.Generator().manual_seed(seed))
    model = BiLSTMClassifier(embedding_matrix, HIDDEN_SIZE, args.layers).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=None if pos_weight is None else pos_weight.to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_f1, best_epoch, best_state = -1.0, 0, None
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for ids, lengths, labels in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(ids.to(DEVICE), lengths), labels.to(DEVICE))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(labels)
        val_f1 = compute_metrics(y_val, predict_proba(model, val_loader))['macro F1']
        history.append({'epoch': epoch, 'train_loss': epoch_loss / len(train_dataset), 'val_macro_f1': val_f1})
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    val_probs = predict_proba(model, val_loader)
    return {'seed': seed, 'best_epoch': best_epoch, 'history': history, 'val_probs': val_probs,
            'test_probs': predict_proba(model, test_loader), 'thresholds': tune_thresholds(y_val, val_probs)}


if __name__ == '__main__':
    print(f"Setting {TAG} | device {DEVICE}")
    print(f"Vocabulary {len(idx_to_word):,} | train OOV {len(train_oov)} -> composed {n_composed}, "
          f"random {len(train_oov) - n_composed}")
    print(oov_report().to_string())
    print(f"Train clauses {len(training_set)} -> {len(train_resampled)} | pos_weight "
          f"{None if pos_weight is None else [round(w, 1) for w in pos_weight.tolist()]}", flush=True)

    runs = []
    for seed in SEEDS:
        start = time.time()
        runs.append(train_model(seed))
        print(f"seed {seed} | best epoch {runs[-1]['best_epoch']:2d} | "
              f"val macro F1 {runs[-1]['history'][runs[-1]['best_epoch'] - 1]['val_macro_f1']:.4f} | "
              f"{time.time() - start:.0f} s", flush=True)
    RUNS_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle({MODEL_NAME: runs}, RUNS_FILE)

    for split, y_true, key in (('validation', y_val, 'val_probs'), ('test', y_test, 'test_probs')):
        for name in ('0.5', 'tuned'):
            scores = [compute_metrics(y_true, run[key], 0.5 if name == '0.5' else run['thresholds'])['macro F1']
                      for run in runs]
            print(f"{split:>10} | threshold {name:>5} | macro F1 {np.mean(scores):.3f} ± {np.std(scores, ddof=1):.3f}")
