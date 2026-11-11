"""
Summary table of the bonus experiments, built from the run files in Experiments/results/.

Every run file holds {model name: [run per seed]} with validation and test probabilities and the thresholds tuned on
validation. Scores are mean ± sample standard deviation over the seeds.

Run from the project root:
    NLP/Scripts/python Experiments/summarise_bonus.py
The tables are printed in Markdown.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'Experiments' / 'results'
LABELS = ['A', 'CH', 'CR', 'LTD', 'TER']
Y = {split: pd.read_csv(ROOT / 'Data' / 'ToS-100' / f'{split}.csv')[LABELS].to_numpy()
     for split in ('validation', 'test')}

STRATEGIES = [
    ('none', 'nopw', 'nessuna mitigazione'),
    ('a4', 'nopw', '`A` ×4'),
    ('pos3', 'nopw', 'positive ×3'),
    ('none', 'pw0.5', '`pos_weight` radice'),
    ('a4', 'pw0.5', '`A` ×4 + `pos_weight` radice'),
    ('none', 'pw', '`pos_weight`'),
    ('a4', 'pw', '`A` ×4 + `pos_weight` (Deliverable)'),
    ('pos3', 'pw', 'positive ×3 + `pos_weight`'),
]


def scores(runs: list[dict], split: str, tuned: bool) -> dict:
    key = 'val_probs' if split == 'validation' else 'test_probs'
    out = {'f1': [], 'p': [], 'r': [], 'f1_A': []}
    for run in runs:
        pred = (run[key] >= (run['thresholds'] if tuned else 0.5)).astype(int)
        per_cat = f1_score(Y[split], pred, average=None, zero_division=0)
        out['f1'].append(per_cat.mean())
        out['f1_A'].append(per_cat[0])
        out['p'].append(precision_score(Y[split], pred, average='macro', zero_division=0))
        out['r'].append(recall_score(Y[split], pred, average='macro', zero_division=0))
    return out


def fmt(values: list[float], std: bool = True) -> str:
    return f"{np.mean(values):.3f} ± {np.std(values, ddof=1):.3f}" if std else f"{np.mean(values):.3f}"


def row(name: str, runs: list[dict]) -> str:
    v05, vt = scores(runs, 'validation', False), scores(runs, 'validation', True)
    t05, tt = scores(runs, 'test', False), scores(runs, 'test', True)
    return (f"| {name} | {fmt(v05['f1'])} | **{fmt(vt['f1'])}** | {fmt(vt['p'], False)} / {fmt(vt['r'], False)} | "
            f"{fmt(vt['f1_A'], False)} | {fmt(t05['f1'])} | {fmt(tt['f1'])} |")


HEADER = ("| Configurazione | Val, soglia 0.5 | Val, soglie tarate | Val P / R tarate | Val F1 `A` tarata | "
          "Test, soglia 0.5 | Test, soglie tarate |\n|---|---:|---:|---:|---:|---:|---:|")


def load(path: Path) -> list[dict] | None:
    return next(iter(pd.read_pickle(path).values())) if path.exists() else None


for model, prefix in (('Baseline', ''), ('Stacked', 'stacked_')):
    print(f"\n### {model}: strategie contro lo sbilanciamento (GloVe)\n\n{HEADER}")
    for resample, pw, name in STRATEGIES:
        runs = load(RESULTS / f'bonus_{prefix}glove_{resample}_{pw}.pkl')
        print(row(name, runs) if runs else f"| {name} | in corso | | | | | |")

print(f"\n### Embedding: GloVe contro Law2Vec (`A` ×4 + `pos_weight`)\n\n{HEADER}")
for model, prefix in (('Baseline', ''), ('Stacked', 'stacked_')):
    for emb in ('glove', 'law2vec'):
        runs = load(RESULTS / f'bonus_{prefix}{emb}_a4_pw.pkl')
        name = f"{model}, {'GloVe' if emb == 'glove' else 'Law2Vec'}"
        print(row(name, runs) if runs else f"| {name} | in corso | | | | | |")

GRID = np.arange(0.05, 0.96, 0.05)


def macro_f1(y: np.ndarray, probs: np.ndarray, thresholds) -> float:
    return f1_score(y, (probs >= thresholds).astype(int), average='macro', zero_division=0)


def oracle_thresholds(y: np.ndarray, probs: np.ndarray) -> np.ndarray:
    return np.array([GRID[np.argmax([f1_score(y[:, c], probs[:, c] >= t, zero_division=0) for t in GRID])]
                     for c in range(y.shape[1])])


print("\n### Trasferimento delle soglie da validation a test (diagnosi, il test non serve a scegliere)\n\n"
      "| Modello | Configurazione | Guadagno taratura, val | Guadagno taratura, test | Test, soglie da val | "
      "Test, soglie ottime sul test | Distanza | Soglie medie (A, CH, CR, LTD, TER) |\n"
      "|---|---|---:|---:|---:|---:|---:|---|")
for model, prefix in (('Baseline', ''), ('Stacked', 'stacked_')):
    for resample, pw, name in STRATEGIES:
        runs = load(RESULTS / f'bonus_{prefix}glove_{resample}_{pw}.pkl')
        if not runs:
            continue
        yv, yt = Y['validation'], Y['test']
        gain_v = np.mean([macro_f1(yv, r['val_probs'], r['thresholds']) - macro_f1(yv, r['val_probs'], 0.5) for r in runs])
        gain_t = np.mean([macro_f1(yt, r['test_probs'], r['thresholds']) - macro_f1(yt, r['test_probs'], 0.5) for r in runs])
        test_val_thr = np.mean([macro_f1(yt, r['test_probs'], r['thresholds']) for r in runs])
        test_oracle = np.mean([macro_f1(yt, r['test_probs'], oracle_thresholds(yt, r['test_probs'])) for r in runs])
        mean_thr = ', '.join(f'{t:.2f}' for t in np.mean([r['thresholds'] for r in runs], axis=0))
        print(f"| {model} | {name} | {gain_v:+.3f} | {gain_t:+.3f} | {test_val_thr:.3f} | {test_oracle:.3f} | "
              f"{test_oracle - test_val_thr:.3f} | {mean_thr} |")

print(f"\n### Transformer: dominio legale contro generale (4 epoche)\n\n{HEADER}")
for name, file in (('LEGAL-BERT (Deliverable)', 'task6_runs.pkl'), ('bert-base-uncased', 'task6_runs_bert-base-uncased_4epochs.pkl')):
    runs = load(RESULTS / file)
    print(row(name, runs) if runs else f"| {name} | in corso | | | | | |")
