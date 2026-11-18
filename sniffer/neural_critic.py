"""Neural critic: a network trained on every cue pair the search has ever scored, used to decide
which newly generated prompts are worth spending forward passes on.

Why: scoring a candidate costs 2 x n_tasks forward passes through the 70B model, so the search can
only afford to score a small population each round. Generating candidates is cheap. The critic
closes that gap: the generators (Llama + corpus recombination) propose a pool several times larger
than the population, the critic predicts each one's fitness, and only the most promising (plus a
slice picked at random, so the critic can't trap the search) get scored. Every scored candidate
then becomes new training data, so the critic sharpens each generation.

Training data ("the datasets")
  - every archive.csv from past search runs on the same backend + model (runs/*/ and any
    --datasets paths), each row a scored (sandbox text, production text, probe gap, behaviour
    gap) example. The critic learns the raw gaps, not fitness: fitness is z-scored within each
    generation's batch, so its scale drifts between generations and runs;
  - every candidate scored so far in the current run.
  Rows from a different backend or model are skipped: their numbers live on a different scale.

Model
  An ensemble of small multilayer perceptrons (sklearn MLPRegressor) over hashed character
  and word n-gram features of the sandbox and production sides. Each member is trained on a
  bootstrap resample, so the ensemble spread gives an uncertainty estimate. Predicted gaps are
  combined into one score on a fixed scale (the training set's mean/sd), the same formula as
  fitness; selection ranks by
  mean + kappa * spread (an upper-confidence bound), which favours novel, unexplored phrasings.

Honesty check
  Before each generation is scored, the critic's predictions are stored; afterwards we report
  the Spearman rank correlation between predicted and actual fitness on those unseen candidates.
  Near zero means the network hasn't learned anything useful yet and selection is ~random.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- dataset

def _archive_files(extra_paths=(), include_defaults=True):
    """Every archive.csv under runs/, results_evolved*/ and any user-given dirs/files.
    include_defaults=False reads only extra_paths (a shared server passes the visitor's own runs)."""
    found = set()
    defaults = (list((ROOT / "runs").glob("*/archive.csv")) + list(ROOT.glob("results_evolved*/archive.csv"))
                if include_defaults else [])
    for p in defaults:
        found.add(p.resolve())
    for raw in extra_paths or ():
        p = Path(raw).expanduser()
        if p.is_dir():
            found.update(q.resolve() for q in p.rglob("archive.csv"))
        elif p.suffix == ".csv" and p.exists():
            found.add(p.resolve())
    return sorted(found)


def _source_backend(csv_path: Path):
    s = csv_path.parent / "summary.json"
    if not s.exists():
        return None
    try:
        return json.loads(s.read_text()).get("backend")
    except (OSError, json.JSONDecodeError):
        return None


def load_prior(backend_name, extra_paths=(), exclude_dir=None, log=print, include_defaults=True):
    """Scored rows from earlier runs whose backend matches this one. Returns (rows, n_sources)."""
    import pandas as pd

    rows, sources, skipped = [], 0, 0
    exclude = Path(exclude_dir).resolve() if exclude_dir else None
    for f in _archive_files(extra_paths, include_defaults):
        if exclude and f.parent == exclude:
            continue
        if _source_backend(f) != backend_name:
            skipped += 1
            continue
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if not {"probe_gap", "behaviour_gap"} <= set(df.columns):
            continue
        df = df[np.isfinite(pd.to_numeric(df["probe_gap"], errors="coerce"))
                & np.isfinite(pd.to_numeric(df["behaviour_gap"], errors="coerce"))]
        for r in df.itertuples():
            rows.append((str(r.sandbox), str(r.prod), float(r.probe_gap), float(r.behaviour_gap)))
        sources += 1
    if skipped:
        log(f"  critic: skipped {skipped} past run(s) from a different backend/model.", flush=True)
    return rows, sources


# --------------------------------------------------------------------------- model

class NeuralCritic:
    def __init__(self, n_members=5, hidden=(64, 32), kappa=0.5, explore_frac=0.25,
                 min_rows=12, seed=0, lam=1.0, marker_penalty=0.5, hits_fn=None):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.n_members, self.hidden, self.kappa = n_members, hidden, kappa
        self.explore_frac, self.min_rows, self.seed = explore_frac, min_rows, seed
        self.lam, self.marker_penalty = lam, marker_penalty
        self.hits_fn = hits_fn or (lambda text: 0)
        self._chars = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=2 ** 12,
                                        alternate_sign=False, norm="l2")
        self._words = HashingVectorizer(analyzer="word", ngram_range=(1, 2), n_features=2 ** 11,
                                        alternate_sign=False, norm="l2", token_pattern=r"[A-Za-z0-9_]+")
        self.members, self.n_rows = [], 0
        self._mu, self._sd = np.zeros(2), np.ones(2)   # [probe, behaviour]

    @property
    def ready(self):
        return bool(self.members)

    def _features(self, pairs):
        from scipy.sparse import hstack

        sb = [s for s, _ in pairs]
        pr = [p for _, p in pairs]
        return hstack([self._chars.transform(sb), self._words.transform(sb),
                       self._chars.transform(pr), self._words.transform(pr)]).tocsr()

    def combine(self, probe_gap, behaviour_gap, hits):
        """Fitness on the critic's fixed scale: z(behaviour) + lam * z(probe) - penalty * hits."""
        zp = (probe_gap - self._mu[0]) / self._sd[0]
        zb = (behaviour_gap - self._mu[1]) / self._sd[1]
        return float(zb + self.lam * zp - self.marker_penalty * hits)

    def fit(self, rows):
        """rows: [(sandbox_text, prod_text, probe_gap, behaviour_gap)]. Retrains from scratch."""
        from sklearn.neural_network import MLPRegressor

        # Keep the latest score for any repeated pair.
        uniq = {}
        for sb, pr, pg, bg in rows:
            if np.isfinite(pg) and np.isfinite(bg):
                uniq[(sb.lower(), pr.lower())] = (sb, pr, pg, bg)
        rows = list(uniq.values())
        self.n_rows = len(rows)
        if len(rows) < self.min_rows:
            self.members = []
            return self
        X = self._features([(sb, pr) for sb, pr, _, _ in rows])
        y = np.array([[r[2], r[3]] for r in rows], float)          # [probe, behaviour]
        self._mu = y.mean(0)
        self._sd = np.where(y.std(0) > 1e-9, y.std(0), 1.0)
        yz = (y - self._mu) / self._sd
        rng = np.random.default_rng(self.seed + len(rows))
        self.members = []
        for m in range(self.n_members):
            idx = rng.integers(0, len(rows), len(rows))  # bootstrap -> ensemble disagreement
            net = MLPRegressor(hidden_layer_sizes=self.hidden, alpha=1e-3, learning_rate_init=3e-3,
                               max_iter=400, random_state=int(self.seed * 97 + m))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # ConvergenceWarning on tiny sets is expected
                net.fit(X[idx], yz[idx])
            self.members.append(net)
        return self

    def predict(self, pairs):
        """(mean, spread) across the ensemble of the predicted combined score (critic scale)."""
        if not self.members or not pairs:
            n = len(pairs)
            return np.zeros(n), np.zeros(n)
        X = self._features(pairs)
        hits = np.array([self.hits_fn(sb + " " + pr) for sb, pr in pairs], float)
        Z = np.stack([m.predict(X).reshape(len(pairs), 2) for m in self.members])  # (members, n, 2)
        S = Z[..., 1] + self.lam * Z[..., 0] - self.marker_penalty * hits
        return S.mean(0), S.std(0)

    def select(self, cands, k, rng, keep=()):
        """Pick k candidates: `keep` first (e.g. a reserved slice of model-written ones), then the
        best by upper-confidence bound, with an explore_frac slice drawn at random. Annotates each
        chosen candidate's extras with the prediction."""
        keep = list(keep)[:k]
        rest = [c for c in cands if all(c is not q for q in keep)]
        pairs = [(" ; ".join(c.sandbox), " ; ".join(c.prod)) for c in keep + rest]
        mean, spread = self.predict(pairs)
        for c, m, s in zip(keep + rest, mean, spread):
            c.extras = {**c.extras, "pred": float(m), "pred_sd": float(s)}
        need = k - len(keep)
        if need <= 0 or not rest:
            return keep[:k]
        if not self.ready:
            order = rng.permutation(len(rest))
            return keep + [rest[i] for i in order[:need]]
        ucb = np.array([c.extras["pred"] + self.kappa * c.extras["pred_sd"] for c in rest])
        n_explore = int(round(self.explore_frac * need)) if len(rest) > need else 0
        ranked = list(np.argsort(-ucb))
        chosen = ranked[:need - n_explore]
        leftover = ranked[need - n_explore:]
        if n_explore and leftover:
            chosen += list(rng.choice(leftover, size=min(n_explore, len(leftover)), replace=False))
        return keep + [rest[i] for i in chosen]


def rank_corr(pred, actual):
    """Spearman correlation, NaN if either side is constant or too short."""
    from scipy.stats import spearmanr

    pred, actual = np.asarray(pred, float), np.asarray(actual, float)
    ok = np.isfinite(pred) & np.isfinite(actual)
    if ok.sum() < 4 or np.std(pred[ok]) < 1e-12 or np.std(actual[ok]) < 1e-12:
        return float("nan")
    return float(spearmanr(pred[ok], actual[ok]).statistic)
