"""The confidence gate.

Why this module is the point of the whole repo:

Every public notebook that bolts database retrieval onto library search scores
WORSE than library search alone (0.25, 0.25, 0.19 against a 0.339 baseline, as
of 19 Sep 2026). The reason is mechanical. MRR@25 pays 1.0 for rank 1 and 0.33
for rank 3. Interleaving retrieved candidates pushes correct library hits down
two or three slots, and the class-1 loss is larger than the class-2 gain.

So the problem is not "retrieve better". It is "know when the library answer is
trustworthy". A calibrated per-molecule probability that the top library
candidate is correct lets retrieval fill slots 2-25 for free when the library
is confident, and take the top slots when it is not. Free because a wrong guess
costs nothing but the slot it occupies.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import lightgbm as lgb
except ImportError:  # keep the module importable without lightgbm
    lgb = None


FEATURE_ORDER: list[str] = [
    "n_candidates", "n_spectra", "top_best_score", "top_rrf", "top_sum_score",
    "top_support_frac", "top_is_fragment_channel", "margin_score", "margin_rrf",
    "margin_ratio", "score_mean", "score_std", "score_top_minus_mean",
    "score_entropy", "precursor_mz", "mean_n_peaks", "log_base_peak",
    "n_adducts", "n_energies",
]


def features_to_matrix(rows: list[dict[str, float]]) -> np.ndarray:
    return np.array([[r.get(f, 0.0) for f in FEATURE_ORDER] for r in rows], dtype=np.float64)


@dataclass
class ConfidenceGate:
    """P(the top library candidate for this molecule is the correct structure).

    Trained on validation folds where the truth is known. The label is simply
    whether the library's rank-1 candidate matched. Deliberately a small model:
    the signal is in the margin and cross-spectrum agreement features, and a
    big model on ~thousands of molecules will just memorise folds.
    """
    n_estimators: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 15
    min_child_samples: int = 30
    model: object | None = field(default=None, repr=False)
    fallback_rate: float = 0.5

    def fit(self, features: list[dict[str, float]], labels: list[int]) -> "ConfidenceGate":
        y = np.asarray(labels, dtype=int)
        if lgb is None or len(y) < 50 or y.min() == y.max():
            # Degenerate data: fall back to the base rate rather than pretending.
            self.model = None
            self.fallback_rate = float(y.mean()) if len(y) else 0.5
            return self
        X = features_to_matrix(features)
        params = {"objective": "binary", "learning_rate": self.learning_rate,
                  "num_leaves": self.num_leaves,
                  "min_data_in_leaf": self.min_child_samples,
                  "verbose": -1, "feature_pre_filter": False}
        self.model = lgb.train(params, lgb.Dataset(X, label=y),
                               num_boost_round=self.n_estimators)
        return self

    def predict_proba(self, features: list[dict[str, float]]) -> np.ndarray:
        if self.model is None:
            return np.full(len(features), self.fallback_rate)
        return np.asarray(self.model.predict(features_to_matrix(features)))


def weighted_rrf(library: list[str], retrieval: list[str], p: float,
                 k: int = 25, rrf_k: int = 10,
                 protect_threshold: float = 0.6) -> list[str]:
    """Merge two ranked SMILES lists using the gate probability as the weight.

    p near 1 -> the library list dominates and its top hit is pinned to rank 1.
    p near 0 -> retrieval takes the top slots.
    In between, both lists contribute in proportion.

    `protect_threshold` is the whole defence against the failure that sinks the
    public notebooks: above it, nothing is allowed to displace the library's
    rank-1 candidate.
    """
    scores: dict[str, float] = {}
    for rank, smi in enumerate(library, start=1):
        scores[smi] = scores.get(smi, 0.0) + p / (rrf_k + rank)
    for rank, smi in enumerate(retrieval, start=1):
        scores[smi] = scores.get(smi, 0.0) + (1.0 - p) / (rrf_k + rank)

    ordered = sorted(scores, key=lambda s: -scores[s])

    if p >= protect_threshold and library:
        pinned = library[0]
        ordered = [pinned] + [s for s in ordered if s != pinned]

    return ordered[:k]


def fill_slots(candidates: list[str], filler: list[str], k: int = 25) -> list[str]:
    """Pad a short list up to k with fallback guesses, de-duplicated.

    A wrong guess costs nothing. Submitting 8 candidates when you could submit
    25 is discarding free expected score.
    """
    out = list(dict.fromkeys(candidates))[:k]
    if len(out) >= k:
        return out
    seen = set(out)
    for s in filler:
        if s not in seen:
            out.append(s)
            seen.add(s)
            if len(out) >= k:
                break
    return out


def evaluate_policy(molecules: list[dict], gate: ConfidenceGate | None,
                    answers: dict[str, str], metric_fn,
                    protect_threshold: float = 0.6, k: int = 25,
                    fixed_p: float | None = None) -> dict:
    """Score a fusion policy over molecules.

    Each molecule dict carries: id, features, library (ranked SMILES),
    retrieval (ranked SMILES), filler (global fallback list).
    `fixed_p=1.0` reproduces library-only; `fixed_p=0.0` retrieval-only;
    `fixed_p=0.5` the naive interleave that the public notebooks do.
    """
    if fixed_p is None:
        if gate is None:
            raise ValueError("need a gate or a fixed_p")
        probs = gate.predict_proba([m["features"] for m in molecules])
    else:
        probs = np.full(len(molecules), float(fixed_p))

    preds = {}
    for m, p in zip(molecules, probs):
        merged = weighted_rrf(m["library"], m["retrieval"], float(p), k=k,
                              protect_threshold=protect_threshold)
        preds[m["id"]] = fill_slots(merged, m.get("filler", []), k=k)
    return {"predictions": preds, "mrr": metric_fn(preds, answers), "probs": probs}


# ---------------------------------------------------------------------------
# Two-sided gating.
#
# A single gate that only asks "is the library top-1 correct?" fails, and fails
# in a way that is obvious in hindsight: when the library is usually wrong the
# probability is low everywhere, so the policy hands the top slots to retrieval
# regardless of whether retrieval is any good. Measured on synthetic data, that
# is worse than doing nothing at all whenever retrieval is weak.
#
# The decision is relative. Rank 1 should go to whichever source is more likely
# to be right FOR THIS MOLECULE, which means scoring both sources and comparing.
# ---------------------------------------------------------------------------

RETRIEVAL_FEATURE_ORDER: list[str] = [
    "ret_n_candidates", "ret_pool_size", "ret_top_sim", "ret_margin",
    "ret_margin_ratio", "ret_mean_sim", "ret_std_sim", "ret_entropy",
    "ret_top_minus_mean", "ret_precursor_mz",
]


def retrieval_features(similarities: np.ndarray, pool_size: int,
                       precursor_mz: float = 0.0) -> dict[str, float]:
    """Describe how confident the retrieval side is.

    Same shape of signal as the library side: how high is the best score, how
    far clear of the runner-up, and how peaked the distribution is. A large
    candidate pool with a flat similarity profile means retrieval is guessing.
    """
    s = np.sort(np.asarray(similarities, dtype=np.float64))[::-1]
    f = {
        "ret_n_candidates": float(s.size),
        "ret_pool_size": float(pool_size),
        "ret_precursor_mz": float(precursor_mz),
    }
    if s.size == 0:
        for k in ("ret_top_sim", "ret_margin", "ret_margin_ratio", "ret_mean_sim",
                  "ret_std_sim", "ret_entropy", "ret_top_minus_mean"):
            f[k] = 0.0
        return f
    top = s[0]
    second = s[1] if s.size > 1 else 0.0
    f["ret_top_sim"] = float(top)
    f["ret_margin"] = float(top - second)
    f["ret_margin_ratio"] = float(top / second) if second > 0 else 10.0
    f["ret_mean_sim"] = float(s.mean())
    f["ret_std_sim"] = float(s.std())
    f["ret_top_minus_mean"] = float(top - s.mean())
    f["ret_entropy"] = _entropy_np(s)
    return f


def _entropy_np(s: np.ndarray) -> float:
    s = np.clip(s, 1e-12, None)
    p = s / s.sum()
    return float(-(p * np.log(p)).sum())


def _matrix(rows: list[dict[str, float]], order: list[str]) -> np.ndarray:
    return np.array([[r.get(f, 0.0) for f in order] for r in rows], dtype=np.float64)


@dataclass
class DualGate:
    """Two calibrated heads: P(library top-1 right) and P(retrieval top-1 right).

    Rank 1 goes to the more confident source. The rest of the list is fused in
    proportion, so a molecule where both sources are unsure still gets 25
    guesses drawn from both.
    """
    n_estimators: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 15
    min_child_samples: int = 30
    lib_model: object | None = field(default=None, repr=False)
    ret_model: object | None = field(default=None, repr=False)
    lib_rate: float = 0.5
    ret_rate: float = 0.5

    def _fit_one(self, X, y):
        """Train one head with LightGBM's native API.

        Deliberately not LGBMClassifier: the sklearn wrapper pulls in
        scikit-learn, and the scoring notebook runs with internet disabled, so
        every extra dependency is another wheel to package. lgb.train needs
        only lightgbm itself.
        """
        y = np.asarray(y, dtype=int)
        if lgb is None or len(y) < 50 or y.min() == y.max():
            return None, float(y.mean()) if len(y) else 0.5
        params = {"objective": "binary", "learning_rate": self.learning_rate,
                  "num_leaves": self.num_leaves,
                  "min_data_in_leaf": self.min_child_samples,
                  "verbose": -1, "feature_pre_filter": False}
        booster = lgb.train(params, lgb.Dataset(X, label=y),
                            num_boost_round=self.n_estimators)
        return booster, float(y.mean())

    def fit(self, lib_feats, lib_labels, ret_feats, ret_labels) -> "DualGate":
        self.lib_model, self.lib_rate = self._fit_one(
            _matrix(lib_feats, FEATURE_ORDER), lib_labels)
        self.ret_model, self.ret_rate = self._fit_one(
            _matrix(ret_feats, RETRIEVAL_FEATURE_ORDER), ret_labels)
        return self

    def predict(self, lib_feats, ret_feats) -> tuple[np.ndarray, np.ndarray]:
        n = len(lib_feats)
        p_lib = (np.full(n, self.lib_rate) if self.lib_model is None
                 else np.asarray(self.lib_model.predict(_matrix(lib_feats, FEATURE_ORDER))))
        p_ret = (np.full(n, self.ret_rate) if self.ret_model is None
                 else np.asarray(self.ret_model.predict(_matrix(ret_feats, RETRIEVAL_FEATURE_ORDER))))
        return p_lib, p_ret


def dual_merge(library: list[str], retrieval: list[str],
               p_lib: float, p_ret: float, k: int = 25, rrf_k: int = 10,
               pin_winner: bool = True) -> list[str]:
    """Merge two lists with learned weights.

    `pin_winner=True` forces rank 1 to the more confident source. Measured on
    synthetic data this is usually WORSE than leaving the blend alone: a
    candidate both sources rank highly is stronger evidence than either source
    being confident on its own, and pinning throws that agreement away. Keep it
    False unless you can show otherwise on your own CV.
    """
    total = p_lib + p_ret
    w_lib = 0.5 if total <= 0 else p_lib / total
    scores: dict[str, float] = {}
    for rank, smi in enumerate(library, start=1):
        scores[smi] = scores.get(smi, 0.0) + w_lib / (rrf_k + rank)
    for rank, smi in enumerate(retrieval, start=1):
        scores[smi] = scores.get(smi, 0.0) + (1.0 - w_lib) / (rrf_k + rank)
    ordered = sorted(scores, key=lambda s: -scores[s])

    if pin_winner:
        winner = library[0] if (p_lib >= p_ret and library) else (retrieval[0] if retrieval else None)
        if winner is not None:
            ordered = [winner] + [s for s in ordered if s != winner]
    return ordered[:k]


def evaluate_dual(molecules: list[dict], gate: "DualGate", answers: dict[str, str],
                  metric_fn, k: int = 25, pin_winner: bool = False) -> dict:
    p_lib, p_ret = gate.predict([m["features"] for m in molecules],
                                [m["ret_features"] for m in molecules])
    preds = {}
    for m, pl, pr in zip(molecules, p_lib, p_ret):
        merged = dual_merge(m["library"], m["retrieval"], float(pl), float(pr), k=k,
                            pin_winner=pin_winner)
        preds[m["id"]] = fill_slots(merged, m.get("filler", []), k=k)
    return {"predictions": preds, "mrr": metric_fn(preds, answers),
            "p_lib": p_lib, "p_ret": p_ret}


def oracle_merge(molecules: list[dict], answers: dict[str, str], metric_fn,
                 inchikey_fn, k: int = 25) -> dict:
    """Upper bound: a perfect gate that always picks the right source.

    Run this every time you change the candidate generators. The gap between
    the oracle and your gate is how much a better gate can still buy; the gap
    between the oracle and 1.0 is how much only better candidates can buy.
    """
    preds = {}
    for m in molecules:
        lib_ok = bool(m["library"]) and inchikey_fn(m["library"][0]) == m["id"]
        ret_ok = bool(m["retrieval"]) and inchikey_fn(m["retrieval"][0]) == m["id"]
        if lib_ok:
            order = m["library"] + m["retrieval"]
        elif ret_ok:
            order = m["retrieval"] + m["library"]
        else:
            order = m["library"] + m["retrieval"]
        preds[m["id"]] = fill_slots(order, m.get("filler", []), k=k)
    return {"predictions": preds, "mrr": metric_fn(preds, answers)}
