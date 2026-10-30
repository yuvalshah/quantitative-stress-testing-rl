"""Stage 7: portfolio-aware recommendation of validated stress scenarios from the Stage 6 library.

A recommendation layer only: it reads ``results/stage6/scenario_library.json`` (schema ``stage6.v2``), turns
each scenario into features, filters for eligibility with the Stage 6 validation thresholds, scores the
eligible scenarios with a transparent weighted sum of four normalised components and selects a shortlist
either by score alone, by greedy diversity-aware ranking, or at random (baseline). It generates no new
scenarios and changes no simulator, limit or plausibility definition.

Score components (each in [0, 1], higher = more useful):
* ``relevance``    - alignment of the scenario's cumulative factor shock with the portfolio's factor
                     sensitivities: (cos + 1) / 2 of the angle between the shock vector and the loss
                     direction -(w'B) (both in factor-sd units).
* ``impact``       - the target portfolio's deterministic path drawdown when the scenario is replayed,
                     min-max scaled across the eligible set.
* ``plausibility`` - ``ln(rank / q) / ln(1 / q)`` clamped to [0, 1] (0 at the threshold q).
* ``robustness``   - lower 95% Wilson bound of the reported valid-breach probability.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from rlstress.scenario_library import (
    FACTOR_ORDER,
    LIBRARY_SCHEMA_VERSION,
    cumulative_feature,
    validate_library,
)
from rlstress.search import plausibility_margin

COMPONENTS = ("relevance", "impact", "plausibility", "robustness")
TARGET_FAMILIES = ("broad-market", "technology-led", "financial-led", "energy-led", "mixed-direction")


@dataclass(frozen=True)
class RecommenderConfig:
    """Eligibility thresholds, score weights and diversity settings (weights must sum to 1)."""

    q: float = 0.01                          # plausibility threshold (unchanged project value)
    min_screening_p_valid: float = 0.80      # Stage 6 selection rule
    min_reporting_episodes: int = 100
    max_p_implausible: float = 0.05
    require_validated: bool = True           # Stage 6 status == "validated"
    weights: dict[str, float] = field(default_factory=lambda: {c: 0.25 for c in COMPONENTS})
    diversity_penalty: float = 0.5           # lambda in score - lambda * max similarity to selected
    similarity_scale: float = 0.671          # tau in exp(-d / tau); default = Stage 6 cluster cut (two action steps)
    top_k: int = 5

    def __post_init__(self) -> None:
        if set(self.weights) != set(COMPONENTS):
            raise ValueError(f"weights must be given for exactly {COMPONENTS}")
        if any(w < 0 for w in self.weights.values()) or not np.isclose(sum(self.weights.values()), 1.0):
            raise ValueError("weights must be non-negative and sum to 1")
        if not 0 < self.q < 1 or not 0 <= self.min_screening_p_valid <= 1 or not 0 <= self.max_p_implausible <= 1:
            raise ValueError("thresholds must be probabilities")
        if self.diversity_penalty < 0 or self.similarity_scale <= 0 or self.top_k < 1 or self.min_reporting_episodes < 0:
            raise ValueError("diversity_penalty >= 0, similarity_scale > 0, top_k >= 1, min_reporting_episodes >= 0")


# --- loading ---------------------------------------------------------------------------------------

def load_library(path: Path) -> dict[str, Any]:
    """Read and validate the Stage 6 library; raises FileNotFoundError / ValueError instead of guessing."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Stage 6 library not found: {path}")
    lib = json.loads(path.read_text())
    meta = lib.get("metadata", {})
    rule = meta.get("selection_rule", {})
    if lib.get("schema_version") != LIBRARY_SCHEMA_VERSION:
        raise ValueError(f"unexpected library schema {lib.get('schema_version')!r}")
    validate_library(lib, meta.get("plausibility_threshold_q", 0.01), rule.get("p_valid_target", 0.80))
    if not lib["scenarios"]:
        raise ValueError("the Stage 6 library contains no scenarios")
    if tuple(meta.get("factor_order", ())) != FACTOR_ORDER:
        raise ValueError(f"factor order {meta.get('factor_order')} != {FACTOR_ORDER}")
    return lib


def library_fingerprint(path: Path) -> str:
    """SHA-256 of the library file (provenance)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- portfolio profile ---------------------------------------------------------------------------

@dataclass
class PortfolioProfile:
    """Risk profile of a portfolio over the shock model's eight holdings (all quantities computed, none assumed)."""

    name: str
    tickers: tuple[str, ...]
    weights: np.ndarray
    capital: float
    factor_loadings: dict[str, float]          # w'B (Stage 4 raw loadings, training fit)
    loss_direction: np.ndarray                 # -(w'B) * factor sd, unit length (direction of factor shocks that hurt)
    sector_weights: dict[str, float]
    annual_volatility: float                   # training split, constant-mix daily returns
    max_drawdown: float                        # training split, negative fraction
    drawdown_limit: float
    vulnerabilities: list[str]

    def summary(self) -> dict[str, Any]:
        """JSON-ready form."""
        d = asdict(self)
        d["weights"] = dict(zip(self.tickers, self.weights.tolist()))
        d["loss_direction"] = dict(zip([f.removesuffix("_return") for f in FACTOR_ORDER], self.loss_direction.tolist()))
        return d


def portfolio_profile(name: str, tickers: Sequence[str], weights: Sequence[float], capital: float, raw_loadings: np.ndarray,
                      factor_sd: np.ndarray, sectors: dict[str, str], train_returns: pd.DataFrame, drawdown_limit: float,
                      annualization: int = 252) -> PortfolioProfile:
    """Build a profile from the shock model's loadings and *training-split* stock returns only."""
    w = np.asarray(weights, dtype=float)
    if w.shape != (len(tickers),) or not np.isclose(w.sum(), 1.0) or (w < 0).any():
        raise ValueError("weights must be non-negative, one per ticker, and sum to 1")
    B = np.asarray(raw_loadings, dtype=float)
    beta_p = w @ B
    direction = -beta_p * np.asarray(factor_sd, dtype=float)
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("portfolio has no factor sensitivity")
    rets = train_returns[list(tickers)].dropna()
    rp = rets.to_numpy() @ w
    values = np.cumprod(1 + rp)
    mdd = float(np.min(values / np.maximum.accumulate(values) - 1))
    sec: dict[str, float] = {}
    for t, wt in zip(tickers, w):
        sec[sectors.get(t, "unknown")] = sec.get(sectors.get(t, "unknown"), 0.0) + float(wt)
    names = [f.removesuffix("_return") for f in FACTOR_ORDER]
    order = np.argsort(beta_p)[::-1]
    vul = [f"largest factor sensitivity: {names[order[0]]} (w'B = {beta_p[order[0]]:.2f})",
           f"largest sector weight: {max(sec, key=sec.get)} ({max(sec.values()):.0%})",
           f"training-period max drawdown {mdd:.1%} vs {drawdown_limit:.0%} 5-day drawdown limit"]
    return PortfolioProfile(name, tuple(tickers), w, float(capital), dict(zip(names, beta_p.tolist())), direction / norm,
                            sec, float(np.std(rp, ddof=1) * np.sqrt(annualization)), mdd, float(drawdown_limit), vul)


# --- features ----------------------------------------------------------------------------------------

def scenario_features(library: dict[str, Any], factor_sd: np.ndarray, horizon: int) -> pd.DataFrame:
    """One row per library scenario; missing evidence stays NaN (never treated as robust)."""
    rows = []
    for s in library["scenarios"]:
        t = int(s["executed_days"])
        full = np.asarray(s["robust_action_path"], dtype=float)      # the path whose robustness Stage 6 measured
        padded = np.zeros((horizon, len(FACTOR_ORDER)))
        padded[: len(full)] = full
        cum_return = np.prod(1 + np.asarray(s["factor_shock_path"], dtype=float), axis=0) - 1
        ci = s.get("p_valid_ci95") or [np.nan, np.nan]
        rows.append({
            "scenario_id": s["scenario_id"], "status": s["status"], "family": s.get("family") or "unclassified",
            "cluster": s.get("cluster"), "discovery_methods": "; ".join(s.get("discovery_methods") or []),
            "representative_candidate_id": s.get("representative_candidate_id"), "executed_days": t,
            "rank": _num(s.get("deterministic_rank")), "deterministic_drawdown": _num(s.get("deterministic_drawdown")),
            "deterministic_breach_day": _num(s.get("deterministic_breach_day")),
            "p_valid": _num(s.get("p_valid")), "p_valid_ci_low": _num(ci[0]), "p_valid_ci_high": _num(ci[1]),
            "p_any_breach": _num(s.get("p_any_breach")), "p_implausible": _num(s.get("p_implausible")),
            "screening_p_valid": _num(s.get("screening_p_valid")), "screening_episodes": _num(s.get("screening_episodes")),
            "reporting_episodes": _num(s.get("reporting_episodes")),
            "mean_breach_day": _num(s.get("mean_breach_day")),
            "shock_sd_vector": _shock_in_sd(s, factor_sd).tolist(),
            "cum_factor_return": cum_return.tolist(),
            "path_feature": cumulative_feature(padded, t).tolist(),
            "padded_action_path": padded.tolist(),
        })
    return pd.DataFrame(rows)


def _num(x: Any) -> float:
    return float(x) if x is not None and np.isfinite(float(x)) else float("nan")


def _shock_in_sd(scenario: dict[str, Any], factor_sd: np.ndarray) -> np.ndarray:
    """Cumulative factor shock in model-sd units (sum of daily factor returns / factor sd)."""
    shocks = np.asarray(scenario["factor_shock_path"], dtype=float)
    return shocks.sum(axis=0) / np.asarray(factor_sd, dtype=float)


# --- eligibility and scoring --------------------------------------------------------------------

def eligibility(features: pd.DataFrame, cfg: RecommenderConfig) -> pd.DataFrame:
    """Adds ``eligible`` and ``ineligible_reason``; missing evidence makes a scenario ineligible."""
    out = features.copy()
    reasons = []
    for _, r in out.iterrows():
        why = []
        if cfg.require_validated and r["status"] != "validated":
            why.append("not validated in Stage 6")
        for col in ("rank", "screening_p_valid", "p_valid", "p_valid_ci_low", "p_implausible", "reporting_episodes"):
            if not np.isfinite(r[col]):
                why.append(f"missing {col}")
        if np.isfinite(r["rank"]) and r["rank"] < cfg.q:
            why.append(f"rank {r['rank']:.4f} < q")
        if np.isfinite(r["screening_p_valid"]) and r["screening_p_valid"] < cfg.min_screening_p_valid:
            why.append(f"screening P(valid) {r['screening_p_valid']:.1%} < {cfg.min_screening_p_valid:.0%}")
        if np.isfinite(r["reporting_episodes"]) and r["reporting_episodes"] < cfg.min_reporting_episodes:
            why.append(f"only {int(r['reporting_episodes'])} reporting episodes")
        if np.isfinite(r["p_implausible"]) and r["p_implausible"] > cfg.max_p_implausible:
            why.append(f"P(implausible) {r['p_implausible']:.1%} > {cfg.max_p_implausible:.0%}")
        reasons.append("; ".join(why))
    out["ineligible_reason"] = reasons
    out["eligible"] = out["ineligible_reason"] == ""
    return out


def relevance(shock_sd: np.ndarray, profile: PortfolioProfile) -> float:
    """(cos + 1) / 2 between the cumulative shock (sd units) and the portfolio's loss direction."""
    v = np.asarray(shock_sd, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        return 0.5
    return float((np.dot(v / n, profile.loss_direction) + 1) / 2)


def score(features: pd.DataFrame, profile: PortfolioProfile, impact_drawdown: dict[str, float], cfg: RecommenderConfig) -> pd.DataFrame:
    """Component scores and the weighted total for *eligible* rows (ineligible rows get NaN).

    ``impact_drawdown`` maps scenario_id to the target portfolio's replayed path drawdown (positive fraction).
    """
    out = features.copy()
    out["portfolio_drawdown"] = out["scenario_id"].map(impact_drawdown).astype(float)
    elig = out["eligible"] & out["portfolio_drawdown"].notna()
    out.loc[out["eligible"] & out["portfolio_drawdown"].isna(), "ineligible_reason"] = "missing portfolio impact"
    out["eligible"] = elig
    out["relevance"] = [relevance(np.asarray(v), profile) for v in out["shock_sd_vector"]]
    dd = out.loc[elig, "portfolio_drawdown"]
    span = dd.max() - dd.min() if len(dd) else 0.0
    out["impact"] = np.where(elig, (out["portfolio_drawdown"] - dd.min()) / span if span > 0 else 1.0, np.nan)
    out["plausibility"] = [plausibility_margin(r, cfg.q) for r in out["rank"]]
    out["robustness"] = out["p_valid_ci_low"].clip(0, 1)
    for c in COMPONENTS:
        out.loc[~elig, c] = np.nan
    out["score"] = sum(cfg.weights[c] * out[c] for c in COMPONENTS)
    return out


# --- selection -----------------------------------------------------------------------------------------

def similarity(a: Sequence[float], b: Sequence[float], scale: float) -> float:
    """exp(-d / scale) with d the Stage 6 cumulative-path Euclidean distance (1 = identical)."""
    return float(np.exp(-np.linalg.norm(np.asarray(a) - np.asarray(b)) / scale))


def _eligible_sorted(scored: pd.DataFrame) -> pd.DataFrame:
    e = scored[scored["eligible"]].copy()
    return e.sort_values(["score", "scenario_id"], ascending=[False, True])


def select_top(scored: pd.DataFrame, k: int) -> list[str]:
    """Highest individual scores (ties broken by scenario_id)."""
    return list(_eligible_sorted(scored)["scenario_id"].head(k))


def select_diverse(scored: pd.DataFrame, k: int, cfg: RecommenderConfig) -> list[dict[str, Any]]:
    """Greedy diversity-aware selection: maximise score - lambda * max similarity to the already selected."""
    pool = _eligible_sorted(scored)
    chosen: list[dict[str, Any]] = []
    feats = {r.scenario_id: r.path_feature for r in pool.itertuples()}
    remaining = list(pool.itertuples())
    while remaining and len(chosen) < k:
        best, best_val, best_sim = None, -np.inf, 0.0
        for r in remaining:
            sim = max((similarity(r.path_feature, feats[c["scenario_id"]], cfg.similarity_scale) for c in chosen), default=0.0)
            val = r.score - cfg.diversity_penalty * sim
            if val > best_val + 1e-12 or (abs(val - best_val) <= 1e-12 and best is not None and r.scenario_id < best.scenario_id):
                best, best_val, best_sim = r, val, sim
        chosen.append({"scenario_id": best.scenario_id, "score": best.score, "adjusted_score": best_val, "max_similarity_to_selected": best_sim})
        remaining = [r for r in remaining if r.scenario_id != best.scenario_id]
    return chosen


def select_random(scored: pd.DataFrame, k: int, seed: int) -> list[str]:
    """Uniform random eligible shortlist (baseline)."""
    ids = sorted(scored.loc[scored["eligible"], "scenario_id"])
    rng = np.random.default_rng(seed)
    return [ids[i] for i in rng.choice(len(ids), size=min(k, len(ids)), replace=False)] if ids else []


# --- evaluation ----------------------------------------------------------------------------------------

def shortlist_metrics(scored: pd.DataFrame, ids: Sequence[str], cfg: RecommenderConfig) -> dict[str, Any]:
    """Quality, diversity and coverage of a shortlist (denominator: shortlist size)."""
    sel = scored.set_index("scenario_id").loc[list(ids)] if ids else scored.iloc[0:0].set_index("scenario_id")
    n = len(sel)
    feats = list(sel["path_feature"]) if n else []
    pair = [np.linalg.norm(np.asarray(a) - np.asarray(b)) for i, a in enumerate(feats) for b in feats[i + 1:]]
    sims = [np.exp(-d / cfg.similarity_scale) for d in pair]
    return {"size": n, "eligibility compliance": float(sel["eligible"].mean()) if n else float("nan"),
            "mean score": float(sel["score"].mean()) if n else float("nan"),
            "mean relevance": float(sel["relevance"].mean()) if n else float("nan"),
            "mean portfolio drawdown": float(sel["portfolio_drawdown"].mean()) if n else float("nan"),
            "mean P(valid)": float(sel["p_valid"].mean()) if n else float("nan"),
            "min P(valid)": float(sel["p_valid"].min()) if n else float("nan"),
            "mean rank": float(sel["rank"].mean()) if n else float("nan"),
            "distinct clusters": int(sel["cluster"].nunique()) if n else 0,
            "distinct families": int(sel["family"].nunique()) if n else 0,
            "min pairwise distance": float(min(pair)) if pair else float("nan"),
            "mean pairwise similarity": float(np.mean(sims)) if sims else float("nan")}


def coverage_gaps(features: pd.DataFrame) -> pd.DataFrame:
    """Which target families are present in the library and among eligible scenarios."""
    present_all = set(features["family"])
    present_elig = set(features.loc[features["eligible"], "family"]) if "eligible" in features else set()
    return pd.DataFrame([{"family": f, "in library": f in present_all, "eligible": f in present_elig} for f in TARGET_FAMILIES])


# --- explanations ------------------------------------------------------------------------------------

def explain(row: pd.Series, profile: PortfolioProfile, previous: Sequence[pd.Series], cfg: RecommenderConfig,
            holding_pnl: dict[str, float] | None = None) -> str:
    """Plain-text explanation built only from computed features."""
    names = [f.removesuffix("_return") for f in FACTOR_ORDER]
    shock = np.asarray(row["shock_sd_vector"])
    lead = names[int(np.argmin(shock))]
    parts = [f"{row['family']} scenario over {int(row['executed_days'])} day(s); largest cumulative shock on {lead} "
             f"({shock.min():+.1f} model sd; cumulative factor returns " + ", ".join(f"{n} {r:+.1%}" for n, r in zip(names, row["cum_factor_return"])) + ")."]
    parts.append(f"Relevance {row['relevance']:.2f}: alignment with this portfolio's loss direction (largest sensitivity "
                 f"{max(profile.factor_loadings, key=profile.factor_loadings.get)}).")
    parts.append(f"Impact: replayed path drawdown {row['portfolio_drawdown']:.2%} for '{profile.name}' (limit {profile.drawdown_limit:.0%}).")
    if holding_pnl:
        worst = sorted(holding_pnl.items(), key=lambda kv: kv[1])[:3]
        parts.append("Largest holding losses: " + ", ".join(f"{t} {p:,.0f}" for t, p in worst) + ".")
    parts.append(f"Evidence: rank {row['rank']:.4f} (q = {cfg.q}); P(valid) {row['p_valid']:.1%} "
                 f"[{row['p_valid_ci_low']:.1%}, {row['p_valid_ci_high']:.1%}] on {int(row['reporting_episodes'])} reporting episodes; "
                 f"P(implausible) {row['p_implausible']:.1%}; screening P(valid) {row['screening_p_valid']:.1%} on {int(row['screening_episodes'])}.")
    if previous:
        d = [(float(np.linalg.norm(np.asarray(row["path_feature"]) - np.asarray(p["path_feature"]))), p["scenario_id"], p) for p in previous]
        dmin, pid, p = min(d)
        diff = "different family" if p["family"] != row["family"] else ("different cluster" if p["cluster"] != row["cluster"] else "same cluster")
        parts.append(f"Differs from earlier picks: nearest {pid} at distance {dmin:.2f} ({diff}).")
    parts.append("Limitations: fixed five-day path replayed in the Stage 4 model (i.i.d. daily shocks, tails lighter than history); "
                 "robustness from Stage 6 reporting episodes (validation-period residual draws, separate from the screening episodes "
                 "Stage 6 used for selection); not a test-period result.")
    return " ".join(parts)


# --- sensitivity and independent evaluation ------------------------------------------------------

def exact_weight_shift(weights: dict[str, float], component: str, delta: float) -> dict[str, float]:
    """Move ``component`` by exactly ``delta`` and each other weight by ``-delta / (n - 1)`` (sum stays 1)."""
    if component not in weights:
        raise ValueError(f"unknown component {component!r}")
    others = [c for c in weights if c != component]
    out = {c: weights[c] - delta / len(others) for c in others}
    out[component] = weights[component] + delta
    if any(v < -1e-12 for v in out.values()):
        raise ValueError(f"shift {delta:+} of {component} makes a weight negative")
    return {c: float(max(0.0, out[c])) for c in weights}


def shortlist_fresh_metrics(fresh: pd.DataFrame, ids: Sequence[str]) -> dict[str, float]:
    """Mean / min P(valid), mean P(implausible) and drawdown at breach of a shortlist from an independent evaluation.

    ``fresh`` is indexed by scenario_id with columns ``p_valid``, ``p_implausible``, ``mean_drawdown_at_breach``,
    ``episodes`` (one independent evaluation per scenario; denominator per scenario: its episodes).
    """
    if not len(ids):
        return {"size": 0, "fresh mean P(valid)": float("nan"), "fresh min P(valid)": float("nan"),
                "fresh mean P(implausible)": float("nan"), "fresh mean drawdown at breach": float("nan")}
    sel = fresh.loc[list(ids)]
    return {"size": len(sel), "fresh mean P(valid)": float(sel["p_valid"].mean()), "fresh min P(valid)": float(sel["p_valid"].min()),
            "fresh mean P(implausible)": float(sel["p_implausible"].mean()),
            "fresh mean drawdown at breach": float(sel["mean_drawdown_at_breach"].mean())}
