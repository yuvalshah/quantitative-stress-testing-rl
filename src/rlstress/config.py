"""Typed configuration: portfolio settings, risk limits and simulation parameters.

The configuration lives in a TOML file (``config/default.toml`` by default) and is
parsed with the standard-library ``tomllib`` so no extra dependency is required.
This module is the base layer of the package and must not import any other
``rlstress`` module.
"""

from __future__ import annotations

import math
import re
import tomllib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.toml"


class ConfigError(ValueError):
    """Raised when a configuration file is missing fields or has invalid values."""


def resolve_path(path: str | Path, root: Path = PROJECT_ROOT) -> Path:
    """Resolve ``path`` against ``root`` unless it is already absolute."""
    p = Path(path)
    return p if p.is_absolute() else root / p


def parse_iso_date(value: str, name: str) -> date:
    """Parse a ``YYYY-MM-DD`` string, raising :class:`ConfigError` with the field name."""
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be an ISO date (YYYY-MM-DD), got {value!r}") from exc


PORTFOLIO_POLICIES = ("long_only",)
REBALANCING_POLICIES = ("daily_constant_mix",)
MISSING_DATA_POLICIES = ("complete_case", "raise")


@dataclass(frozen=True)
class PortfolioConfig:
    """Static description of the portfolio under test.

    Structural checks happen here; weight-policy checks (finite, long-only, sum to
    one within ``weight_tolerance``) live in :mod:`rlstress.portfolio.validation`
    and run whenever a :class:`~rlstress.portfolio.Portfolio` is built.
    """

    name: str
    tickers: tuple[str, ...]
    weights: tuple[float, ...]
    initial_capital: float
    base_currency: str = "USD"
    policy: str = "long_only"
    weight_tolerance: float = 1e-8
    rebalancing: str = "daily_constant_mix"
    missing_data_policy: str = "complete_case"
    return_column_template: str = "{ticker}_return"


ASSET_GROUPS = ("stock", "market_indicator", "volatility")
MACRO_KINDS = ("yield", "rate", "price")
_SYMBOL_RE = re.compile(r"^\^?[A-Z0-9.\-]{1,10}$")
_SERIES_ID_RE = re.compile(r"^[A-Z0-9]{1,30}$")
_ALIAS_RE = re.compile(r"^[A-Z][A-Z0-9]{0,19}$")


@dataclass(frozen=True)
class AssetSpec:
    """A Yahoo Finance series (stock, market-indicator ETF or volatility index).

    Attributes:
        symbol: Source ticker exactly as requested from Yahoo Finance (e.g. ``^VIX``).
        alias: Identifier used in file and column names (e.g. ``VIX``).
        name: Human-readable name.
        group: One of :data:`ASSET_GROUPS`.
        sector: Sector label (stocks only).
        role: Analytical role of the series in the project.
        required: Whether a download failure makes the collection incomplete.
        enabled: Disabled assets are configured but never downloaded.
    """

    symbol: str
    alias: str
    name: str
    group: str
    sector: str = ""
    role: str = ""
    required: bool = True
    enabled: bool = True


@dataclass(frozen=True)
class MacroSeriesSpec:
    """A FRED series.

    Attributes:
        series_id: FRED series identifier (e.g. ``DGS10``).
        alias: Identifier used in file and column names (e.g. ``WTI``).
        name: FRED series title.
        kind: ``"yield"`` or ``"rate"`` (percent levels, changes in basis points)
            or ``"price"`` (positive-price levels, simple/log returns).
        unit: Unit of the level, e.g. ``"percent"`` or ``"usd_per_barrel"``.
        required: Whether a failure makes the collection incomplete.
        enabled: Disabled series are configured but never downloaded.
    """

    series_id: str
    alias: str
    name: str
    kind: str
    unit: str
    required: bool = False
    enabled: bool = True

    @property
    def level_column(self) -> str:
        """Column name of the level in processed datasets."""
        if self.kind == "yield":
            return f"{self.alias}_yield_pct"
        if self.kind == "rate":
            return f"{self.alias}_rate_pct"
        return f"{self.alias}_price_{self.unit}"


@dataclass(frozen=True)
class DataPaths:
    """Output directories, relative to the project root unless absolute."""

    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    reports_dir: str = "data/reports"

    def resolve(self, root: Path = PROJECT_ROOT) -> DataPaths:
        """Return a copy with every directory resolved to an absolute path."""
        return DataPaths(
            raw_dir=str(resolve_path(self.raw_dir, root)),
            processed_dir=str(resolve_path(self.processed_dir, root)),
            reports_dir=str(resolve_path(self.reports_dir, root)),
        )


@dataclass(frozen=True)
class YahooSettings:
    """yfinance request settings."""

    auto_adjust: bool = False
    actions: bool = True
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0
    pause_seconds: float = 0.5


@dataclass(frozen=True)
class FredSettings:
    """FRED request settings. The API key is read from ``api_key_env`` at run time."""

    api_key_env: str = "FRED_API_KEY"
    api_url: str = "https://api.stlouisfed.org/fred/series/observations"
    series_info_url: str = "https://api.stlouisfed.org/fred/series"
    public_csv_url: str = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    allow_public_csv_fallback: bool = False
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0
    timeout_seconds: float = 30.0


@dataclass(frozen=True)
class CleaningSettings:
    """Validation thresholds and missing-data policies."""

    conflict_policy: str = "drop"
    max_gap_calendar_days: int = 5
    partial_coverage_tolerance_days: int = 10
    extreme_return_threshold: float = 0.15
    extreme_change_bp_threshold: float = 50.0
    yield_min_pct: float = -1.0
    yield_max_pct: float = 25.0
    macro_ffill_limit: int = 0


@dataclass(frozen=True)
class FeatureSettings:
    """Rolling-window feature settings."""

    windows: tuple[int, ...] = (5, 20, 60)
    min_obs_fraction: float = 0.8
    annualization_days: int = 252
    correlation_windows: tuple[int, ...] = (20, 60)
    drawdown_windows: tuple[int, ...] = (60,)
    include_log_returns: bool = True
    correlation_pairs: dict[str, tuple[str, ...]] = field(default_factory=dict)
    regime_source: str = "VIX_close"
    regime_quantiles: tuple[float, ...] = (0.25, 0.75)
    regime_labels: tuple[str, ...] = ("low", "normal", "high")

    def min_periods(self, window: int) -> int:
        """Minimum number of valid observations required in a window of ``window`` rows."""
        return max(1, math.ceil(self.min_obs_fraction * window))


@dataclass(frozen=True)
class SplitSettings:
    """Inclusive chronological split boundaries (ISO dates)."""

    train: tuple[str, str]
    validation: tuple[str, str]
    test: tuple[str, str]

    def as_dict(self) -> dict[str, tuple[str, str]]:
        """Splits in chronological order."""
        return {"train": self.train, "validation": self.validation, "test": self.test}


@dataclass(frozen=True)
class DataConfig:
    """Stage 2 data collection, cleaning and feature-engineering settings."""

    start_date: str
    end_date: str
    assets: tuple[AssetSpec, ...]
    macro: tuple[MacroSeriesSpec, ...]
    splits: SplitSettings
    interval: str = "1d"
    primary_return: str = "simple"
    seed: int = 42
    force_refresh: bool = False
    paths: DataPaths = field(default_factory=DataPaths)
    yahoo: YahooSettings = field(default_factory=YahooSettings)
    fred: FredSettings = field(default_factory=FredSettings)
    cleaning: CleaningSettings = field(default_factory=CleaningSettings)
    features: FeatureSettings = field(default_factory=FeatureSettings)

    def enabled_assets(self, group: str | None = None) -> tuple[AssetSpec, ...]:
        """Enabled Yahoo assets, optionally restricted to one group."""
        return tuple(
            a for a in self.assets if a.enabled and (group is None or a.group == group)
        )

    def enabled_macro(self) -> tuple[MacroSeriesSpec, ...]:
        """Enabled FRED series."""
        return tuple(m for m in self.macro if m.enabled)


# Metric name -> the only unit it may be expressed in. Values are loss magnitudes
# (positive = loss), so every limit compares "observed >= threshold" style rules.
RISK_METRIC_UNITS = {
    "cumulative_loss": "fraction_of_initial_value",
    "max_drawdown": "fraction_of_peak_value",
    "historical_var": "fraction_of_portfolio_value",
    "historical_cvar": "fraction_of_portfolio_value",
}
BREACH_RULES = (">=", ">")


@dataclass(frozen=True)
class RiskLimitSpec:
    """One risk limit with explicit metric, units, horizon, threshold and breach rule.

    Attributes:
        name: Identifier used in results.
        metric: One of :data:`RISK_METRIC_UNITS`.
        units: Must equal ``RISK_METRIC_UNITS[metric]``.
        horizon_days: Trading days the metric covers (1 for daily historical VaR/CVaR).
        threshold: Limit as a positive fraction, e.g. ``0.10`` = 10%.
        breach_rule: ``">="`` (equality breaches) or ``">"``.
        confidence: Required for ``historical_var``/``historical_cvar``, else unset.
        description: Free text.
    """

    name: str
    metric: str
    units: str
    horizon_days: int
    threshold: float
    breach_rule: str = ">="
    confidence: float | None = None
    description: str = ""


@dataclass(frozen=True)
class DirectShockScenario:
    """User-specified hypothetical asset returns over ``horizon_days`` (not a forecast)."""

    name: str
    horizon_days: int
    shocks: dict[str, float]
    description: str = ""


@dataclass(frozen=True)
class HistoricalReplaySpec:
    """A historical window of actual returns to replay against the portfolio."""

    name: str
    start_date: str
    horizon_days: int
    description: str = ""


@dataclass(frozen=True)
class RiskConfig:
    """Stage 3 risk-engine settings."""

    confidence_levels: tuple[float, ...] = (0.95, 0.99)
    default_confidence: float = 0.95
    annualization_days: int = 252
    volatility_windows: tuple[int, ...] = (5, 20, 60)
    min_observations: int = 250
    min_tail_observations: int = 10
    estimation_window: str = "train"
    report_windows: tuple[str, ...] = ("train", "full_sample")
    stress_horizon_days: int = 5
    comparison_tolerance: float = 1e-12
    extreme_loss_dates: int = 10
    worst_windows: int = 5
    returns_path: str = "data/processed/features/consolidated_daily.csv"
    output_dir: str = "results/stage3"
    limits: tuple[RiskLimitSpec, ...] = ()
    stress_scenarios: tuple[DirectShockScenario, ...] = ()
    historical_replays: tuple[HistoricalReplaySpec, ...] = ()


RESIDUAL_MODES = ("sample", "mean")
SHRINKAGE_METHODS = ("schafer_strimmer", "fixed")


@dataclass(frozen=True)
class SimulationConfig:
    """Stage 4 shock model and simulator settings (see docs/stage4_methodology.md)."""

    horizon_days: int = 5
    seed: int = 42
    fit_window: str = "train"
    anchor_window: str = "train"
    market_factor: str = "SPY_return"
    factors: tuple[str, ...] = ("SPY_return", "QQQ_return", "XLF_return", "XLE_return")
    candidate_factors: tuple[str, ...] = ()
    state_columns: tuple[str, ...] = ("VIX_close", "SPY_vol_ann_20d")
    max_factor_sigma: float = 8.0
    max_abs_factor_return: float = 0.25
    max_abs_stock_return: float = 0.60
    residual_mode: str = "sample"
    max_residual_draws: int = 200
    dof_grid: tuple[float, ...] = (3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0)
    shrinkage_method: str = "schafer_strimmer"
    fixed_shrinkage: float = 0.1
    em_max_iter: int = 1000
    em_tol: float = 1e-9
    min_plausibility: float = 0.01
    surprisal_cap: float = 25.0
    diagnostic_simulations: int = 50_000
    output_dir: str = "results/stage4"


@dataclass(frozen=True)
class EnvironmentConfig:
    """Reward-shaping parameters for the stress-testing environment."""

    breach_reward: float = 10.0
    severity_weight: float = 5.0
    plausibility_penalty: float = 0.2
    magnitude_penalty: float = 0.5
    step_penalty: float = 0.01
    invalid_penalty: float = 5.0
    implausible_breach_penalty: float = 0.0   # charged once when a breach is rejected as implausible (0 = legacy)
    implausible_breach_credit: float = 0.0    # kappa in [0, 1): graded partial credit for implausible breaches (0 = legacy)
    terminate_on_breach: bool = True


@dataclass(frozen=True)
class TrainingConfig:
    """Stage 5 PPO hyperparameters, seeds, evaluation protocol and output location."""

    algorithm: str = "PPO"
    policy: str = "MlpPolicy"
    total_timesteps: int = 100_000
    learning_rate: float = 3e-4
    n_steps: int = 1024
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    log_std_init: float = -1.0
    net_arch: tuple[int, ...] = (64, 64)
    normalize_observations: bool = True
    seeds: tuple[int, ...] = (0, 1, 2)
    device: str = "cpu"
    torch_threads: int = 1
    eval_freq: int = 10_000
    checkpoint_freq: int = 25_000
    selection_episodes: int = 200
    selection_seed: int = 10_000
    validation_episodes: int = 500
    eval_seed_sets: tuple[int, ...] = (20_000, 30_000, 40_000)
    random_baseline_seed_offset: int = 7_777
    test_episodes: int = 500
    test_seed: int = 90_000
    plausibility_thresholds: tuple[float, ...] = (0.005, 0.01, 0.02)
    scripted_action: float = -0.15
    n_representative_scenarios: int = 3
    output_dir: str = "results/stage5"


@dataclass(frozen=True)
class RecommendationConfig:
    """Weights of the scenario-ranking score."""

    severity_weight: float
    plausibility_weight: float
    relevance_weight: float
    top_k: int


@dataclass(frozen=True)
class MitigationConfig:
    """Candidate mitigation strategies and trading constraints."""

    strategies: tuple[str, ...]
    max_turnover: float
    transaction_cost_bps: float


@dataclass(frozen=True)
class StorageConfig:
    """Where discovered scenarios are persisted."""

    scenario_dir: str = "results/scenarios"


@dataclass(frozen=True)
class ProjectConfig:
    """Top-level configuration object combining every section."""

    portfolio: PortfolioConfig
    data: DataConfig
    risk: RiskConfig
    simulation: SimulationConfig
    environment: EnvironmentConfig
    training: TrainingConfig
    recommendation: RecommendationConfig
    mitigation: MitigationConfig
    storage: StorageConfig = field(default_factory=StorageConfig)


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    if name not in raw:
        raise ConfigError(f"Missing config section [{name}]")
    return dict(raw[name])


def _build(cls: type, values: dict[str, Any], section: str) -> Any:
    try:
        return cls(**values)
    except TypeError as exc:
        raise ConfigError(f"Invalid fields in [{section}]: {exc}") from exc


def data_config_from_dict(raw: dict[str, Any]) -> DataConfig:
    """Build (without cross-field validation) a :class:`DataConfig` from the ``[data]`` table."""
    data = dict(raw)
    assets = tuple(
        _build(AssetSpec, dict(a), "data.assets") for a in data.pop("assets", [])
    )
    macro = tuple(
        _build(MacroSeriesSpec, dict(m), "data.macro") for m in data.pop("macro", [])
    )
    if "splits" not in data:
        raise ConfigError("Missing config section [data.splits]")
    splits_raw = dict(data.pop("splits"))
    try:
        splits = SplitSettings(
            **{k: tuple(v) for k, v in splits_raw.items()}  # type: ignore[arg-type]
        )
    except TypeError as exc:
        raise ConfigError(f"Invalid fields in [data.splits]: {exc}") from exc

    features_raw = dict(data.pop("features", {}))
    pairs = features_raw.pop("correlation_pairs", {})
    for key in ("windows", "correlation_windows", "drawdown_windows",
                "regime_quantiles", "regime_labels"):
        if key in features_raw:
            features_raw[key] = tuple(features_raw[key])
    features_raw["correlation_pairs"] = {k: tuple(v) for k, v in pairs.items()}

    return _build(
        DataConfig,
        {
            **data,
            "assets": assets,
            "macro": macro,
            "splits": splits,
            "paths": _build(DataPaths, dict(data.get("paths", {})), "data.paths"),
            "yahoo": _build(YahooSettings, dict(data.get("yahoo", {})), "data.yahoo"),
            "fred": _build(FredSettings, dict(data.get("fred", {})), "data.fred"),
            "cleaning": _build(
                CleaningSettings, dict(data.get("cleaning", {})), "data.cleaning"
            ),
            "features": _build(FeatureSettings, features_raw, "data.features"),
        },
        "data",
    )


def risk_config_from_dict(raw: dict[str, Any]) -> RiskConfig:
    """Build a :class:`RiskConfig` from the ``[risk]`` table."""
    risk = dict(raw)
    limits = tuple(_build(RiskLimitSpec, dict(x), "risk.limits") for x in risk.pop("limits", []))
    scenarios = tuple(
        _build(DirectShockScenario, {**x, "shocks": dict(x.get("shocks", {}))}, "risk.stress_scenarios")
        for x in risk.pop("stress_scenarios", [])
    )
    replays = tuple(
        _build(HistoricalReplaySpec, dict(x), "risk.historical_replays")
        for x in risk.pop("historical_replays", [])
    )
    for key in ("confidence_levels", "volatility_windows", "report_windows"):
        if key in risk:
            risk[key] = tuple(risk[key])
    return _build(
        RiskConfig,
        {**risk, "limits": limits, "stress_scenarios": scenarios, "historical_replays": replays},
        "risk",
    )


def config_from_dict(raw: dict[str, Any]) -> ProjectConfig:
    """Build and validate a :class:`ProjectConfig` from a parsed TOML mapping.

    Args:
        raw: Mapping with one key per config section.

    Returns:
        A validated, immutable configuration object.

    Raises:
        ConfigError: If a section is missing, has unknown fields or fails validation.
    """
    portfolio = _section(raw, "portfolio")
    weights = portfolio.pop("weights", {})
    if not isinstance(weights, dict):
        raise ConfigError("[portfolio.weights] must be a table of ticker = weight")
    portfolio["tickers"] = tuple(weights)
    portfolio["weights"] = tuple(weights.values())
    training = _section(raw, "training")
    for key in ("net_arch", "seeds", "eval_seed_sets", "plausibility_thresholds"):
        if key in training:
            training[key] = tuple(training[key])
    simulation = _section(raw, "simulation")
    for key in ("factors", "candidate_factors", "state_columns", "dof_grid"):
        if key in simulation:
            simulation[key] = tuple(simulation[key])
    mitigation = _section(raw, "mitigation")
    mitigation["strategies"] = tuple(mitigation.get("strategies", ()))

    config = ProjectConfig(
        portfolio=_build(PortfolioConfig, portfolio, "portfolio"),
        data=data_config_from_dict(_section(raw, "data")),
        risk=risk_config_from_dict(_section(raw, "risk")),
        simulation=_build(SimulationConfig, simulation, "simulation"),
        environment=_build(EnvironmentConfig, _section(raw, "environment"), "environment"),
        training=_build(TrainingConfig, training, "training"),
        recommendation=_build(
            RecommendationConfig, _section(raw, "recommendation"), "recommendation"
        ),
        mitigation=_build(MitigationConfig, mitigation, "mitigation"),
        storage=_build(StorageConfig, dict(raw.get("storage", {})), "storage"),
    )
    validate_config(config)
    return config


def load_config(path: str | Path | None = None) -> ProjectConfig:
    """Load and validate a TOML configuration file.

    Args:
        path: Path to a TOML file. Defaults to ``config/default.toml``.

    Returns:
        A validated :class:`ProjectConfig`.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with config_path.open("rb") as fh:
        raw = tomllib.load(fh)
    return config_from_dict(raw)


def validate_config(config: ProjectConfig) -> None:
    """Check cross-field consistency of a configuration.

    Raises:
        ConfigError: Describing the first violated constraint.
    """
    p = config.portfolio
    if not p.tickers:
        raise ConfigError("[portfolio.weights] must not be empty")
    for w in p.weights:
        if isinstance(w, bool) or not isinstance(w, (int, float)):
            raise ConfigError(f"portfolio weights must be numbers, got {w!r}")
    if not (isinstance(p.initial_capital, (int, float)) and math.isfinite(p.initial_capital)
            and p.initial_capital > 0):
        raise ConfigError("portfolio.initial_capital must be a positive finite number")
    if p.policy not in PORTFOLIO_POLICIES:
        raise ConfigError(f"portfolio.policy must be one of {PORTFOLIO_POLICIES}")
    if p.rebalancing not in REBALANCING_POLICIES:
        raise ConfigError(f"portfolio.rebalancing must be one of {REBALANCING_POLICIES}")
    if p.missing_data_policy not in MISSING_DATA_POLICIES:
        raise ConfigError(f"portfolio.missing_data_policy must be one of {MISSING_DATA_POLICIES}")
    if not 0 < p.weight_tolerance < 1e-3:
        raise ConfigError("portfolio.weight_tolerance must be in (0, 1e-3)")
    if "{ticker}" not in p.return_column_template:
        raise ConfigError("portfolio.return_column_template must contain '{ticker}'")

    validate_data_config(config.data)
    validate_risk_config(config.risk, config.data)

    validate_simulation_config(config.simulation, config.environment, config.data)
    validate_training_config(config.training)

    rec = config.recommendation
    rec_total = rec.severity_weight + rec.plausibility_weight + rec.relevance_weight
    if not math.isclose(rec_total, 1.0, abs_tol=1e-6):
        raise ConfigError("recommendation weights must sum to 1")
    if rec.top_k < 1:
        raise ConfigError("recommendation.top_k must be >= 1")

    if not 0 < config.mitigation.max_turnover <= 1:
        raise ConfigError("mitigation.max_turnover must be in (0, 1]")


def _validate_windows(name: str, windows: tuple[int, ...]) -> None:
    if not windows:
        raise ConfigError(f"data.features.{name} must not be empty")
    if any(not isinstance(w, int) or isinstance(w, bool) or w < 2 for w in windows):
        raise ConfigError(f"data.features.{name} must contain integers >= 2, got {windows}")
    if len(set(windows)) != len(windows):
        raise ConfigError(f"data.features.{name} contains duplicates")


def validate_data_config(data: DataConfig) -> None:
    """Check dates, identifiers, windows and split boundaries of the ``[data]`` section.

    Raises:
        ConfigError: Describing the first violated constraint.
    """
    start = parse_iso_date(data.start_date, "data.start_date")
    end = parse_iso_date(data.end_date, "data.end_date")
    if start >= end:
        raise ConfigError("data.start_date must be before data.end_date")
    if data.interval != "1d":
        raise ConfigError("data.interval: only daily data ('1d') is supported")
    if data.primary_return not in {"simple", "log"}:
        raise ConfigError("data.primary_return must be 'simple' or 'log'")

    if not data.assets:
        raise ConfigError("data.assets must not be empty")
    aliases: set[str] = set()
    for a in data.assets:
        if not _SYMBOL_RE.match(a.symbol):
            raise ConfigError(f"Invalid Yahoo symbol {a.symbol!r} in data.assets")
        if not _ALIAS_RE.match(a.alias):
            raise ConfigError(f"Invalid alias {a.alias!r}: use upper-case letters/digits")
        if a.group not in ASSET_GROUPS:
            raise ConfigError(f"Asset {a.alias}: group must be one of {ASSET_GROUPS}")
        if a.alias in aliases:
            raise ConfigError(f"Duplicate alias {a.alias!r}")
        aliases.add(a.alias)
    if not data.enabled_assets("stock"):
        raise ConfigError("At least one enabled stock asset is required")
    for m in data.macro:
        if not _SERIES_ID_RE.match(m.series_id):
            raise ConfigError(f"Invalid FRED series id {m.series_id!r}")
        if not _ALIAS_RE.match(m.alias):
            raise ConfigError(f"Invalid alias {m.alias!r}: use upper-case letters/digits")
        if m.kind not in MACRO_KINDS:
            raise ConfigError(f"Macro series {m.series_id}: kind must be one of {MACRO_KINDS}")
        if m.alias in aliases:
            raise ConfigError(f"Duplicate alias {m.alias!r}")
        aliases.add(m.alias)

    f = data.features
    _validate_windows("windows", f.windows)
    _validate_windows("correlation_windows", f.correlation_windows)
    _validate_windows("drawdown_windows", f.drawdown_windows)
    if not 0 < f.min_obs_fraction <= 1:
        raise ConfigError("data.features.min_obs_fraction must be in (0, 1]")
    if f.annualization_days <= 0:
        raise ConfigError("data.features.annualization_days must be positive")
    for stock, indicators in f.correlation_pairs.items():
        for alias in (stock, *indicators):
            if alias not in aliases:
                raise ConfigError(f"correlation_pairs references unknown alias {alias!r}")
    q = f.regime_quantiles
    if list(q) != sorted(q) or any(not 0 < x < 1 for x in q):
        raise ConfigError("data.features.regime_quantiles must be increasing values in (0, 1)")
    if len(f.regime_labels) != len(q) + 1:
        raise ConfigError("data.features.regime_labels needs len(regime_quantiles) + 1 labels")

    c = data.cleaning
    if c.conflict_policy not in {"drop", "raise"}:
        raise ConfigError("data.cleaning.conflict_policy must be 'drop' or 'raise'")
    if c.max_gap_calendar_days < 1:
        raise ConfigError("data.cleaning.max_gap_calendar_days must be >= 1")
    if c.macro_ffill_limit < 0:
        raise ConfigError("data.cleaning.macro_ffill_limit must be >= 0")
    if c.yield_min_pct >= c.yield_max_pct:
        raise ConfigError("data.cleaning.yield_min_pct must be below yield_max_pct")

    previous_end: date | None = None
    for split_name, (s_raw, e_raw) in data.splits.as_dict().items():
        s = parse_iso_date(s_raw, f"data.splits.{split_name}[0]")
        e = parse_iso_date(e_raw, f"data.splits.{split_name}[1]")
        if s > e:
            raise ConfigError(f"data.splits.{split_name}: start is after end")
        if previous_end is not None and s <= previous_end:
            raise ConfigError(
                f"data.splits.{split_name} overlaps or precedes the previous split"
            )
        if s < start or e > end:
            raise ConfigError(f"data.splits.{split_name} lies outside the data date range")
        previous_end = e


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def validate_risk_config(risk: RiskConfig, data: DataConfig) -> None:
    """Check the ``[risk]`` section: confidences, windows, limits and scenarios.

    Raises:
        ConfigError: Describing the first violated constraint.
    """
    if not risk.confidence_levels:
        raise ConfigError("risk.confidence_levels must not be empty")
    for a in (*risk.confidence_levels, risk.default_confidence):
        if not (isinstance(a, float) and 0 < a < 1):
            raise ConfigError(f"risk confidence levels must satisfy 0 < alpha < 1, got {a!r}")
    if risk.default_confidence not in risk.confidence_levels:
        raise ConfigError("risk.default_confidence must be one of risk.confidence_levels")
    if not _is_int(risk.annualization_days) or risk.annualization_days <= 0:
        raise ConfigError("risk.annualization_days must be a positive integer")
    if any(not _is_int(w) or w < 2 for w in risk.volatility_windows):
        raise ConfigError("risk.volatility_windows must be integers >= 2")
    for name in ("min_observations", "min_tail_observations", "stress_horizon_days",
                 "extreme_loss_dates", "worst_windows"):
        value = getattr(risk, name)
        if not _is_int(value) or value < 1:
            raise ConfigError(f"risk.{name} must be a positive integer")
    windows = {*data.splits.as_dict(), "full_sample"}
    if risk.estimation_window not in windows:
        raise ConfigError(f"risk.estimation_window must be one of {sorted(windows)}")
    for w in risk.report_windows:
        if w not in windows:
            raise ConfigError(f"risk.report_windows: unknown window {w!r}")
    if not 0 <= risk.comparison_tolerance < 1e-6:
        raise ConfigError("risk.comparison_tolerance must be in [0, 1e-6)")

    names: set[str] = set()
    for lim in risk.limits:
        where = f"risk.limits[{lim.name}]"
        if lim.name in names:
            raise ConfigError(f"duplicate risk limit name {lim.name!r}")
        names.add(lim.name)
        if lim.metric not in RISK_METRIC_UNITS:
            raise ConfigError(f"{where}: metric must be one of {sorted(RISK_METRIC_UNITS)}")
        if lim.units != RISK_METRIC_UNITS[lim.metric]:
            raise ConfigError(
                f"{where}: units for {lim.metric} must be {RISK_METRIC_UNITS[lim.metric]!r}"
            )
        if not _is_int(lim.horizon_days) or lim.horizon_days < 1:
            raise ConfigError(f"{where}: horizon_days must be a positive integer")
        if not (isinstance(lim.threshold, (int, float)) and 0 < lim.threshold < 1):
            raise ConfigError(f"{where}: threshold must be a fraction in (0, 1)")
        if lim.breach_rule not in BREACH_RULES:
            raise ConfigError(f"{where}: breach_rule must be one of {BREACH_RULES}")
        if lim.metric.startswith("historical_"):
            if lim.confidence is None or not 0 < lim.confidence < 1:
                raise ConfigError(f"{where}: confidence in (0, 1) is required")
            if lim.horizon_days != 1:
                raise ConfigError(f"{where}: historical VaR/CVaR use daily returns (horizon_days = 1)")
        elif lim.confidence is not None:
            raise ConfigError(f"{where}: confidence applies only to historical VaR/CVaR")

    for sc in risk.stress_scenarios:
        if not _is_int(sc.horizon_days) or sc.horizon_days < 1:
            raise ConfigError(f"stress scenario {sc.name}: horizon_days must be a positive integer")
        if not sc.shocks:
            raise ConfigError(f"stress scenario {sc.name}: shocks must not be empty")
    for rp in risk.historical_replays:
        parse_iso_date(rp.start_date, f"historical replay {rp.name}.start_date")
        if not _is_int(rp.horizon_days) or rp.horizon_days < 1:
            raise ConfigError(f"historical replay {rp.name}: horizon_days must be a positive integer")


def validate_simulation_config(sim: SimulationConfig, env: EnvironmentConfig, data: DataConfig) -> None:
    """Check the ``[simulation]`` and ``[environment]`` sections.

    Raises:
        ConfigError: Describing the first violated constraint.
    """
    if not _is_int(sim.horizon_days) or sim.horizon_days < 1:
        raise ConfigError("simulation.horizon_days must be a positive integer")
    splits = set(data.splits.as_dict())
    for name in ("fit_window", "anchor_window"):
        if getattr(sim, name) not in splits:
            raise ConfigError(f"simulation.{name} must be one of {sorted(splits)}")
    if not sim.factors or len(set(sim.factors)) != len(sim.factors):
        raise ConfigError("simulation.factors must be a non-empty list without duplicates")
    if sim.market_factor not in sim.factors:
        raise ConfigError("simulation.market_factor must be one of simulation.factors")
    if set(sim.candidate_factors) & set(sim.factors):
        raise ConfigError("simulation.candidate_factors must not repeat model factors")
    positive = ("max_factor_sigma", "max_abs_factor_return", "max_abs_stock_return", "surprisal_cap",
                "em_tol")
    for name in positive:
        if not getattr(sim, name) > 0:
            raise ConfigError(f"simulation.{name} must be positive")
    if sim.max_abs_factor_return >= 1:
        raise ConfigError("simulation.max_abs_factor_return must be below 1 (100%)")
    if sim.residual_mode not in RESIDUAL_MODES:
        raise ConfigError(f"simulation.residual_mode must be one of {RESIDUAL_MODES}")
    if sim.shrinkage_method not in SHRINKAGE_METHODS:
        raise ConfigError(f"simulation.shrinkage_method must be one of {SHRINKAGE_METHODS}")
    if not 0 <= sim.fixed_shrinkage <= 1:
        raise ConfigError("simulation.fixed_shrinkage must be in [0, 1]")
    if not sim.dof_grid or any(not v > 2 for v in sim.dof_grid):
        raise ConfigError("simulation.dof_grid values must exceed 2 (finite covariance)")
    for name in ("max_residual_draws", "em_max_iter", "diagnostic_simulations"):
        if not _is_int(getattr(sim, name)) or getattr(sim, name) < 1:
            raise ConfigError(f"simulation.{name} must be a positive integer")
    if not 0 < sim.min_plausibility < 1:
        raise ConfigError("simulation.min_plausibility must be in (0, 1)")
    for name in ("breach_reward", "severity_weight", "plausibility_penalty", "magnitude_penalty",
                 "step_penalty", "invalid_penalty", "implausible_breach_penalty"):
        value = getattr(env, name)
        if not isinstance(value, (int, float)) or value < 0:
            raise ConfigError(f"environment.{name} must be a non-negative number")
    if not isinstance(env.implausible_breach_credit, (int, float)) or not 0.0 <= env.implausible_breach_credit < 1.0:
        raise ConfigError("environment.implausible_breach_credit must be in [0, 1) so a valid breach always pays more")


def validate_training_config(t: TrainingConfig) -> None:
    """Check the ``[training]`` section.

    Raises:
        ConfigError: Describing the first violated constraint.
    """
    if t.algorithm != "PPO":
        raise ConfigError("training.algorithm: only 'PPO' is implemented")
    ints = ("total_timesteps", "n_steps", "batch_size", "n_epochs", "torch_threads", "eval_freq",
            "checkpoint_freq", "selection_episodes", "validation_episodes", "test_episodes",
            "n_representative_scenarios")
    for name in ints:
        if not _is_int(getattr(t, name)) or getattr(t, name) < 1:
            raise ConfigError(f"training.{name} must be a positive integer")
    if t.n_steps % t.batch_size and t.n_steps < t.batch_size:
        raise ConfigError("training.batch_size must not exceed n_steps")
    for name in ("gamma", "gae_lambda"):
        if not 0 < getattr(t, name) <= 1:
            raise ConfigError(f"training.{name} must be in (0, 1]")
    if not (t.learning_rate > 0 and t.clip_range > 0 and t.vf_coef >= 0 and t.ent_coef >= 0):
        raise ConfigError("training: learning_rate and clip_range must be positive, coefficients >= 0")
    if not t.seeds or len(set(t.seeds)) != len(t.seeds):
        raise ConfigError("training.seeds must be non-empty and unique")
    if not t.eval_seed_sets or len(set(t.eval_seed_sets)) != len(t.eval_seed_sets):
        raise ConfigError("training.eval_seed_sets must be non-empty and unique")
    report = {(s, s + t.validation_episodes) for s in t.eval_seed_sets}
    blocks = sorted([(t.selection_seed, t.selection_seed + t.selection_episodes), *report,
                     (t.test_seed, t.test_seed + t.test_episodes)])
    if any(a[1] > b[0] for a, b in zip(blocks, blocks[1:])):
        raise ConfigError("training: selection, validation-report and test episode seed ranges must not overlap")
    if not t.plausibility_thresholds or any(not 0 < x < 1 for x in t.plausibility_thresholds):
        raise ConfigError("training.plausibility_thresholds must be shares in (0, 1)")
    if not -1 <= t.scripted_action <= 1:
        raise ConfigError("training.scripted_action must be in [-1, 1]")
