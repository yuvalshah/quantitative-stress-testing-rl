"""Stage 3 demonstration: run the risk engine on the Stage 2 processed dataset.

Usage::

    python examples/stage3_risk_engine_demo.py              # writes results/stage3/
    python examples/stage3_risk_engine_demo.py --overwrite  # replace earlier outputs

Requires the Stage 2 output ``data/processed/features/consolidated_daily.csv``
(run ``python -m rlstress.data.pipeline`` first). No network access.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from rlstress.config import PROJECT_ROOT, ProjectConfig, load_config, resolve_path
from rlstress.data.storage import read_frame
from rlstress.risk_report import run_risk_engine


def run_demo(
    config: ProjectConfig, returns_path: Path, output_dir: Path, overwrite: bool = False
) -> dict[str, Any]:
    """Load the processed returns and run the engine."""
    if not returns_path.exists():
        raise FileNotFoundError(f"{returns_path} not found; run the Stage 2 pipeline first")
    source = read_frame(returns_path)
    return run_risk_engine(config, source, output_dir, str(returns_path), overwrite)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--returns", type=Path, default=None, help="Stage 2 consolidated_daily.csv")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config(args.config)
    returns_path = args.returns or resolve_path(config.risk.returns_path, PROJECT_ROOT)
    output_dir = args.output_dir or resolve_path(config.risk.output_dir, PROJECT_ROOT)
    summary = run_demo(config, returns_path, output_dir, args.overwrite)
    est = summary["metrics"][config.risk.estimation_window]
    print(f"Estimation window {est['window']}: {est['data']['start_date']} to {est['data']['end_date']} "
          f"({est['volatility']['n_observations']} obs)")
    print(f"  annualised vol {est['volatility']['annualized']:.2%}")
    for v, c in zip(est["var"], est["cvar"]):
        print(f"  VaR {v['confidence']:.0%}: {v['var_pct']:.2%} (${v['var_usd']:,.0f})   "
              f"CVaR: {c['cvar_pct']:.2%} (${c['cvar_usd']:,.0f})")
    print(f"  max drawdown {est['max_drawdown']['max_drawdown']:.2%}")
    for s in summary["stress"]:
        print(f"  stress {s.scenario_name}: loss {s.loss_pct:.2%} (${s.loss_usd:,.0f}), breached={s.breached}")
    print(f"Outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
