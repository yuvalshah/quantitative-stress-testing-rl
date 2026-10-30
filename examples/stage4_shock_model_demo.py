"""Stage 4 demonstration: fit the shock model on the training split, validate it, run scripted episodes.

Usage::

    python examples/stage4_shock_model_demo.py              # writes results/stage4/
    python examples/stage4_shock_model_demo.py --overwrite

Requires the Stage 2 output ``data/processed/features/consolidated_daily.csv``.
No network access, no RL training.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from rlstress.config import PROJECT_ROOT, ProjectConfig, load_config, resolve_path
from rlstress.data.storage import read_frame
from rlstress.shock_report import run_stage4


def run_demo(config: ProjectConfig, returns_path: Path, output_dir: Path, overwrite: bool = False,
             baseline_episodes: int = 200) -> dict[str, Any]:
    """Load processed data and run the Stage 4 fit and validation."""
    if not returns_path.exists():
        raise FileNotFoundError(f"{returns_path} not found; run the Stage 2 pipeline first")
    return run_stage4(config, read_frame(returns_path), output_dir, str(returns_path), overwrite,
                      baseline_episodes)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--returns", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config(args.config)
    returns_path = args.returns or resolve_path(config.risk.returns_path, PROJECT_ROOT)
    output_dir = args.output_dir or resolve_path(config.simulation.output_dir, PROJECT_ROOT)
    out = run_demo(config, returns_path, output_dir, args.overwrite)
    m = out["model"]
    print(f"Fitted on {m.fit_start} to {m.fit_end} ({m.n_obs} days): nu joint {m.nu}, nu plausibility {m.plaus_nu}")
    for e in out["episodes"]["scripted_episodes"]:
        print(f"  {e['episode']:<42} steps {e['steps']}  loss {e['cumulative_loss']:+.2%}  "
              f"DD {e['path_drawdown']:.2%}  breach {e['breached']}  valid {e['valid_breach']}  "
              f"reward {e['total_reward']:.2f}")
    b = out["episodes"]["random_policy_baseline"]
    print(f"  random policy: breach {b['breach_rate']:.1%}, valid {b['valid_breach_rate']:.1%}")
    print(f"Outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
