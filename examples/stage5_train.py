"""Stage 5 training: create a run and train PPO for every (threshold, seed).

Usage::

    python examples/stage5_train.py --jobs 9                    # new run, all thresholds x seeds
    python examples/stage5_train.py --run-dir RUN --threshold 0.01 --seed 0   # one job in an existing run
    python examples/stage5_train.py --run-dir RUN --resume PATH --seed 0 --threshold 0.01 --timesteps 50000
    python examples/stage5_train.py --jobs 3 --label ent001 --override training.ent_coef=0.01

Requires ``pip install -e ".[rl]"`` and the Stage 2 processed dataset.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from rlstress.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config, resolve_path
from rlstress.rl.experiment import init_run, load_run, train_all, train_one
from rlstress.rl.training import train_ppo


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    p.add_argument("--returns", type=Path, default=None)
    p.add_argument("--run-dir", type=Path, default=None, help="existing run (default: create a new one)")
    p.add_argument("--label", default="")
    p.add_argument("--out-root", type=Path, default=None, help="default: training.output_dir")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--resume", type=Path, default=None, help="checkpoint .zip to continue from")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--override", action="append", default=[], metavar="SECTION.FIELD=JSON",
                   help='e.g. --override training.ent_coef=0.01 --override "training.plausibility_thresholds=[0.01]"')
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config(args.config)
    run_dir = args.run_dir
    if run_dir is None:
        returns = args.returns or resolve_path(config.risk.returns_path, PROJECT_ROOT)
        out_root = args.out_root or resolve_path(config.training.output_dir, PROJECT_ROOT)
        overrides = {k: json.loads(v) for k, v in (o.split("=", 1) for o in args.override)}
        run_dir = init_run(args.config, returns, out_root, args.label, overrides)
        print(f"Created {run_dir}")
    if args.resume is not None:
        cfg, source, model, _ = load_run(run_dir)
        out = run_dir / "training" / f"thr_{args.threshold:g}" / f"seed_{args.seed}_resumed"
        res = train_ppo(cfg, source, model, args.threshold, args.seed, out, args.timesteps, resume_from=args.resume)
        print(f"Resumed to {res.timesteps} timesteps in {out}")
    elif args.threshold is not None and args.seed is not None:
        print(train_one(run_dir, args.threshold, args.seed, args.timesteps))
    else:
        for r in train_all(run_dir, args.jobs, args.timesteps):
            print(f"threshold {r['threshold']:g} seed {r['seed']}: {r['seconds']:.0f}s, best {r['best']}")
    print(f"Next: python examples/stage5_evaluate.py --run-dir {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
