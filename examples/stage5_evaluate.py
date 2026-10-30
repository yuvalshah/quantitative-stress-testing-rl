"""Stage 5 evaluation: validation (default) or the explicit, one-time final test.

Usage::

    python examples/stage5_evaluate.py --run-dir RUN                         # validation + selection + report
    python examples/stage5_evaluate.py --run-dir RUN --final-test --confirm-final-test   # test split, once

The final test uses the configuration selected on validation and refuses to
run a second time for the same run.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from rlstress.rl.experiment import final_test, validate_run


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--final-test", action="store_true")
    p.add_argument("--confirm-final-test", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.final_test:
        out = final_test(args.run_dir, confirm=args.confirm_final_test)
        print(json.dumps({k: v["summary"]["valid_breach_rate"] for k, v in out["results"].items()}, indent=2))
        print(f"Test results: {args.run_dir / 'test' / 'test_evaluation.json'}")
    else:
        summary = validate_run(args.run_dir)
        print(f"Selected threshold {summary['selection']['threshold']['selected']}, "
              f"seed {summary['selection']['selected_seed']}")
        print(f"Report: {args.run_dir / 'stage5_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
