"""Command line: `python -m fraudlab <stage>`.

Stages, in pipeline order:
  eda       data audit, train-vs-test shift table, README figures
  select    cross-validate the model zoo (6 learners + 4 blends) and pick the final model
  train     fit the selected model on all 20,000 training rows
  predict   score the test file and write results/final/submission.csv
  verify    check the rebuilt model and submission against the shipped model and the 0.19586 upload
  all       every stage above, in order
"""
import argparse
import sys
import time
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m fraudlab", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["eda", "select", "train", "predict", "verify", "all"])
    p.add_argument("--test", type=Path, help="predict: a different test CSV (same columns as the competition file)")
    p.add_argument("--out", type=Path, help="predict: output CSV path")
    p.add_argument("--skip", nargs="*", default=[], help="select: base learners to leave out, e.g. --skip random_forest knn")
    args = p.parse_args(argv)

    from . import eda, select, train
    stages = {
        "eda": eda.run,
        "select": lambda: select.run(skip=args.skip),
        "train": train.train,
        "predict": lambda: train.predict(**{k: v for k, v in
                                            {"test_path": args.test, "out_path": args.out}.items() if v}),
        "verify": train.verify,
    }
    order = list(stages) if args.stage == "all" else [args.stage]
    ok = True
    for name in order:
        t0 = time.time()
        print(f"\n== {name} " + "=" * (60 - len(name)))
        try:
            result = stages[name]()
        except FileNotFoundError as e:
            print(f"\nERROR: {e}", file=sys.stderr)
            return 2
        ok &= result is not False
        print(f"== {name} done in {time.time() - t0:.0f}s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
