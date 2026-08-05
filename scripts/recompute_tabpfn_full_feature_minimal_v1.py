from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_tabpfn_full_feature_minimal_v1 import ROOT, validate_formal_readback


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay the locked full-feature TabPFN metric and coverage checks."
    )
    parser.add_argument(
        "--formal-dir",
        default=str(ROOT / "experiments/tabpfn_full_feature_minimal_v1/formal"),
    )
    args = parser.parse_args()
    formal_dir = Path(args.formal_dir).resolve()
    if not formal_dir.is_dir():
        raise FileNotFoundError(f"Formal result directory missing: {formal_dir}")
    result = validate_formal_readback(formal_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
