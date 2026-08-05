from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_tabpfn_baseline_v1 import ROOT, validate_formal_readback


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone replay of the locked TabPFN metric and integrity checks."
    )
    parser.add_argument(
        "--formal-dir",
        default=str(ROOT / "experiments/tabpfn_baseline_v1/formal"),
    )
    args = parser.parse_args()
    formal_dir = Path(args.formal_dir).resolve()
    if not formal_dir.is_dir():
        raise FileNotFoundError(f"Formal result directory missing: {formal_dir}")
    result = validate_formal_readback(formal_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
