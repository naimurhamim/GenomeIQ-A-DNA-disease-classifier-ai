"""Command-line interface for running predictions.

Usage examples:
    python -m src.predict --sequence ATGGAT...
    python -m src.predict --file path/to/sequence.fasta
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .api import service


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict disease from DNA sequence(s).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sequence", "-s", help="Raw DNA sequence to classify.")
    group.add_argument(
        "--file", "-f", help="Path to a FASTA or plain-text file with sequences."
    )
    parser.add_argument(
        "--explain/--no-explain",
        dest="explain",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--ood/--no-ood",
        dest="ood",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--similar/--no-similar",
        dest="similar",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.sequence:
        payload = service.predict_payload(
            args.sequence,
            include_explain=args.explain,
            include_ood=args.ood,
            include_similar=args.similar,
            top_k_similar=args.top_k,
        )
    else:
        text = Path(args.file).read_text(encoding="utf-8", errors="ignore")
        payload = service.batch_payload(text)

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
