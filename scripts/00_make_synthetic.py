"""Generate synthetic bank exports + gold labels.

    python scripts/00_make_synthetic.py --out data/synthetic --rows 400

Writes data/synthetic/raw/<bank>/<bank>.csv (same layouts as real exports) and
data/synthetic/gold.jsonl keyed by txn_id. Use it to dry-run the whole pipeline
before touching real statements, for tests, and for the public demo.
"""

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from upi_decoder import ingest, synthetic  # noqa: E402
from upi_decoder.io import write_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--rows", type=int, default=400, help="rows per bank")
    ap.add_argument("--start", default="2025-10-01")
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    gold = []
    for bank in synthetic.LAYOUTS:
        rows = synthetic.generate_bank(bank, args.rows, date.fromisoformat(args.start), date.fromisoformat(args.end), args.seed)
        path = out / "raw" / bank / f"{bank}_statement.csv"
        synthetic.write_bank_csv(rows, path)
        parsed = ingest.parse_statement(path, bank)  # round-trip to get the same txn_ids ingest will produce
        assert len(parsed) == len(rows), f"{bank}: parsed {len(parsed)} of {len(rows)}"
        for p, r in zip(parsed, rows):
            gold.append({"txn_id": p["txn_id"], "label": r["label"]})
        print(f"{bank:6} {len(rows)} rows -> {path}")
    write_jsonl(out / "gold.jsonl", gold)
    print(f"gold labels: {len(gold)} -> {out / 'gold.jsonl'}")


if __name__ == "__main__":
    main()
