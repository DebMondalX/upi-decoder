"""Apply consistency rules to the teacher's labels before human review.

    python scripts/03b_fix_labels.py

The teacher is inconsistent on a few patterns that are mechanical to fix:
  * small ACH credits from listed companies are dividends  -> interest_income (+interest)
  * credits from your own phone-number UPI id at another bank are your own money moving
    (pass --self-vpa-pattern; default matches '<PHONE>@pz ... HDFC BANK') -> self_transfer
  * "Unknown Merchant" / "Unknown Person" -> "Unknown"

Writes data/interim/teacher_labels.fixed.jsonl and records what changed in each row's
"autofix" field; 04_review.py shows that in the 'check' column so you can confirm.
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from upi_decoder.io import read_jsonl, write_jsonl  # noqa: E402
from upi_decoder.schema import validate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="data/splits/train_pool.jsonl")
    ap.add_argument("--teacher", default="data/interim/teacher_labels.jsonl")
    ap.add_argument("--out", default="data/interim/teacher_labels.fixed.jsonl")
    ap.add_argument("--self-vpa-pattern", default=r"^UPI/<PHONE>@pz/.*HDFC BANK",
                    help="regex for credits that come from your own account at another bank")
    ap.add_argument("--dividend-max", type=float, default=5000)
    args = ap.parse_args()

    rows = {r["txn_id"]: r for r in read_jsonl(args.split)}
    self_rx = re.compile(args.self_vpa_pattern)
    out, changed = [], {}
    for t in read_jsonl(args.teacher):
        lab = t.get("label")
        r = rows.get(t["txn_id"])
        fix = None
        if lab and r:
            n, credit = r["narration"], r["direction"] == "credit"
            if credit and n.startswith("ACH/") and r["amount"] <= args.dividend_max and lab["category"] != "interest_income":
                lab = {**lab, "category": "interest_income", "counterparty_type": "merchant",
                       "flags": sorted(set(lab.get("flags", [])) | {"interest"})}
                fix = "dividend"
            elif credit and self_rx.search(n) and lab["category"] != "self_transfer":
                lab = {**lab, "merchant": "Self", "category": "self_transfer", "counterparty_type": "self", "flags": []}
                fix = "own account"
            elif lab["merchant"].lower().startswith("unknown") and lab["merchant"] != "Unknown":
                lab = {**lab, "merchant": "Unknown"}
                fix = "unknown name"
            assert validate(lab)[0] is not None
        if fix:
            changed[fix] = changed.get(fix, 0) + 1
        out.append({**t, "label": lab, "autofix": fix})
    write_jsonl(args.out, out)
    print(f"auto-fixed: {changed} -> {args.out}")


if __name__ == "__main__":
    main()
