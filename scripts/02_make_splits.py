"""Make honest train / test splits.

    python scripts/02_make_splits.py --unseen-bank axis --test-from 2026-08

  test_unseen : rows from one bank never seen in training      (does it generalise to new formats?)
  test_in     : rows from the other banks in later months       (does it work going forward?)
  train_pool  : everything else, near-duplicates capped         (what gets teacher-labelled)

Near-duplicates = same narration once refs/ids are blanked (rules.dedup_key).
Tests keep one row per key, so 200 Swiggy lines can't inflate accuracy.
"""

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from upi_decoder.io import read_jsonl, write_jsonl  # noqa: E402


def one_per_key(rows, rng):
    by = defaultdict(list)
    for r in rows:
        by[r["dedup_key"]].append(r)
    return [rng.choice(v) for v in by.values()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/interim/transactions.jsonl")
    ap.add_argument("--out-dir", default="data/splits")
    ap.add_argument("--unseen-bank", default=None, help="bank held out entirely (folder name under data/raw)")
    ap.add_argument("--test-from", default=None, help="YYYY-MM; rows from this month on go to test_in (default: last 2 months)")
    ap.add_argument("--test-size", type=int, default=300)
    ap.add_argument("--unseen-size", type=int, default=150)
    ap.add_argument("--cap", type=int, default=3, help="max training rows per dedup key")
    ap.add_argument("--strict", action="store_true", help="drop test rows whose dedup key also occurs in train")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    rows = read_jsonl(args.inp)
    if not rows:
        sys.exit(f"{args.inp} is empty; run 01_ingest_scrub.py first")

    unseen = [r for r in rows if args.unseen_bank and r["bank"] == args.unseen_bank.lower()]
    seen = [r for r in rows if not (args.unseen_bank and r["bank"] == args.unseen_bank.lower())]
    if args.unseen_bank and not unseen:
        sys.exit(f"no rows for bank '{args.unseen_bank}'. banks: {sorted({r['bank'] for r in rows})}")

    months = sorted({r["date"][:7] for r in seen})
    cutoff = args.test_from or (months[-2] if len(months) >= 4 else months[-1])
    later = [r for r in seen if r["date"][:7] >= cutoff]
    earlier = [r for r in seen if r["date"][:7] < cutoff]

    # train pool: cap near-duplicates
    rng.shuffle(earlier)
    per_key: Counter[str] = Counter()
    train = []
    for r in earlier:
        if per_key[r["dedup_key"]] < args.cap:
            per_key[r["dedup_key"]] += 1
            train.append(r)
    train_keys = set(per_key)

    test_in = one_per_key(later, rng)
    test_unseen = one_per_key(unseen, rng)
    if args.strict:
        test_in = [r for r in test_in if r["dedup_key"] not in train_keys]
        test_unseen = [r for r in test_unseen if r["dedup_key"] not in train_keys]
    rng.shuffle(test_in)
    rng.shuffle(test_unseen)
    test_in, test_unseen = test_in[: args.test_size], test_unseen[: args.unseen_size]

    out = Path(args.out_dir)
    write_jsonl(out / "train_pool.jsonl", sorted(train, key=lambda r: r["dedup_key"]))
    write_jsonl(out / "test_in.jsonl", test_in)
    write_jsonl(out / "test_unseen.jsonl", test_unseen)

    overlap = sum(r["dedup_key"] in train_keys for r in test_in) / max(1, len(test_in))
    print(f"cutoff month for test_in: {cutoff}")
    print(f"train_pool : {len(train):5}  (from {len(earlier)} rows, cap {args.cap}/key)")
    print(f"test_in    : {len(test_in):5}  (one per key; {overlap:.0%} of keys also seen in train)")
    print(f"test_unseen: {len(test_unseen):5}  (bank: {args.unseen_bank or '-'})")
    if len(test_in) < 150:
        print("note: test_in is small; use --test-from an earlier month or add more statements")


if __name__ == "__main__":
    main()
