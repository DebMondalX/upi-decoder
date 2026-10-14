"""Turn verified labels into chat-format training data for Tinker.

    python scripts/05_build_sft.py                          # real labels only
    python scripts/05_build_sft.py --synthetic-ratio 0.3    # top up with synthetic rows (max 50%)

Output: data/sft/train.jsonl, one {"messages": [system, user, assistant]} per line,
the format the cookbook's FromConversationFileBuilder reads. The system prompt is
the SHORT one: the fine-tuned model learns the instructions instead of reading them.
"""

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from upi_decoder.io import read_jsonl, write_jsonl  # noqa: E402
from upi_decoder.prompts import sft_example  # noqa: E402
from upi_decoder.schema import validate  # noqa: E402


def synthetic_rows(n, seed):
    """Fresh synthetic rows (different seed from the demo set), scrubbed like real data."""
    from datetime import date

    from upi_decoder import scrub, synthetic

    s = scrub.Scrubber()
    rng = random.Random(seed)
    rows = []
    per_bank = n // len(synthetic.LAYOUTS) + 1
    for bank in synthetic.LAYOUTS:
        for r in synthetic.generate_bank(bank, per_bank, date(2025, 1, 1), date(2026, 9, 30), seed=1000 + seed):
            r["narration"] = s.scrub(r["narration"])
            rows.append(r)
    rng.shuffle(rows)
    return rows[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/labeled/train_pool.jsonl")
    ap.add_argument("--out", default="data/sft/train.jsonl")
    ap.add_argument("--synthetic-ratio", type=float, default=0.0, help="synthetic share of the final set (0-0.5)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rows = read_jsonl(args.labels)
    if not rows:
        sys.exit(f"{args.labels} is empty; run 04_review.py import first")
    ratio = min(max(args.synthetic_ratio, 0.0), 0.5)
    n_syn = int(len(rows) * ratio / (1 - ratio)) if ratio else 0
    if n_syn:
        rows = rows + synthetic_rows(n_syn, args.seed)

    examples, cats = [], Counter()
    for r in rows:
        lab, errs = validate(r["label"])
        if lab is None:
            continue
        examples.append(sft_example(r["narration"], r["direction"], r["amount"], lab))
        cats[lab.category] += 1
    random.Random(args.seed).shuffle(examples)
    write_jsonl(args.out, examples)

    print(f"wrote {len(examples)} examples ({n_syn} synthetic) -> {args.out}")
    print("category balance:")
    for c, k in cats.most_common():
        print(f"  {c:16} {k:5}")
    rare = [c for c, k in cats.items() if k < 15]
    if rare:
        print(f"few examples for: {rare}. Consider --synthetic-ratio or labelling more of these.")


if __name__ == "__main__":
    main()
