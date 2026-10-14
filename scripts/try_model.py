"""Eyeball a model on a few rows (no gold labels needed).

    python scripts/try_model.py --run runs/r0-teacher            # 12 random test_in rows
    python scripts/try_model.py --run runs/r0-teacher --split test_unseen --n 20
    python scripts/try_model.py --spec tinker:Qwen/Qwen3.5-4B --mode few   # base model, for contrast
"""

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from upi_decoder.io import read_jsonl  # noqa: E402
from upi_decoder.llm import make_backend  # noqa: E402
from upi_decoder.prompts import build_messages  # noqa: E402
from upi_decoder.schema import parse_model_output  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--spec", default=None)
    ap.add_argument("--mode", default=None, help="short (fine-tuned) | few | zero")
    ap.add_argument("--split", default="test_in")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    if args.run:
        run = Path(args.run)
        cfg = json.loads((run / "run_config.json").read_text())
        spec = f"tinker:{cfg['model']}@{(run / 'sampler_path.txt').read_text().strip()}"
        mode = args.mode or "short"
    else:
        spec, mode = args.spec, args.mode or "few"
    rows = read_jsonl(f"data/splits/{args.split}.jsonl")
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.n]

    backend = make_backend(spec)
    res = asyncio.run(backend.generate_many(
        [build_messages(r["narration"], r["direction"], r["amount"], mode=mode) for r in rows], progress=False))
    for r, g in zip(rows, res):
        lab, _ = parse_model_output(g.text)
        out = f"{lab.merchant[:20]:20} {lab.category:16} {lab.counterparty_type:9} {','.join(lab.flags)}" if lab else f"INVALID: {g.text[:60]!r}"
        print(f"{r['direction'][:2]} {r['amount']:>10.2f} | {r['narration'][:58]:58} => {out}  ({g.latency_s * 1000:.0f} ms)")


if __name__ == "__main__":
    main()
