"""Pre-compute the fine-tuned model's answers for the synthetic demo statement.

    python scripts/make_demo_cache.py --run runs/r1          # via Tinker (costs ~1 cent)
    python scripts/make_demo_cache.py --spec ollama:upi-decoder

Writes app/demo_cache.json. The hosted demo (Render) has no GPU and must never see real
statements, so it shows these cached model outputs for the synthetic statement only.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from upi_decoder.llm import make_backend  # noqa: E402
from upi_decoder.prompts import build_messages  # noqa: E402
from upi_decoder.schema import parse_model_output  # noqa: E402
from upi_decoder.synthetic import demo_statement_records  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--spec", default=None)
    ap.add_argument("--out", default="app/demo_cache.json")
    args = ap.parse_args()
    if args.run:
        run = Path(args.run)
        cfg = json.loads((run / "run_config.json").read_text())
        spec = f"tinker:{cfg['model']}@{(run / 'sampler_path.txt').read_text().strip()}"
    else:
        spec = args.spec

    uniq = {}
    for r in demo_statement_records():
        uniq.setdefault(r["key"], r)
    backend = make_backend(spec)
    res = asyncio.run(backend.generate_many(
        [build_messages(r["narration"], r["direction"], r["amount"], mode="short") for r in uniq.values()]))
    cache, bad = {}, 0
    for k, g in zip(uniq, res):
        lab = parse_model_output(g.text)[0]
        if lab:
            cache[k] = lab.to_dict()
        else:
            bad += 1
    Path(args.out).write_text(json.dumps({"model": spec.split("@")[0], "labels": cache}, indent=1))
    print(f"cached {len(cache)} distinct narrations ({bad} invalid) -> {args.out}")


if __name__ == "__main__":
    main()
