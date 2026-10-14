"""Pre-label the training pool with a big open-weight teacher on Tinker.

    export TINKER_API_KEY=...
    python scripts/03_teacher_label.py                         # Qwen3.5-397B-A17B, few-shot, no thinking
    python scripts/03_teacher_label.py --thinking              # slower/pricier, usually better labels
    python scripts/03_teacher_label.py --limit 20              # try a handful first

Only scrubbed narrations are sent. Resumable: already-labelled txn_ids are skipped.
Labels the train pool only; test sets are hand-labelled from scratch (04_review.py --blank).
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from upi_decoder.io import append_jsonl, read_jsonl  # noqa: E402
from upi_decoder.llm import TinkerBackend  # noqa: E402
from upi_decoder.prompts import build_messages  # noqa: E402
from upi_decoder.schema import parse_model_output  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="data/splits/train_pool.jsonl")
    ap.add_argument("--out", default="data/interim/teacher_labels.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3.5-397B-A17B")
    ap.add_argument("--thinking", action="store_true", help="let the teacher reason before answering")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=32)
    args = ap.parse_args()

    rows = read_jsonl(args.split)
    done = {r["txn_id"] for r in read_jsonl(args.out) if r.get("label")}
    todo = [r for r in rows if r["txn_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(rows)} rows in split, {len(done)} already labelled, {len(todo)} to go")
    if not todo:
        return

    renderer = None
    if args.thinking:
        from tinker_cookbook import model_info

        renderer = model_info.get_recommended_renderer_names(args.model)[0]  # thinking variant
    teacher = TinkerBackend(
        args.model, renderer_name=renderer, max_tokens=4096 if args.thinking else 200, concurrency=args.concurrency
    )
    print(f"teacher: {args.model}  renderer: {teacher.renderer_name}")

    async def run():
        n_ok = n_in = n_out = 0
        for i in range(0, len(todo), 200):  # write in chunks so a crash loses little
            chunk = todo[i : i + 200]
            msgs = [build_messages(r["narration"], r["direction"], r["amount"], mode="few") for r in chunk]
            results = await teacher.generate_many(msgs)
            out = []
            for r, res in zip(chunk, results):
                label, errs = parse_model_output(res.text)
                n_ok += label is not None
                n_in += res.input_tokens
                n_out += res.output_tokens
                out.append({
                    "txn_id": r["txn_id"], "teacher": args.model, "raw": res.text,
                    "label": label.to_dict() if label else None, "errors": errs or ([res.error] if res.error else []),
                })
            append_jsonl(args.out, out)
        return n_ok, n_in, n_out

    n_ok, n_in, n_out = asyncio.run(run())
    print(f"valid labels: {n_ok}/{len(todo)}   tokens in/out: {n_in:,}/{n_out:,}")
    print(f"wrote {args.out}. Invalid ones are kept with label=null; re-run to retry them.")


if __name__ == "__main__":
    main()
