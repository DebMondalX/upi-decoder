"""The comparison table.

    # everything (needs TINKER_API_KEY):
    python scripts/07_eval.py --run runs/r1
    # cheap trial on 30 rows per split:
    python scripts/07_eval.py --run runs/r1 --limit 30
    # add the local exported model and/or a closed API:
    python scripts/07_eval.py --run runs/r1 --local upi-decoder --openai gpt-4.1-mini
    # offline (rules baseline only), e.g. the synthetic dry run:
    python scripts/07_eval.py --only-rules

Systems:
  rules         regex/keyword parser (what most expense apps do)
  student-few   base Qwen3.5-4B, full prompt + few-shot
  teacher-few   Qwen3.5-397B-A17B, full prompt + few-shot
  finetuned     Qwen3.5-4B + your Tinker LoRA, one-line prompt
  local         the exported model running in Ollama on this machine (laptop latency)
  openai        optional closed model via any OpenAI-compatible API

Writes results/results.md (paste into the post), results/results.json,
results/confusion_<split>.csv, and per-row predictions (git-ignored).
"""

import argparse
import asyncio
import csv
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from upi_decoder.io import read_jsonl, write_jsonl  # noqa: E402
from upi_decoder.llm import PRICES, make_backend  # noqa: E402
from upi_decoder.metrics import confusion, score  # noqa: E402
from upi_decoder.prompts import build_messages  # noqa: E402
from upi_decoder.rules import rules_predict  # noqa: E402
from upi_decoder.schema import CATEGORIES, parse_model_output, validate  # noqa: E402


def run_rules(rows):
    preds, lat = [], []
    for r in rows:
        t0 = time.perf_counter()
        preds.append(rules_predict(r["narration"], r["direction"], r["amount"]))
        lat.append(time.perf_counter() - t0)
    return preds, {"latency_p50_s": statistics.median(lat), "in_tok": 0, "out_tok": 0, "raw": [p.to_json() for p in preds]}


def run_model(spec, mode, rows):
    backend = make_backend(spec)
    msgs = [build_messages(r["narration"], r["direction"], r["amount"], mode=mode) for r in rows]
    results = asyncio.run(backend.generate_many(msgs))
    preds = [parse_model_output(res.text)[0] for res in results]
    lats = [res.latency_s for res in results if not res.error]
    errs = [res.error for res in results if res.error]
    if errs:
        print(f"  {len(errs)} request errors, e.g. {errs[0]}")
    return preds, {
        "latency_p50_s": statistics.median(lats) if lats else float("nan"),
        "in_tok": statistics.mean(r.input_tokens for r in results),
        "out_tok": statistics.mean(r.output_tokens for r in results),
        "raw": [res.text for res in results],
        "backend": backend.name,
    }


def cost_per_1k(name, in_tok, out_tok):
    p = PRICES.get(name)
    if not p:
        return None
    return 1000 * (in_tok * p[0] + out_tok * p[1]) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["test_in", "test_unseen"])
    ap.add_argument("--run", default=None, help="runs/<name> from 06_train.py (finds the fine-tuned checkpoint)")
    ap.add_argument("--finetuned", default=None, help="BASE_MODEL@tinker://... (instead of --run)")
    ap.add_argument("--student", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--teacher", default="Qwen/Qwen3.5-397B-A17B")
    ap.add_argument("--local", default=None, help="Ollama model name of the exported fine-tune")
    ap.add_argument("--openai", default=None, help="model name on an OpenAI-compatible API")
    ap.add_argument("--skip", nargs="*", default=[], help="system names to skip, e.g. teacher-few")
    ap.add_argument("--only-rules", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    finetuned = args.finetuned
    if args.run and not finetuned:
        run = Path(args.run)
        cfg = json.loads((run / "run_config.json").read_text())
        finetuned = f"{cfg['model']}@{(run / 'sampler_path.txt').read_text().strip()}"

    systems = [("rules", None, None)]
    if not args.only_rules:
        systems += [("student-few", f"tinker:{args.student}", "few"), ("teacher-few", f"tinker:{args.teacher}", "few")]
        if finetuned:
            systems.append(("finetuned", f"tinker:{finetuned}", "short"))
        if args.local:
            systems.append(("local", f"ollama:{args.local}", "short"))
        if args.openai:
            systems.append(("openai", f"openai:{args.openai}", "few"))
    systems = [s for s in systems if s[0] not in args.skip]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    all_results, md = {}, ["# Results", ""]
    for split in args.splits:
        rows = read_jsonl(f"data/labeled/{split}.jsonl")
        if not rows:
            print(f"skip {split}: data/labeled/{split}.jsonl missing (label it with 04_review.py)")
            continue
        if args.limit:
            rows = rows[: args.limit]
        golds = [validate(r["label"])[0] for r in rows]
        print(f"\n== {split} ({len(rows)} rows) ==")
        md += [f"## {split} (n={len(rows)})", "",
               "| system | category acc | macro-F1 | merchant exact | self-transfer F1 | refund F1 | valid JSON | input tok/row | p50 latency | $/1k rows |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        split_res = {}
        for name, spec, mode in systems:
            print(f"- {name}")
            if spec is None:
                preds, info = run_rules(rows)
                info["backend"] = "rules"
            else:
                preds, info = run_model(spec, mode, rows)
            m = score(golds, preds)
            cost = cost_per_1k(info["backend"], info["in_tok"], info["out_tok"])
            split_res[name] = {**{k: v for k, v in m.items()}, **{k: v for k, v in info.items() if k != "raw"}, "cost_per_1k": cost}
            lat = info["latency_p50_s"]
            md.append(
                f"| {name} | {m['category_acc']:.1%} | {m['category_macro_f1']:.3f} | {m['merchant_exact']:.1%} | "
                f"{m['self_transfer_f1']:.3f} | {m['refund_f1']:.3f} | {m['valid_json']:.1%} | {info['in_tok']:.0f} | "
                f"{lat * 1000:.0f} ms | {'-' if cost is None else f'${cost:.3f}'} |"
            )
            print(f"  cat acc {m['category_acc']:.1%}  macro-F1 {m['category_macro_f1']:.3f}  merchant {m['merchant_exact']:.1%}  valid {m['valid_json']:.0%}")
            write_jsonl(out / f"predictions_{split}_{name}.jsonl", [
                {"txn_id": r["txn_id"], "narration": r["narration"], "gold": r["label"], "pred": p.to_dict() if p else None, "raw": raw}
                for r, p, raw in zip(rows, preds, info["raw"])
            ])
            # confusion matrix for the system we care about most
            if name == ("finetuned" if finetuned else systems[-1][0]):
                conf = confusion(golds, preds)
                labels = list(CATEGORIES) + ["<invalid>"]
                with open(out / f"confusion_{split}.csv", "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["gold \\ pred"] + labels)
                    for g in CATEGORIES:
                        w.writerow([g] + [conf.get((g, p), 0) for p in labels])
        all_results[split] = split_res
        md.append("")

    md += ["Latency for Tinker systems is remote API latency; `local` is on-device.",
           "Cost uses Tinker sampling prices; the local model costs nothing per row."]
    (out / "results.md").write_text("\n".join(md))
    (out / "results.json").write_text(json.dumps(all_results, indent=2, default=str))
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
