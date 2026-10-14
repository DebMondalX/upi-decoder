"""Parse every statement export and scrub PII. Everything stays on your machine.

    python scripts/01_ingest_scrub.py --self-name "YOUR NAME" --self-name "SURNAME FIRSTNAME" --auto-people

Input layout:  data/raw/<bank>/*.csv|xls|xlsx   (folder name = bank, e.g. hdfc, icici, sbi, axis)
Output:        data/interim/transactions.jsonl  (scrubbed narrations only; raw text is not stored)
               data/interim/scrub_report.txt     (read this! leftover names = add to people.txt)
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from upi_decoder import ingest, rules, scrub  # noqa: E402
from upi_decoder.io import write_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--people", default="data/people.txt", help="one name per line (you, family, friends, landlord)")
    ap.add_argument("--self-name", action="append", default=[], help="your own name as banks print it (repeatable)")
    ap.add_argument("--auto-people", action="store_true",
                    help="also scrub person-like UPI names found automatically (written to data/people.suggested.txt)")
    ap.add_argument("--out", default="data/interim/transactions.jsonl")
    args = ap.parse_args()

    records = ingest.ingest_dir(args.raw)
    if not records:
        sys.exit(f"No statements found under {args.raw}/<bank>/. See data/README.md.")

    people = scrub.load_people(args.people) + args.self_name
    if args.auto_people:
        sugg = scrub.suggest_people([r["narration"] for r in records])
        sp = Path(args.people).with_name("people.suggested.txt")
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text("# auto-suggested; move real people into people.txt, delete businesses\n" + "\n".join(sugg) + "\n")
        people = sorted(set(people) | set(sugg), key=len, reverse=True)
        print(f"auto-people: {len(sugg)} person-like names -> {sp}")
    scrubber = scrub.Scrubber(people=people, self_names=args.self_name)
    for r in records:
        r["narration"] = scrubber.scrub(r["narration"])
        r["mode"] = rules.payment_mode(r["narration"])
        r["dedup_key"] = rules.dedup_key(r["narration"])
    n = write_jsonl(args.out, records)

    banks = Counter(r["bank"] for r in records)
    months = sorted({r["date"][:7] for r in records})
    report = [
        f"transactions: {n}",
        f"unique (dedup) narrations: {len({r['dedup_key'] for r in records})}",
        f"by bank: {dict(banks)}",
        f"months: {months[0]} .. {months[-1]} ({len(months)})",
        f"people names loaded for scrubbing: {len(people)}",
        "",
        "== Most common ALL-CAPS word pairs (look for personal names) ==",
        *[f"{c:5}  {pair}" for pair, c in scrub.leak_report([r["narration"] for r in records])],
        "",
        "== Most common VPAs kept (look for personal ids) ==",
        *[f"{c:5}  {v}" for v, c in scrub.vpa_report([r["narration"] for r in records])],
    ]
    rp = Path(args.out).with_name("scrub_report.txt")
    rp.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report[:5]))
    print(f"\nwrote {args.out}\nreview {rp} for leftover names before step 03 (which sends narrations to Tinker)")


if __name__ == "__main__":
    main()
