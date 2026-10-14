"""Scanned (image-only) PDF statements -> common records, via Tesseract OCR.

Some banks (HSBC India) only offer PDF statements, and those PDFs have no text
layer: every page is pictures of glyphs. We render each page, OCR it with word
positions, and rebuild rows from the table's column layout:

    Date | Transaction Details (multi-line) | Deposits | Withdrawals | Balance

A transaction ends on the line that carries a balance; the detail lines since the
previous balance belong to it. OCR mistakes in amounts are caught by checking
every row against the running balance (prev ± amount == balance).

Requires the `pdftoppm` (poppler) and `tesseract` binaries.
    macOS:  brew install poppler tesseract
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

_DATE = re.compile(r"^([0-9OoIl]{2})([A-Za-z0-9|]{3})(\d{4})$")
_MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")


def _fix_date(tok: str) -> str | None:
    """'27Ju12026' -> '2026-07-27' (OCR swaps l/1, O/0 in the month and day)."""
    m = _DATE.match(tok)
    if not m:
        return None
    day = m.group(1).translate(str.maketrans("OoIl", "0011"))
    mon = m.group(2).lower().translate(str.maketrans("10|5", "lols"))
    if mon not in _MONTHS or not day.isdigit() or not 1 <= int(day) <= 31:
        return None
    return f"{m.group(3)}-{_MONTHS.index(mon) + 1:02d}-{int(day):02d}"
_AMOUNT = re.compile(r"^\d{1,3}(?:,\d{2,3})*\.\d{2}$")
DEBUG = False


def _ocr_words(png: Path) -> tuple[list[dict], int]:
    out = subprocess.run(
        ["tesseract", str(png), "-", "--psm", "6", "tsv"], capture_output=True, text=True, check=True
    ).stdout
    words, width = [], 0
    for r in csv.DictReader(io.StringIO(out), delimiter="\t", quoting=csv.QUOTE_NONE):
        if r["level"] == "1":
            width = int(r["width"])
        txt = (r.get("text") or "").strip()
        if r["level"] == "5" and txt and float(r["conf"]) > 0:
            words.append({"text": txt, "x": int(r["left"]), "y": int(r["top"]),
                          "r": int(r["left"]) + int(r["width"]), "h": int(r["height"])})
    return words, width


def _lines(words: list[dict]) -> list[list[dict]]:
    """Group words into visual lines by vertical position."""
    words = sorted(words, key=lambda w: (w["y"], w["x"]))
    lines: list[list[dict]] = []
    for w in words:
        if lines and abs(lines[-1][0]["y"] - w["y"]) <= max(12, w["h"] // 2):
            lines[-1].append(w)
        else:
            lines.append([w])
    return [sorted(l, key=lambda w: w["x"]) for l in lines]


def _amount(s: str) -> float:
    return float(s.replace(",", ""))


def _clean_details(parts: list[str]) -> str:
    """HSBC prints e.g. 'UPI20260625000709732' / '617623040917' / 'HSBC INDIA WPB'
    on separate lines. Join with '/' and split the UPI prefix off its ref so the
    rest of the pipeline sees the rail."""
    parts = [re.sub(r"^[^A-Za-z0-9+]+", "", p).strip() for p in parts]  # OCR'd table borders: '|', '.'
    s = "/".join(p for p in parts if p)
    s = re.sub(r"\bUP[I1l]\s?(\d{4,})\s?(\d*)", r"UPI/\1\2", s)  # OCR reads 'UPI' as 'UP1' at times
    return s


def parse_scanned_pdf(path: str | Path, bank: str, dpi: int = 300) -> list[dict]:
    path = Path(path)
    rows: list[dict] = []
    pending: list[str] = []
    date = None
    issues = Counter()

    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["pdftoppm", "-r", str(dpi), "-png", str(path), f"{d}/p"], check=True)
        pages = sorted(Path(d).glob("p-*.png"), key=lambda p: int(p.stem.split("-")[1]))
        for png in pages:
            words, width = _ocr_words(png)
            in_table = False
            fwd_balance_next = False
            for line in _lines(words):
                text = " ".join(w["text"] for w in line)
                if not in_table:
                    # the table starts at the first dated line, or at "Balance Brought Forward"
                    dated = bool(_fix_date("".join(w["text"] for w in line if w["x"] < 0.15 * width)))
                    if dated or re.search(r"(?i)brought\s*forward", text):
                        in_table = True
                    else:
                        continue
                if text.lower().startswith(("the hongkong", "india area", "incorporated", "visit us")):
                    break  # page footer
                details, amounts = [], []
                # the date can come out split ("27" "Jul2026"), so join the date column first
                dcol = "".join(w["text"] for w in line if w["x"] / width < 0.15)
                if _fix_date(dcol):
                    date = _fix_date(dcol)
                for w in line:
                    rel = w["x"] / width
                    if rel < 0.15:
                        continue
                    elif _AMOUNT.match(w["text"]) and w["r"] / width > 0.48:
                        amounts.append(w)
                    elif 0.15 <= rel < 0.5:
                        details.append(w["text"])
                is_fwd = bool(re.search(r"(?i)(brought|carried)\s*forward", text))
                if details and not is_fwd:
                    pending.append(" ".join(details))
                if is_fwd:
                    # page boundary marker; a half-printed transaction may continue on the next page
                    if amounts:
                        rows.append({"anchor": _amount(max(amounts, key=lambda w: w["r"])["text"])})
                    else:
                        fwd_balance_next = True  # OCR sometimes puts the amount on its own line
                    continue
                if not amounts:
                    continue
                if fwd_balance_next and len(amounts) == 1 and not details:
                    rows.append({"anchor": _amount(amounts[0]["text"])})
                    fwd_balance_next = False
                    continue
                fwd_balance_next = False
                bal_w = max(amounts, key=lambda w: w["r"])
                if bal_w["r"] / width < 0.83 or len(amounts) < 2:
                    issues["row without balance"] += 1
                    if DEBUG:
                        print("   no-balance row:", text[:100], "| pending:", pending[-2:])
                    continue
                amt_w = [w for w in amounts if w is not bal_w][0]
                rows.append({
                    "date": date, "narration": _clean_details(pending), "amt": _amount(amt_w["text"]),
                    "col": "credit" if amt_w["r"] / width < 0.69 else "debit", "bal": _amount(bal_w["text"]),
                })
                pending = []

    records = [
        {"bank": bank.lower(), "source_file": path.name, "date": r["date"], "narration": r["narration"],
         "direction": r["direction"], "amount": r["amount"], "balance": r["balance"]}
        for r in _reconcile(rows, issues)
    ]

    seen: Counter[str] = Counter()
    for r in records:
        base = f"{r['bank']}|{r['date']}|{r['narration']}|{r['direction']}|{r['amount']}"
        seen[base] += 1
        r["txn_id"] = hashlib.sha1(f"{base}|{seen[base]}".encode()).hexdigest()[:12]
    if issues:
        print(f"  {path.name}: {dict(issues)}")
    return records


def _reconcile(rows: list[dict], issues: Counter) -> list[dict]:
    """Fix OCR misreads using the running balance.

    For each row, either the amount or the printed balance can be misread. If
    prev ± amount != balance, try both readings and keep the one under which the
    NEXT row is consistent.
    """
    out = []
    bal = None
    txns = [i for i, r in enumerate(rows) if "anchor" not in r]

    def consistent(b, j):
        """does row j agree with a previous balance b?"""
        if j is None:
            return True
        r = rows[j]
        if "anchor" in r:
            return abs(r["anchor"] - b) < 0.01
        return abs(abs(r["bal"] - b) - r["amt"]) < 0.01

    for k, r in enumerate(rows):
        if "anchor" in r:
            if bal is not None and abs(r["anchor"] - bal) > 0.01:
                issues["ignored misread brought-forward"] += 1  # continuity beats one OCR'd number
            else:
                bal = r["anchor"]
            continue
        nxt = k + 1 if k + 1 < len(rows) else None
        if bal is None:
            bal = r["bal"] - r["amt"] if r["col"] == "credit" else r["bal"] + r["amt"]
        diff = round(r["bal"] - bal, 2)
        if abs(abs(diff) - r["amt"]) < 0.01:
            new_bal, amt, d = r["bal"], r["amt"], ("credit" if diff > 0 else "debit")
        else:
            # A: the amount is right (and its column); B: the printed balance is right
            a_bal = round(bal + r["amt"] if r["col"] == "credit" else bal - r["amt"], 2)
            if consistent(a_bal, nxt) or not consistent(r["bal"], nxt):
                new_bal, amt, d = a_bal, r["amt"], r["col"]
                issues["fixed misread balance"] += 1
            else:
                new_bal, amt, d = r["bal"], abs(diff), ("credit" if diff > 0 else "debit")
                issues["fixed misread amount"] += 1
        out.append({**r, "amount": round(amt, 2), "direction": d, "balance": new_bal})
        bal = new_bal
    return out
