"""Bank statement exports (CSV / XLS / XLSX) -> one common transaction format.

Every bank exports differently: preamble rows before the header, different
column names, narrations wrapped over two rows, "Cr"/"Dr" suffixes on amounts.
Instead of one adapter per bank, we detect the header row and map columns by
keyword, which covers HDFC, ICICI, SBI, Axis, Kotak and most others.

Common record (one dict per transaction):
    txn_id, bank, source_file, date (YYYY-MM-DD), narration,
    direction ("debit" | "credit"), amount (float), balance (float | None)
"""

from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter
from pathlib import Path

import pandas as pd

# Keyword -> logical column. Order matters: earlier keys win on ties.
COLUMN_KEYWORDS: dict[str, list[str]] = {
    "date": ["txn date", "transaction date", "tran date", "trans date", "date"],
    "narration": ["narration", "transaction remarks", "description", "particulars", "remarks", "details"],
    "debit": ["withdrawal amt", "withdrawal amount", "withdrawal", "debit amount", "debit", "dr"],
    "credit": ["deposit amt", "deposit amount", "deposit", "credit amount", "credit", "cr"],
    "amount": ["amount", "transaction amount"],
    "drcr": ["dr / cr", "dr/cr", "cr/dr", "type"],
    "balance": ["closing balance", "balance", "bal"],
}
# Headers that contain a keyword but must NOT be mapped to it.
_EXCLUDE = {"date": ["value"], "debit": ["count", "/"], "credit": ["count", "card", "/"]}


def _norm(s: object) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower().replace(".", "")


def _map_columns(header: list[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    cells = [_norm(h) for h in header]
    for logical, keywords in COLUMN_KEYWORDS.items():
        for kw in keywords:
            for i, cell in enumerate(cells):
                if i in mapping.values() or not cell:
                    continue
                if any(x in cell for x in _EXCLUDE.get(logical, [])):
                    continue
                short = len(kw) <= 3  # "dr", "cr", "bal": exact match only
                if (short and (cell == kw or cell.startswith(kw + " ") or cell.startswith(kw + "("))) or (
                    not short and kw in cell
                ):
                    mapping[logical] = i
                    break
            if logical in mapping:
                break
    # fall back to value date if there is no transaction date column
    if "date" not in mapping:
        for i, cell in enumerate(cells):
            if "date" in cell and i not in mapping.values():
                mapping["date"] = i
                break
    return mapping


def _read_rows(path: Path) -> list[list[str]]:
    suffix = path.suffix.lower()
    if suffix in (".xls", ".xlsx"):
        df = pd.read_excel(path, header=None, dtype=str)
        return df.fillna("").astype(str).values.tolist()
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    dialect = csv.excel
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        pass
    return [row for row in csv.reader(text.splitlines(), dialect)]


def _find_header(rows: list[list[str]]) -> tuple[int, dict[str, int]]:
    for i, row in enumerate(rows[:80]):
        m = _map_columns(row)
        has_amount = ("debit" in m and "credit" in m) or "amount" in m
        if "date" in m and "narration" in m and has_amount:
            return i, m
    raise ValueError("could not find a header row with date, narration and amount columns")


def _amount(s: object) -> float:
    t = str(s or "").strip().replace(",", "").replace("₹", "")
    t = re.sub(r"(?i)\s*(cr|dr)\.?$", "", t)
    if t in ("", "-", "nan", "None"):
        return 0.0
    try:
        return abs(float(t))
    except ValueError:
        return 0.0


def _parse_date(s: str) -> str | None:
    s = str(s or "").strip()
    if not s:
        return None
    # Indian exports are day-first: 14/10/26, 14-10-2026, 14 Oct 2026
    for dayfirst in (True,):
        try:
            d = pd.to_datetime(s, dayfirst=dayfirst, errors="raise")
            return d.strftime("%Y-%m-%d")
        except (ValueError, TypeError, OverflowError):
            continue
    return None


def parse_statement(path: str | Path, bank: str) -> list[dict]:
    """Parse one statement export into common records."""
    path = Path(path)
    rows = _read_rows(path)
    h, cols = _find_header(rows)

    def cell(row: list[str], key: str) -> str:
        i = cols.get(key)
        return row[i].strip() if i is not None and i < len(row) else ""

    records: list[dict] = []
    for row in rows[h + 1 :]:
        if not any(str(c).strip() for c in row):
            continue
        date = _parse_date(cell(row, "date"))
        narration = re.sub(r"\s+", " ", cell(row, "narration"))
        if "debit" in cols and "credit" in cols:
            debit, credit = _amount(cell(row, "debit")), _amount(cell(row, "credit"))
        else:
            amt = _amount(cell(row, "amount"))
            flag = (cell(row, "drcr") or cell(row, "amount")).lower()
            is_cr = "cr" in flag and "dr" not in flag
            debit, credit = (0.0, amt) if is_cr else (amt, 0.0)

        if date is None:
            # wrapped narration continuation (HDFC does this), or footer junk
            if narration and records and debit == 0 and credit == 0:
                records[-1]["narration"] += " " + narration
            continue
        if debit == 0 and credit == 0:
            continue  # opening-balance lines etc.
        bal = cell(row, "balance")
        records.append(
            {
                "bank": bank.lower(),
                "source_file": path.name,
                "date": date,
                "narration": narration,
                "direction": "debit" if debit > 0 else "credit",
                "amount": round(debit if debit > 0 else credit, 2),
                "balance": _amount(bal) if bal else None,
            }
        )

    # stable ids; identical lines on the same day get an occurrence suffix
    seen: Counter[str] = Counter()
    for r in records:
        base = f"{r['bank']}|{r['date']}|{r['narration']}|{r['direction']}|{r['amount']}"
        seen[base] += 1
        r["txn_id"] = hashlib.sha1(f"{base}|{seen[base]}".encode()).hexdigest()[:12]
    return records


def ingest_dir(raw_dir: str | Path) -> list[dict]:
    """Parse every export under raw_dir/<bank>/*.{csv,xls,xlsx}."""
    raw_dir = Path(raw_dir)
    out: list[dict] = []
    for bank_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        for f in sorted(bank_dir.iterdir()):
            if f.suffix.lower() in (".csv", ".xls", ".xlsx"):
                out.extend(parse_statement(f, bank=bank_dir.name))
    # de-duplicate overlapping exports (same txn appearing in two downloads)
    uniq = {r["txn_id"]: r for r in out}
    return sorted(uniq.values(), key=lambda r: (r["bank"], r["date"], r["txn_id"]))
