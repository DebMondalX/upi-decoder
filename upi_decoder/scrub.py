"""PII scrubber. Runs locally, BEFORE any narration is sent anywhere (Tinker included).

What it removes:
  * account numbers, card masks, UPI/IMPS/NEFT reference numbers  -> <ACCT>, <REF12> ...
  * phone numbers (bare or as a VPA handle)                        -> <PHONE>
  * IFSC codes (keeps the 4-letter bank code, which is useful)     -> HDFC0<IFSC>
  * emails, PAN                                                    -> <EMAIL>, <PAN>
  * names of people you list in data/people.txt                   -> stable fake names
  * personal-looking VPA handles (rahulk99@okhdfcbank)             -> user_7f3a@okhdfcbank

What it keeps: merchant names and merchant VPAs (swiggy.stores@ye), payment rail
words (UPI/NEFT/ACH), and short numbers. Those are exactly the signal the model needs.

It is a first pass, not a guarantee. `scripts/01_ingest_scrub.py` writes a report
of the most common remaining capitalised tokens so you can eyeball for leaks and
add names to data/people.txt.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

# UPI handles that individuals use. Merchants mostly use their own or @ybl/@paytm
# with a recognisable prefix; we only pseudonymise local parts that look personal.
PERSONAL_PSP_HANDLES = {
    "okhdfcbank", "okicici", "okaxis", "oksbi", "ybl", "ibl", "axl", "apl", "paytm",
    "upi", "jupiteraxis", "fbl", "freecharge", "kotak", "icici", "hdfcbank", "sbi",
    "axisbank", "naviaxis", "superyes", "waaxis", "wahdfcbank", "waicici", "wasbi",
    "pthdfc", "ptsbi", "ptyes", "ptaxis", "yesbank", "idfcbank", "airtel", "barodampay",
}

# local parts that mean "this is a business" even on a personal PSP handle
MERCHANT_HINTS = re.compile(
    r"(?i)(merchant|store|stores|shop|mart|pay\b|payments?|razorpay|cashfree|billdesk|bharatpe|"
    r"paytmqr|q\d{6,}|mab\.|gpay-|\.rzp|swiggy|zomato|zepto|blinkit|amazon|flipkart|uber|ola|"
    r"irctc|bescom|airtel|jio|zerodha|groww|cred|phonepe|bigbasket|instamart|myntra|rapido)"
)

FAKE_NAMES = [
    "AARAV MEHTA", "ISHA RAO", "KABIR SEN", "MEERA IYER", "ROHAN DAS", "TARA NAIR",
    "VIVAAN SHAH", "ANYA BOSE", "ARJUN PILLAI", "DIYA KAPOOR", "NEEL JOSHI", "RIYA GHOSH",
    "SAMAR KHANNA", "ZOYA MALIK", "DEV MENON", "KIARA DUTTA", "OM KULKARNI", "PIHU SAHA",
    "YASH VERMA", "LEELA REDDY", "ADI CHAWLA", "NAINA GUPTA", "RAHUL TIWARI", "SARA PAUL",
]

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_PAN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
_IFSC = re.compile(r"\b([A-Z]{4})0[A-Z0-9]{6}\b")
_MASKED_ACCT = re.compile(r"(?i)\b(?:X{2,}|\*{2,})[X*]*\d{2,6}\b")
# VPA local parts are lowercase/digits in practice; requiring that stops the regex
# from swallowing an upper-case name glued on with '-' (e.g. "HEALTHCARE-curefit@icici").
_VPA = re.compile(r"(?<![\w.])([a-z0-9][a-z0-9._]*(?:-[a-z0-9._]+)*)@([a-zA-Z][\w]*)\b")
_CARD = re.compile(r"\b\d{4,6}[X*]{4,}\d{4}\b")
_PHONE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)")
_LONG_NUM = re.compile(r"(?<![\d<])\d{6,}(?![\d>])")


def _stable_hash(s: str, n: int = 4) -> str:
    return hashlib.sha1(s.lower().encode()).hexdigest()[:n]


def load_people(path: str | Path | None) -> list[str]:
    """One name per line. Include yourself, family, friends, landlord, flatmates.
    Also add the truncated forms banks print (e.g. 'DEBARGHYA M', 'DEBARGH')."""
    if not path or not Path(path).exists():
        return []
    names = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if len(line) >= 3:
            names.append(line)
    # longest first so 'RAHUL KUMAR' is replaced before 'RAHUL'
    return sorted(set(names), key=len, reverse=True)


class Scrubber:
    def __init__(self, people: list[str] | None = None, self_names: list[str] | None = None):
        self.people = people or []
        self.self_names = [n.lower() for n in (self_names or [])]
        # Banks truncate names ("RAHUL TIWARI" -> "RAHUL TIWA"), so also match
        # prefixes of each name down to 8 characters.
        self._people_re = []
        for p in self.people:
            variants = {p} | {p[:k].rstrip() for k in range(min(8, len(p)), len(p))}
            alt = "|".join(re.escape(v) for v in sorted(variants, key=len, reverse=True))
            self._people_re.append((re.compile(r"(?i)(?<![A-Za-z])(?:" + alt + r")(?![A-Za-z])"), p))
        self._people_tokens = {w.lower() for p in self.people for w in p.split() if len(w) >= 3}

    def fake_name(self, real: str) -> str:
        if real.lower() in self.self_names:
            return "SELF"
        idx = int(_stable_hash(real, 6), 16) % len(FAKE_NAMES)
        return FAKE_NAMES[idx]

    def _vpa(self, m: re.Match) -> str:
        local, handle = m.group(1), m.group(2)
        if _PHONE.fullmatch(local) or re.fullmatch(r"\d{10}", local):
            return f"<PHONE>@{handle}"
        tokens = set(re.findall(r"[a-z]{3,}", local.lower()))
        personal = (
            bool(tokens & self._people_tokens)                     # contains a listed person's name
            or re.fullmatch(r"[a-z]+\d{1,4}", local) is not None  # sneha778
            or (handle.lower() in PERSONAL_PSP_HANDLES             # rahul.tiwari@okaxis
                and re.fullmatch(r"[a-z]{3,}\.[a-z]{3,}", local) is not None
                and not MERCHANT_HINTS.search(local))
        )
        if personal:
            return f"user_{_stable_hash(local)}@{handle}"
        return m.group(0)  # merchant VPA: keep, it's the best signal we have

    def scrub(self, text: str) -> str:
        t = text
        # order matters: emails before VPAs (both contain '@')
        t = _EMAIL.sub("<EMAIL>", t)
        t = _VPA.sub(self._vpa, t)
        t = _PAN.sub("<PAN>", t)
        t = _IFSC.sub(lambda m: f"{m.group(1)}0<IFSC>", t)
        t = _CARD.sub("<CARD>", t)
        t = _MASKED_ACCT.sub("<ACCT>", t)
        t = _PHONE.sub("<PHONE>", t)
        t = _LONG_NUM.sub(lambda m: f"<REF{len(m.group(0))}>" if len(m.group(0)) == 12 else "<NUM>", t)
        for rx, real in self._people_re:
            t = rx.sub(self.fake_name(real), t)
        return t


def leak_report(narrations: list[str], top: int = 60) -> list[tuple[str, int]]:
    """Most frequent ALL-CAPS word pairs left after scrubbing.
    Personal names tend to show up here; merchants too. Eyeball it."""
    from collections import Counter

    stop = {"UPI", "NEFT", "IMPS", "RTGS", "ACH", "NACH", "ATM", "POS", "DR", "CR", "SELF", "BANK",
            "PAYMENT", "TRANSFER", "TO", "FROM", "BY", "LTD", "LIMITED", "PVT", "INDIA", "THE", "SAL"}
    c: Counter[str] = Counter()
    for n in narrations:
        words = re.findall(r"\b[A-Z][A-Z]{2,}\b", n)
        words = [w for w in words if w not in stop]
        for a, b in zip(words, words[1:]):
            c[f"{a} {b}"] += 1
    return c.most_common(top)


def vpa_report(narrations: list[str], top: int = 60) -> list[tuple[str, int]]:
    """Most frequent VPAs kept after scrubbing. A person's name here = add them to people.txt."""
    from collections import Counter

    c: Counter[str] = Counter()
    for n in narrations:
        for m in _VPA.finditer(n):
            if not m.group(1).startswith("user_") and "<" not in m.group(0):
                c[m.group(0)] += 1
    return c.most_common(top)
