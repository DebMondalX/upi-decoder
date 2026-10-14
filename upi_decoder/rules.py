"""Deterministic layer.

1. `payment_mode()`  - which rail a line went through (UPI, NEFT, ACH, ATM ...).
2. `dedup_key()`     - narration with refs/ids blanked, for near-duplicate removal.
3. `rules_predict()` - a keyword/regex categoriser. This is baseline #1 in the eval:
   roughly what most expense-tracker apps do today.
"""

from __future__ import annotations

import re

from .schema import Label

MODES = [
    ("UPI", r"\bUPI\b"),
    ("NEFT", r"\bNEFT\b"),
    ("IMPS", r"\bIMPS\b|\bMMT\b"),
    ("RTGS", r"\bRTGS\b"),
    ("ACH", r"\bN?ACH\b|\bECS\b|\bSI\b"),
    ("ATM", r"\bATM\b|\bATW\b|\bNWD\b|\bCASH ?WDL\b|\bAWB\b"),
    ("CARD", r"\bPOS\b|\bECOM\b|\bVPS\b|\bIPS\b|\bDEBIT CARD\b"),
    ("CHEQUE", r"\bCHQ\b|\bCLG\b|\bCHEQUE\b"),
    ("INTEREST", r"\bINT\.?\s?PD\b|\bINTEREST\b|\bCREDIT INTEREST\b"),
]


def payment_mode(narration: str) -> str:
    n = narration.upper()
    for mode, rx in MODES:
        if re.search(rx, n):
            return mode
    return "OTHER"


def dedup_key(narration: str) -> str:
    """'UPI/DR/<REF12>/SWIGGY/YESB/swiggy.stores@ye/Payment' and the same line
    with a different ref map to one key."""
    k = narration.upper()
    k = re.sub(r"<[A-Z0-9]+>", "#", k)
    k = re.sub(r"USER_[0-9A-F]{4}", "USER", k)
    k = re.sub(r"\d+", "#", k)
    k = re.sub(r"[^A-Z#@.]+", " ", k)
    return " ".join(k.split())


# ---------------------------------------------------------------------------
# Baseline categoriser
# ---------------------------------------------------------------------------

# (pattern, canonical merchant, category, counterparty_type, flags)
MERCHANTS: list[tuple[str, str, str, str, list[str]]] = [
    (r"SWIGGY(?!.*INSTAMART)", "Swiggy", "food_delivery", "merchant", []),
    (r"INSTAMART", "Swiggy Instamart", "groceries", "merchant", []),
    (r"ZOMATO", "Zomato", "food_delivery", "merchant", []),
    (r"BLINKIT|GROFERS", "Blinkit", "groceries", "merchant", []),
    (r"ZEPTO|KIRANAKART", "Zepto", "groceries", "merchant", []),
    (r"BIG ?BASKET|INNOVATIVE RETAIL", "BigBasket", "groceries", "merchant", []),
    (r"DMART|AVENUE SUPERMARTS", "DMart", "groceries", "merchant", []),
    (r"AMAZON ?PAY.*(BILL|RECHARGE)", "Amazon Pay", "utilities", "merchant", []),
    (r"AMAZON|AMZN", "Amazon", "shopping", "merchant", []),
    (r"FLIPKART", "Flipkart", "shopping", "merchant", []),
    (r"MYNTRA", "Myntra", "shopping", "merchant", []),
    (r"AJIO|RELIANCE RETAIL", "Ajio", "shopping", "merchant", []),
    (r"NYKAA", "Nykaa", "shopping", "merchant", []),
    (r"UBER", "Uber", "travel", "merchant", []),
    (r"\bOLA\b|ANI TECHNOLOGIES", "Ola", "travel", "merchant", []),
    (r"RAPIDO|ROPPEN", "Rapido", "travel", "merchant", []),
    (r"IRCTC", "IRCTC", "travel", "merchant", []),
    (r"MAKEMYTRIP|MMT ", "MakeMyTrip", "travel", "merchant", []),
    (r"INDIGO|INTERGLOBE", "IndiGo", "travel", "merchant", []),
    (r"REDBUS", "redBus", "travel", "merchant", []),
    (r"BMRCL|METRO RAIL", "Namma Metro", "travel", "merchant", []),
    (r"HPCL|HINDUSTAN PETROLEUM", "HPCL", "fuel", "merchant", []),
    (r"BPCL|BHARAT PETROLEUM", "BPCL", "fuel", "merchant", []),
    (r"IOCL|INDIAN OIL", "Indian Oil", "fuel", "merchant", []),
    (r"\bSHELL\b", "Shell", "fuel", "merchant", []),
    (r"BESCOM", "BESCOM", "utilities", "government", []),
    (r"BWSSB", "BWSSB", "utilities", "government", []),
    (r"AIRTEL", "Airtel", "utilities", "merchant", []),
    (r"\bJIO\b|RELIANCE JIO", "Jio", "utilities", "merchant", []),
    (r"ACT ?FIBERNET|ATRIA CONVERGENCE", "ACT Fibernet", "utilities", "merchant", []),
    (r"BBPS|BILLDESK", "Bill payment", "utilities", "merchant", []),
    (r"NOBROKER", "NoBroker", "rent", "merchant", []),
    (r"CRED\b|DREAMPLUG", "CRED", "emi_loan", "merchant", []),
    (r"ZERODHA", "Zerodha", "investment", "merchant", []),
    (r"GROWW|NEXTBILLION", "Groww", "investment", "merchant", []),
    (r"ICCL|INDIAN CLEARING|NSE CLEARING|BSE ?STAR|MF UTILITIES|CAMS|KFIN", "Mutual fund", "investment", "merchant", ["sip"]),
    (r"\bLIC\b|LIFE INSURANCE CORP", "LIC", "insurance", "merchant", []),
    (r"PRU ?LIFE|HDFC LIFE|SBI LIFE|MAX LIFE|TATA AIA|POLICYBAZAAR|STAR HEALTH|NIVA BUPA|ACKO|DIGIT INSUR",
     "Insurance", "insurance", "merchant", []),
    (r"\bMF\b|MUTUAL FUND|\bAMC\b(?!\s*CHG)", "Mutual fund", "investment", "merchant", ["sip"]),
    (r"STARBUCKS|CAFE|COFFEE|RESTAURANT|DHABA|BAR\b|BREW|CHAAYOS|THIRD WAVE", "Restaurant", "dining", "merchant", []),
]

_SALARY = re.compile(r"\bSAL(ARY)?\b|\bPAYROLL\b")
_INTEREST = re.compile(r"\bINT\.?\s?PD\b|\bINTEREST\b|\bDIVIDEND\b|\bDIV\b")
_ATM = re.compile(r"\bATM\b|\bATW\b|\bNWD\b|\bCASH ?WDL\b|\bAWB\b")
_FEES = re.compile(r"\bCHGS?\b|\bCHARGES?\b|\bFEE\b|\bGST\b|\bSMS ?ALERT|\bAMC ?CHG|\bPENALTY\b|\bMIN BAL")
_SELF = re.compile(r"\bSELF\b|\bOWN A/?C\b|\bTO OWN\b")
_REFUND = re.compile(r"\bREFUND\b|\bRFND\b")
_REVERSAL = re.compile(r"\bREV(ERSAL|RSL)?\b|\bRETURN(ED)?\b")
_RENT = re.compile(r"\bRENT\b|\bHOUSE RENT\b|\bMAINTENANCE\b")
_EMI = re.compile(r"\bEMI\b|\bLOAN\b|\bBAJAJ FIN|\bHOME LOAN|\bCC ?PAYMENT\b|\bCREDIT CARD\b|\bBILLPAY CC\b")

_NOISE = {
    "UPI", "DR", "CR", "P2A", "P2M", "PAYMENT", "PAY", "NEFT", "IMPS", "RTGS", "MMT", "TO", "FROM", "BY",
    "TRANSFER", "ACH", "NACH", "D", "C", "BD", "INB", "MOB", "NA", "NULL", "SENT", "USING", "PAYTM",
}


def _guess_counterparty(narration: str) -> str:
    """Pull the most name-like chunk from a UPI/NEFT narration."""
    parts = re.split(r"[/\-:|]", narration)
    for p in parts:
        p = p.strip()
        if not p or "@" in p or "<" in p:
            continue
        words = [w for w in re.findall(r"[A-Za-z][A-Za-z.&']+", p) if w.upper() not in _NOISE]
        # skip 4-letter bank codes like YESB / HDFC
        if len(words) == 1 and re.fullmatch(r"[A-Z]{4}", words[0]):
            continue
        if words:
            return " ".join(words)[:40].title()
    return "Unknown"


def rules_predict(narration: str, direction: str = "debit", amount: float = 0.0) -> Label:
    n = narration.upper()
    flags: list[str] = []
    if _REFUND.search(n):
        flags.append("refund")
    if _REVERSAL.search(n):
        flags.append("reversal")

    if _ATM.search(n):
        return Label("ATM", "bank", "cash_withdrawal", flags)
    if _INTEREST.search(n) and direction == "credit":
        return Label("Bank interest", "bank", "interest_income", flags + ["interest"])
    if _FEES.search(n) and direction == "debit" and amount < 2000:
        return Label("Bank charges", "bank", "fees_charges", flags)
    if _SALARY.search(n) and direction == "credit":
        return Label(_guess_counterparty(narration), "employer", "salary_income", flags + ["salary"])
    if _SELF.search(n):
        return Label("Self", "self", "self_transfer", flags)

    for rx, merchant, cat, cp, extra in MERCHANTS:
        if re.search(rx, n):
            f = flags + extra
            if payment_mode(narration) == "ACH" and cat in ("investment", "insurance", "emi_loan"):
                f.append("recurring")
            return Label(merchant, cp, cat, f)

    if _EMI.search(n):
        return Label(_guess_counterparty(narration), "bank", "emi_loan", flags + ["emi"])
    if _RENT.search(n):
        return Label(_guess_counterparty(narration), "person", "rent", flags)

    mode = payment_mode(narration)
    if mode in ("UPI", "IMPS", "NEFT", "RTGS"):
        # unknown counterparty on a P2P rail: most likely a person
        return Label(_guess_counterparty(narration), "person", "p2p_transfer", flags)
    return Label(_guess_counterparty(narration), "merchant", "other", flags)
