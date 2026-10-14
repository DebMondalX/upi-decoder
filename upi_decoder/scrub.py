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

# merchant QR / terminal ids that look like name+digits but are businesses
_MERCHANT_ID_PREFIX = re.compile(r"(q\d|paytmqr|paytm-|gpay-|bharatpe|mab\.|ombk|vyapar|yespay|ibkpos|stq|msw|pos\d|upiswiggy)")

FAKE_NAMES = [
    "AARAV MEHTA", "ISHA RAO", "KABIR SEN", "MEERA IYER", "ROHAN DAS", "TARA NAIR",
    "VIVAAN SHAH", "ANYA BOSE", "ARJUN PILLAI", "DIYA KAPOOR", "NEEL JOSHI", "RIYA GHOSH",
    "SAMAR KHANNA", "ZOYA MALIK", "DEV MENON", "KIARA DUTTA", "OM KULKARNI", "PIHU SAHA",
    "YASH VERMA", "LEELA REDDY", "ADI CHAWLA", "NAINA GUPTA", "RAHUL TIWARI", "SARA PAUL",
    "IRA BANERJEE", "KAVYA HEGDE", "MIHIR DESAI", "NISHA CHOPRA", "PRANAV JAIN", "RUHI SETHI",
    "SAHIL BHATT", "TANVI MISHRA", "UDAY KAMATH", "VEDA SHARMA", "AYAAN QURESHI", "BELA FERNANDES",
    "CHIRAG SONI", "DHRUV AGARWAL", "ESHA BHAT", "FARHAN ALI", "GAURI PATIL", "HARSH VYAS",
    "INDU MENON", "JAI SINGH", "KETAN SHAH", "LATA PRASAD", "MANAV SURI", "NIKHIL ARORA",
]

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_PAN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
_IFSC = re.compile(r"\b([A-Z]{4})0[A-Z0-9]{6}\b")
_MASKED_ACCT = re.compile(r"(?i)\b(?:X{2,}|\*{2,})[X*]*\d{2,6}\b")
# VPA local parts are lowercase/digits in practice; requiring that stops the regex
# from swallowing an upper-case name glued on with '-' (e.g. "HEALTHCARE-curefit@icici").
_VPA = re.compile(r"(?<![\w.])([a-z0-9][a-z0-9._]*(?:-[a-z0-9._]+)*)@([a-zA-Z][\w]*)\b")
_CARD = re.compile(r"\b\d{4,6}[X*]{4,}\d{4}\b|\b\d{4}[ -]\d{4}[ -]\d{4}[ -]\d{4}\b")
_PHONE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)")
# gateway/transaction ids like ICI75e5ef09fd924df2b2..., PAZN69hq6zcF33..., XQOGP1V4GZS1
_TXN_ID = re.compile(r"\b(?=[A-Za-z0-9]*\d)(?=[A-Za-z0-9]*[A-Za-z])[A-Za-z0-9]{12,}\b(?!@)")
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
        # your own name is scrubbed too (to "SELF"), whether or not it's in people.txt
        names = set(people or []) | set(self_names or [])
        names = {n for n in names if len(n.strip()) >= 4 and not _NOT_NAMES.match(n.strip())}
        self.people = sorted(names, key=len, reverse=True)
        self.self_names = [n.lower() for n in (self_names or [])]
        # Banks truncate names ("RAHUL TIWARI" -> "RAHUL TIWA"), so also match
        # prefixes of each name down to 8 characters.
        self._people_re = []
        for p in self.people:
            variants = {p} | {p[:k].rstrip() for k in range(min(8, len(p)), len(p))}
            joined = re.sub(r"[^A-Za-z]", "", p)  # truncated VPAs: "debarghyamonda"
            if len(joined) >= 8 and joined != p:
                variants |= {joined[:k] for k in range(8, len(joined) + 1)}
            alt = "|".join(re.escape(v) for v in sorted(variants, key=len, reverse=True))
            self._people_re.append((re.compile(r"(?i)(?<![A-Za-z])(?:" + alt + r")[A-Za-z]*"), p))
        self._people_tokens = {
            w.lower() for p in self.people for w in p.split()
            if len(w) >= 4 and not _BRAND_SUBSTR.search(w.lower()) and not _BUSINESS_WORDS.search(w)
        }

    def fake_name(self, real: str) -> str:
        r = re.sub(r"[^a-z]", "", real.lower())
        for sn in self.self_names:
            j = re.sub(r"[^a-z]", "", sn)
            if r and (j.startswith(r) or r == j):  # "MONDAL DEB" is a truncation of "MONDAL DEBARGHYA"
                return "SELF"
        idx = int(_stable_hash(real, 6), 16) % len(FAKE_NAMES)
        return FAKE_NAMES[idx]

    def _vpa(self, m: re.Match) -> str:
        local, handle = m.group(1), m.group(2)
        if local.startswith("user_"):
            return m.group(0)  # already pseudonymised (keeps re-scrubbing idempotent)
        if _PHONE.fullmatch(local) or re.fullmatch(r"\d{10}", local):
            return f"<PHONE>@{handle}"
        tokens = set(re.findall(r"[a-z]{3,}", local.lower()))
        personal = (
            bool(tokens & self._people_tokens)                     # contains a listed person's name
            or (re.fullmatch(r"[a-z][a-z._]+\d+[-._\d]*", local) is not None  # sneha778, rayb05724-2
                and not _MERCHANT_ID_PREFIX.match(local) and not MERCHANT_HINTS.search(local))
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
        t = _TXN_ID.sub(lambda m: m.group(0) if m.group(0).startswith(("user_",)) else "<ID>", t)
        t = _PHONE.sub("<PHONE>", t)
        t = _LONG_NUM.sub(lambda m: f"<REF{len(m.group(0))}>" if len(m.group(0)) == 12 else "<NUM>", t)
        for rx, real in self._people_re:
            t = rx.sub(self.fake_name(real), t)
        return self._truncated_vpas(self._vpa_next_to_person(t))

    _FAKE = None

    def _vpa_next_to_person(self, t: str) -> str:
        parts = t.split("/")
        if len(parts) < 3 or parts[0].strip().upper() != "UPI":
            return t
        if Scrubber._FAKE is None:
            Scrubber._FAKE = set(FAKE_NAMES) | {"SELF"}
        # a field holding a fake name plus leftover initials/surname ("MR X Y", "X Y YADAV") -> just the fake name
        for i, f in enumerate(parts[1:4], start=1):
            for fk in Scrubber._FAKE:
                if fk in f and f.strip() != fk and re.fullmatch(r"[A-Za-z .'-]+", f.strip()):
                    parts[i] = fk
                    break
        if (parts[1].strip() in Scrubber._FAKE and "@" in parts[2] and not parts[2].startswith(("user_", "<"))
                and "<" not in parts[2] and not _merchant_vpa(parts[2])):
            local, _, handle = parts[2].partition("@")
            parts[2] = f"user_{_stable_hash(local)}@{handle}"
        return "/".join(parts)

    def _truncated_vpas(self, t: str) -> str:
        """ICICI prints 'UPI/<name>/<vpa>/<note>/...' and cuts the VPA at ~15 chars,
        often dropping the '@handle' (e.g. 'gokulraj.js', 'rameshreddychi'). Treat bare
        lowercase tokens in fields 1-2 of such lines like VPA local parts."""
        parts = t.split("/")
        if len(parts) < 4 or parts[0].strip().upper() != "UPI" or parts[1].upper() in ("DR", "CR", "P2A", "P2M"):
            return t
        for i in (1, 2):
            f, at = (parts[i][:-1], "@") if parts[i].endswith("@") else (parts[i], "")
            if f.startswith("user_") or "@" in f or "<" in f or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{4,15}", f) or not re.search(r"[a-z]{3}", f):
                continue
            if _MERCHANT_ID_PREFIX.match(f) or MERCHANT_HINTS.search(f) or _BRAND_SUBSTR.search(f):
                continue
            parts[i] = f"user_{_stable_hash(f)}{at}"
        return "/".join(parts)


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


# Words that mark a UPI name field as a business rather than a person.
_BUSINESS_WORDS = re.compile(
    r"(?i)\b(ltd|limited|pvt|private|llp|inc|corp|co|company|bank|store|stores|shop|mart|traders?|enterprises?|"
    r"agency|agencies|services?|solutions?|tech|technologies|india|retail|foods?|restaurant|hotel|cafe|coffee|"
    r"bakery|sweets|medical|medicals|pharma|pharmacy|hospital|clinic|resi|residency|petrol|fuel|motors|"
    r"travels?|tours?|trip|online|digital|payments?|pay|recharge|prepaid|mobile|broadband|fibernet|"
    r"google|amazon|flipkart|zomato|swiggy|zerodha|groww|jio|airtel|reliance|tata|licious|rapido|uber|ola|"
    r"qwikcilver|woohoo|starbucks|chaayos|bigtree|bookmyshow|livpure|thomas|indian|clearing|iccl|razorpay|"
    r"bangalore|bengaluru|bmtc|bescom|temple|ttd|devasthanam|sagar|roll|rolls|kitchen|dhaba|bhavan|darshini|"
    r"general|provision|kirana|super|market|fresh|dairy|milk|salon|spa|gym|fitness|school|college|"
    r"university|society|association|trust|foundation|excitel|easy|cle|kotak|mahindra|hdfc|icici|axis|sbi|medi|mandate\w*|merchant|credit|card|generating|powered\w*|payvia\w*|test|stock|ipo|nps|gold|leftovers|refill|"
    r"irctc|ticketing|verificati\w*|cinepolis|nilgiris|westside|abhibus|shoffr|safegold|easebuzz|fancode|bmrcl|"
    r"donatekart|pvr|inox|ixigo|ibibo|redbus|makemytrip|blinkit|bigbasket|zepto|instamart)\b"
)


_BRAND_SUBSTR = re.compile(
    r"(paytm|grofers|excitel|broadban|razorpay|zerodha|swiggy|zomato|amazon|flipkart|google|jio|airtel|"
    r"store|mart|shop|pay|bank|merchant|recharge|online|india|metro|bmrc|starbucks|thomascook|qwikcilver|"
    r"easytrip|livpure|hotstar|housing|spotify|licious|blinkit|bookmyshow|playstore|citrus|rzp|iccl|bsestar|"
    r"woohoo|rapido|cloud|hmshost|irctc|uber|ola|netflix|youtube|apple|myntra|nykaa|ajio|dmart|zepto|bigbasket|"
    r"instamart|cred|phonepe|bharatpe|billdesk|cashfree|payu|ccavenue|tata|reliance|lic|insurance|mf|fund|"
    r"shoffr|cinepolis|nilgiris|westside|abhibus|makemytrip|safegold|easebuzz|fancode|bmrcl|donatekart|"
    r"pvr|inox|ixigo|ibibo|redbus|cleartrip|yatra|klook|agoda|oyo|decathlon|ikea|croma|lenskart|pharmeasy|"
    r"apollo|medplus|practo|cult|curefit|urbanclap|urbancompany|dunzo|bluedart|delhivery|matchpass|bundlepass|"
    r"beginnerpack|verificati|ticketing|intent|remark|refund|order|donation|hosp|consulta|education|dento)"
)


# never treat these as names, whatever position they appear in
_NOT_NAMES = re.compile(
    r"(?i)^(upi|upi .*|.* upi|imps|neft|rtgs|nach|ach|self|na|null|payment.*|pay .*|collect|refund|food|lunch|dinner|"
    r"rest|rent.*|bill.*|autopay|autope.*|juspay|dtax|tax|gst|salary|sal|cash|atm|transfer|trf|test|misc.*|"
    r"split|trip|cab|gift|loan|emi|fees?|charges?|recharge|order|txn|transaction.*|sent .*|received .*)$"
)


def _merchant_vpa(v: str) -> bool:
    local = v.strip().lower().split("@")[0]
    if not local or local.startswith("user_") or "<" in local:
        return False
    return bool(_MERCHANT_ID_PREFIX.match(local) or MERCHANT_HINTS.search(local) or _BRAND_SUBSTR.search(local))


def _name_like(f: str) -> bool:
    return (
        re.fullmatch(r"[A-Za-z][A-Za-z .'-]{2,29}", f) is not None
        and not _BUSINESS_WORDS.search(f)
        and not _NOT_NAMES.match(f.strip())
        and re.search(r"[A-Za-z]{3}", f) is not None
    )


def suggest_people(narrations: list[str]) -> list[str]:
    """Candidate personal names from UPI counterparty fields, for data/people.txt.

    Looks at the name-like fields of UPI lines (ICICI/Kotak: field 1, Axis/SBI: field 3),
    keeps the ones that are letters only and contain no business words. Small shops are
    often paid to the owner's personal account; those get treated as people, which is
    the privacy-safe side to err on. Review the file before using it.
    """
    from collections import Counter

    c: Counter[str] = Counter()
    for n in narrations:
        parts = [p.strip() for p in re.split(r"[/]", n)]
        if not parts:
            continue
        # any name-like field right after a long numeric ref (HSBC lines whose 'UPI' prefix OCR dropped)
        if re.fullmatch(r"\d{10,22}", parts[0]) and len(parts) > 1 and _name_like(parts[1]):
            c[parts[1]] += 1
        if "UPI" not in parts[0].upper():
            continue
        # ICICI/Kotak put the name in field 1; Axis (P2A/P2M) and SBI (DR/CR) in field 3.
        # HSBC: UPI/<ref>/<ref>/<name>
        if len(parts) > 3 and (parts[1].upper() in ("P2A", "P2M", "DR", "CR") or parts[1].isdigit()):
            idxs = [3]
        else:
            idxs = [1]
        for idx in idxs:
            if idx >= len(parts):
                continue
            f = parts[idx]
            if not _name_like(f):
                continue
            if idx == 1 and " " not in f.strip() and len(parts) > 2 and _merchant_vpa(parts[2]):
                continue  # one-word payee on a merchant VPA (SHOFFR, CINEPOLIS): a business
            if " " not in f and f.islower() and _BRAND_SUBSTR.search(f):  # e.g. excitelbroadban, grofersindia
                continue
            if f.upper() in {"UPI", "NA", "PAYMENT", "SELF", "COLLECT", "REFUND", "FOOD"} or f.upper().startswith(("PAYMENT", "UPI")):
                continue
            c[f] += 1
    return [name for name, _ in c.most_common()]
