"""Label schema: what the model must produce for each narration.

The model only predicts the *semantic* fields. Mechanical fields (date, amount,
debit/credit, payment rail) come from the statement columns and `rules.py`.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

CATEGORIES: dict[str, str] = {
    "food_delivery": "Swiggy, Zomato, other food-delivery apps",
    "groceries": "Blinkit, Zepto, BigBasket, Instamart, DMart, kirana stores",
    "dining": "restaurants, cafes, bars paid in person",
    "shopping": "Amazon, Flipkart, Myntra, Ajio, Nykaa, retail stores",
    "travel": "cabs, autos, metro, trains, flights, buses, hotels",
    "fuel": "petrol pumps (HPCL, BPCL, IOCL, Shell)",
    "utilities": "electricity, water, gas, mobile, broadband, DTH, bill payments",
    "rent": "house rent, PG, maintenance paid to a landlord or society",
    "emi_loan": "loan EMIs, credit-card bill payments, BNPL repayments",
    "investment": "SIPs, mutual funds, stocks, NPS, PPF, FD/RD creation",
    "insurance": "life, health, motor insurance premiums",
    "salary_income": "salary credits from an employer",
    "interest_income": "savings/FD interest, dividends",
    "p2p_transfer": "money sent to or received from another person",
    "self_transfer": "moves between the account holder's own accounts/wallets",
    "cash_withdrawal": "ATM / branch cash withdrawals",
    "fees_charges": "bank charges, SMS fees, AMC, GST on charges, penalties",
    "other": "anything that clearly fits none of the above",
}

COUNTERPARTY_TYPES = ("merchant", "person", "self", "bank", "government", "employer")

FLAGS = ("refund", "reversal", "emi", "sip", "salary", "interest", "recurring")

# Categories that matter at ITR time; used by the app's "tax view".
TAX_RELEVANT = ("rent", "investment", "insurance", "interest_income", "salary_income")


@dataclass
class Label:
    merchant: str
    counterparty_type: str
    category: str
    flags: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        """Canonical, compact JSON used as the training target."""
        d = asdict(self)
        d["flags"] = sorted(set(d["flags"]))
        return json.dumps(d, ensure_ascii=False, separators=(", ", ": "))

    def to_dict(self) -> dict:
        return json.loads(self.to_json())


def validate(d: dict) -> tuple[Label | None, list[str]]:
    """Validate a raw dict against the schema. Returns (label, errors)."""
    errors: list[str] = []
    if not isinstance(d, dict):
        return None, ["not a JSON object"]
    merchant = d.get("merchant")
    if not isinstance(merchant, str) or not merchant.strip():
        errors.append("merchant missing")
    cp = d.get("counterparty_type")
    if cp not in COUNTERPARTY_TYPES:
        errors.append(f"bad counterparty_type: {cp!r}")
    cat = d.get("category")
    if cat not in CATEGORIES:
        errors.append(f"bad category: {cat!r}")
    flags = d.get("flags", [])
    if isinstance(flags, str):
        flags = [f.strip() for f in flags.split(",") if f.strip()]
    if not isinstance(flags, list):
        errors.append("flags not a list")
        flags = []
    bad = [f for f in flags if f not in FLAGS]
    if bad:
        errors.append(f"bad flags: {bad}")
    if errors:
        return None, errors
    return Label(merchant.strip(), cp, cat, [f for f in flags if f in FLAGS]), []


_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def parse_model_output(text: str) -> tuple[Label | None, list[str]]:
    """Extract and validate the first JSON object in a model's reply."""
    if text is None:
        return None, ["empty output"]
    t = text.strip()
    # strip ``` fences and any leftover <think> block
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL).strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.MULTILINE).strip()
    m = _JSON_OBJ.search(t)
    if not m:
        return None, ["no JSON object found"]
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return None, [f"invalid JSON: {e.msg}"]
    return validate(d)


def normalize_merchant(name: str) -> str:
    """Loose normalisation for merchant exact-match scoring."""
    s = name.lower()
    s = re.sub(r"\b(pvt|private|ltd|limited|llp|inc|india|technologies|tech|services|stores?)\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())
