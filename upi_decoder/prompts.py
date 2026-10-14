"""Prompts.

Two system prompts on purpose:
  * FULL_SYSTEM  - long instructions + category definitions + rules. Used by the
                   teacher (labelling) and by the few-shot baselines.
  * SHORT_SYSTEM - one line. Used to train and run the fine-tuned student.
The gap between them is part of the result: the fine-tuned model has absorbed the
instructions, so every call is ~10x fewer input tokens (prompt distillation).
"""

from __future__ import annotations

import json

from .schema import CATEGORIES, COUNTERPARTY_TYPES, FLAGS, Label

_cat_lines = "\n".join(f"- {k}: {v}" for k, v in CATEGORIES.items())

FULL_SYSTEM = f"""You decode Indian bank-statement narrations into structured JSON.

Input: one narration line, its direction (debit/credit) and amount in INR.
Output: ONLY a JSON object, no prose, with exactly these keys:
  "merchant": clean human-readable name of the counterparty (e.g. "Swiggy", "Airtel", "Rahul Tiwari", "Self")
  "counterparty_type": one of {list(COUNTERPARTY_TYPES)}
  "category": one of the categories below
  "flags": list, subset of {list(FLAGS)} (empty list if none)

Categories:
{_cat_lines}

Rules:
- Placeholders like <REF12>, <ACCT>, <PHONE>, <NUM>, HDFC0<IFSC> are redacted identifiers; ignore them.
- user_xxxx@handle is a redacted personal UPI id -> almost always a person (p2p_transfer).
- Merchant VPAs (swiggy.stores@ye, paytmqr...@paytm) identify the merchant even if the name field is truncated.
- Names are often truncated by the bank ("SWIGGY LIMI", "RAHUL TIWAR"); expand to the obvious full name only if certain.
- SELF / own account transfers -> category self_transfer, counterparty_type self, merchant "Self".
- Refunds: keep the original category (a Swiggy refund is food_delivery) and add the "refund" flag.
- Failed-transaction reversals: add "reversal".
- Mutual fund / SIP debits via ACH or BSE/ICCL -> investment + "sip"; recurring mandates -> "recurring".
- Credit-card bill payments and loan EMIs -> emi_loan (+ "emi" for EMIs).
- Amazon: shopping, unless it is clearly Amazon Pay bill/recharge (utilities) or Amazon Fresh (groceries).
- Salary credits -> salary_income, counterparty_type employer, flag "salary".
- Bank interest / dividends -> interest_income, counterparty_type bank, flag "interest".
- If a person is paid a large round amount around the start of the month with "rent" in the note -> rent.
- When genuinely unsure of the category, use "other" rather than guessing."""

SHORT_SYSTEM = "Decode the Indian bank narration into JSON with merchant, counterparty_type, category, flags."


def format_user(narration: str, direction: str, amount: float) -> str:
    return f"Narration: {narration}\nDirection: {direction}\nAmount: {amount:.2f}"


# Few-shot examples (synthetic; already scrubbed form).
FEW_SHOT: list[tuple[dict, Label]] = [
    (
        {"narration": "UPI/DR/<REF12>/SWIGGY LIMI/YESB/swiggy.stores@ye/Payment fr", "direction": "debit", "amount": 412.0},
        Label("Swiggy", "merchant", "food_delivery", []),
    ),
    (
        {"narration": "UPI-AARAV MEHTA-user_3fa2@okhdfcbank-HDFC0<IFSC>-<REF12>-DINNER SPLIT", "direction": "credit", "amount": 650.0},
        Label("Aarav Mehta", "person", "p2p_transfer", []),
    ),
    (
        {"narration": "NEFT CR-CITI0<IFSC>-ACME TECHNOLOGIES PVT LTD-SALARY SEP 2026-<NUM>", "direction": "credit", "amount": 98500.0},
        Label("Acme Technologies", "employer", "salary_income", ["salary"]),
    ),
    (
        {"narration": "ACH D- BD-ICICI PRU MF-<ACCT>", "direction": "debit", "amount": 5000.0},
        Label("ICICI Prudential MF", "merchant", "investment", ["recurring", "sip"]),
    ),
    (
        {"narration": "IMPS/P2A/<REF12>/SELF/SBIN/<ACCT>", "direction": "debit", "amount": 20000.0},
        Label("Self", "self", "self_transfer", []),
    ),
    (
        {"narration": "UPI/CR/<REF12>/REFUND/AMAZON PAY/YESB/amazonrefunds@apl", "direction": "credit", "amount": 1299.0},
        Label("Amazon", "merchant", "shopping", ["refund"]),
    ),
]


def build_messages(narration: str, direction: str, amount: float, mode: str = "short") -> list[dict]:
    """mode: 'short' (fine-tuned student), 'zero' (full prompt, no examples),
    'few' (full prompt + few-shot examples)."""
    user = format_user(narration, direction, amount)
    if mode == "short":
        return [{"role": "system", "content": SHORT_SYSTEM}, {"role": "user", "content": user}]
    msgs = [{"role": "system", "content": FULL_SYSTEM}]
    if mode == "few":
        for row, label in FEW_SHOT:
            msgs.append({"role": "user", "content": format_user(row["narration"], row["direction"], row["amount"])})
            msgs.append({"role": "assistant", "content": label.to_json()})
    msgs.append({"role": "user", "content": user})
    return msgs


def sft_example(narration: str, direction: str, amount: float, label: Label) -> dict:
    """One training conversation for the cookbook's FromConversationFileBuilder."""
    msgs = build_messages(narration, direction, amount, mode="short")
    msgs.append({"role": "assistant", "content": label.to_json()})
    return {"messages": msgs}


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)
