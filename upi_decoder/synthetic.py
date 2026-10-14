"""Synthetic statements with gold labels.

Uses: unit tests, the public demo (no real data ever leaves your laptop), few-shot
examples, and optional training augmentation (capped; NEVER in a test split).

Narration formats imitate how HDFC / ICICI / SBI / Axis / Kotak print lines. All
names, VPAs, refs and companies are invented.
"""

from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path

from .schema import Label

ACCOUNT_HOLDER = "ANIK ROY"  # fictional statement owner

# display name, bank-printed name, vpa, category, counterparty_type, flags
MERCHANTS = [
    ("Swiggy", "SWIGGY LIMITED", "swiggy.stores@ye", "food_delivery", "merchant", []),
    ("Swiggy", "BUNDL TECHNOLOGIES", "swiggyupi@axb", "food_delivery", "merchant", []),
    ("Zomato", "ZOMATO LTD", "zomato-order@ptybl", "food_delivery", "merchant", []),
    ("Swiggy Instamart", "SWIGGY INSTAMART", "instamart@ye", "groceries", "merchant", []),
    ("Blinkit", "BLINK COMMERCE PVT", "blinkit.payu@hdfcbank", "groceries", "merchant", []),
    ("Zepto", "KIRANAKART TECHNOLOGIES", "zeptonow@ybl", "groceries", "merchant", []),
    ("BigBasket", "INNOVATIVE RETAIL CONCEPTS", "bigbasket@icici", "groceries", "merchant", []),
    ("Sri Lakshmi Stores", "SRI LAKSHMI STORES", "paytmqr2810050501@paytm", "groceries", "merchant", []),
    ("Amazon", "AMAZON PAY INDIA", "amazonupi@apl", "shopping", "merchant", []),
    ("Flipkart", "FLIPKART INTERNET", "fkrt@ybl", "shopping", "merchant", []),
    ("Myntra", "MYNTRA DESIGNS", "myntra@icici", "shopping", "merchant", []),
    ("Uber", "UBER INDIA SYSTEMS", "uber.rides@hdfcbank", "travel", "merchant", []),
    ("Rapido", "ROPPEN TRANSPORTATION", "rapido@axisbank", "travel", "merchant", []),
    ("Namma Metro", "BMRCL", "bmrcl.qr@kotak", "travel", "government", []),
    ("IRCTC", "IRCTC UTS", "irctc.uts@sbi", "travel", "government", []),
    ("IndiGo", "INTERGLOBE AVIATION", "indigo@icici", "travel", "merchant", []),
    ("HPCL", "HP PETROL PUMP KORAMANGALA", "hpcl.q1234@ybl", "fuel", "merchant", []),
    ("BESCOM", "BESCOM", "bescom@billdesk", "utilities", "government", []),
    ("Airtel", "AIRTEL PAYMENTS BANK", "airtelpaymentsbank@airtel", "utilities", "merchant", []),
    ("Jio", "RELIANCE JIO INFOCOMM", "jio@sbi", "utilities", "merchant", []),
    ("ACT Fibernet", "ATRIA CONVERGENCE TECH", "actfibernet@icici", "utilities", "merchant", []),
    ("Third Wave Coffee", "THIRD WAVE COFFEE", "twc.q99@ybl", "dining", "merchant", []),
    ("Truffles", "TRUFFLES RESTAURANT", "paytmqr1rtx2938@paytm", "dining", "merchant", []),
    ("CRED", "CRED CLUB", "cred.club@axisb", "emi_loan", "merchant", []),
    ("Zerodha", "ZERODHA BROKING", "zerodhabroking@hdfcbank", "investment", "merchant", []),
    ("Groww", "NEXTBILLION TECHNOLOGY", "groww.brk@validicici", "investment", "merchant", []),
    ("Cult.fit", "CUREFIT HEALTHCARE", "curefit@icici", "other", "merchant", []),
]

AMCS = [
    ("ICICI Prudential MF", "ICICI PRU MF"),
    ("Parag Parikh MF", "PPFAS MUTUAL FUND"),
    ("HDFC MF", "HDFC MUTUAL FUND"),
    ("Mutual fund (BSE StAR)", "BSE LIMITED STAR MF"),
]
INSURERS = [("LIC", "LIC OF INDIA"), ("HDFC Life", "HDFC LIFE INSURANCE"), ("Star Health", "STAR HEALTH AND ALLIED")]
EMPLOYERS = [("Acme Technologies", "ACME TECHNOLOGIES PVT LTD"), ("Nimbus Analytics", "NIMBUS ANALYTICS INDIA PVT LTD")]
PEOPLE = ["RAHUL TIWARI", "SNEHA PATIL", "ARJUN NAIR", "PRIYA SHARMA", "KUNAL BASU", "MEGHA IYER", "SOURAV PAL"]
LANDLORD = "VENKATESH RAO"
PERSON_HANDLES = ["okhdfcbank", "okicici", "okaxis", "oksbi", "ybl", "ibl", "axl"]
BANK_NAMES = {"HDFC": "HDFC BANK", "ICIC": "ICICI BANK", "SBIN": "STATE BANK OF INDIA", "UTIB": "AXIS BANK", "KKBK": "KOTAK MAHINDRA BANK", "YESB": "YES BANK"}
NOTES = ["Payment from Ph", "UPI", "Sent using Paytm", "NA", "Pay to merchant", "PAYMENT", "collect"]


def _ref(rng, n=12):
    return "".join(rng.choice("0123456789") for _ in range(n))


def _ifsc(rng, code=None):
    code = code or rng.choice(list(BANK_NAMES))
    return code + "0" + "".join(rng.choice("0123456789") for _ in range(6))


def _person_vpa(rng, name):
    first = name.split()[0].lower()
    style = rng.random()
    if style < 0.3:
        local = "".join(rng.choice("6789")) + _ref(rng, 9)  # phone number
    elif style < 0.7:
        local = f"{first}{rng.randint(10, 999)}"
    else:
        local = f"{first}.{name.split()[-1].lower()}"
    return f"{local}@{rng.choice(PERSON_HANDLES)}"


def _upi_line(rng, bank, name, vpa, direction, note=None):
    """Format a UPI narration the way each bank prints it."""
    ref = _ref(rng)
    note = note or rng.choice(NOTES)
    code = rng.choice(list(BANK_NAMES))
    trunc = name[: rng.choice([10, 12, 15, 20, 30])]
    if bank == "hdfc":
        return f"UPI-{trunc}-{vpa}-{_ifsc(rng, code)}-{ref}-{note}"
    if bank == "icici":
        return f"UPI/{ref}/{note}/{vpa}/{BANK_NAMES[code].title()}"
    if bank == "sbi":
        pre = "TO TRANSFER-UPI/DR" if direction == "debit" else "BY TRANSFER-UPI/CR"
        return f"{pre}/{ref}/{name[:10]}/{code}/{vpa[:12]}/{note[:8]}--"
    if bank == "axis":
        kind = "P2A" if "@ok" in vpa or vpa.split("@")[0][:1].isdigit() else "P2M"
        return f"UPI/{kind}/{ref}/{trunc}/{note}/{BANK_NAMES[code]}"
    # kotak
    return f"UPI/{trunc}/{ref}/{note}"


def _make_txn(rng, bank, d):
    """Return (narration, direction, amount, Label)."""
    r = rng.random()
    if r < 0.42:  # merchant UPI payment
        disp, printed, vpa, cat, cp, flags = rng.choice(MERCHANTS)
        amt = {"food_delivery": (120, 900), "groceries": (80, 2500), "travel": (40, 6000), "fuel": (300, 3000),
               "utilities": (199, 2500), "dining": (150, 3000), "shopping": (199, 8000), "emi_loan": (2000, 45000),
               "investment": (1000, 25000)}.get(cat, (100, 3000))
        return _upi_line(rng, bank, printed, vpa, "debit"), "debit", round(rng.uniform(*amt), 2), Label(disp, cp, cat, flags)
    if r < 0.58:  # p2p
        name = rng.choice(PEOPLE)
        direction = rng.choice(["debit", "credit"])
        note = rng.choice(["DINNER SPLIT", "TRIP", "UPI", "NA", "FOOD", "CAB SHARE", "Payment from Ph"])
        return (_upi_line(rng, bank, name, _person_vpa(rng, name), direction, note), direction,
                float(rng.choice([150, 200, 350, 500, 650, 1000, 1500, 2000, 2500])), Label(name.title(), "person", "p2p_transfer", []))
    if r < 0.62:  # rent
        return (_upi_line(rng, bank, LANDLORD, _person_vpa(rng, LANDLORD), "debit", rng.choice(["RENT", "HOUSE RENT OCT", "Rent"])),
                "debit", float(rng.choice([18000, 22000, 25000])), Label(LANDLORD.title(), "person", "rent", ["recurring"]))
    if r < 0.66:  # SIP via ACH
        disp, printed = rng.choice(AMCS)
        fmt = {"hdfc": f"ACH D- BD-{printed}-{_ref(rng, 10)}", "icici": f"ACH/{printed}/{_ref(rng, 9)}",
               "sbi": f"DEBIT-ACHDr {_ref(rng, 6)} {printed}", "axis": f"NACH-DR-{printed}-{_ref(rng, 8)}",
               "kotak": f"NACH DR {printed} {_ref(rng, 10)}"}[bank]
        return fmt, "debit", float(rng.choice([2000, 3000, 5000, 10000])), Label(disp, "merchant", "investment", ["recurring", "sip"])
    if r < 0.68:  # insurance
        disp, printed = rng.choice(INSURERS)
        return (f"ACH D- {printed}-{_ref(rng, 9)}" if bank in ("hdfc", "kotak") else f"NACH/{printed}/{_ref(rng, 8)}",
                "debit", float(rng.choice([1500, 2400, 12000, 24000])), Label(disp, "merchant", "insurance", ["recurring"]))
    if r < 0.71:  # salary
        disp, printed = rng.choice(EMPLOYERS)
        utr = "CITIN" + _ref(rng, 11)
        mon = d.strftime("%b %Y").upper()
        fmt = {"hdfc": f"NEFT CR-{_ifsc(rng, 'CITI')}-{printed}-{ACCOUNT_HOLDER}-{utr}",
               "icici": f"NEFT-{utr}-{printed}-SALARY {mon}", "sbi": f"BY TRANSFER-NEFT*CITI0000002*{utr}*{printed[:18]}--",
               "axis": f"NEFT/{utr}/{printed}/SAL {mon}", "kotak": f"NEFT {utr} {printed} SALARY"}[bank]
        return fmt, "credit", float(rng.choice([86500, 98500, 112000])), Label(disp, "employer", "salary_income", ["salary"])
    if r < 0.75:  # ATM
        loc = rng.choice(["KORAMANGALA", "HSR LAYOUT", "INDIRANAGAR", "BERHAMPORE"])
        fmt = {"hdfc": f"NWD-XXXXXX{_ref(rng, 4)}-S1AW{_ref(rng, 4)}-{loc}", "icici": f"ATM/CASH WDL/{loc}/{_ref(rng, 6)}",
               "sbi": f"ATM WDL ATM CASH {_ref(rng, 4)} {loc}", "axis": f"ATM-CASH/{loc}/{_ref(rng, 6)}",
               "kotak": f"ATW-{_ref(rng, 6)}-{loc}"}[bank]
        return fmt, "debit", float(rng.choice([500, 1000, 2000, 5000])), Label("ATM", "bank", "cash_withdrawal", [])
    if r < 0.79:  # self transfer
        fmt = rng.choice([f"IMPS/P2A/{_ref(rng)}/SELF/SBIN/XXXXXXXX{_ref(rng, 4)}",
                          f"IMPS-{_ref(rng)}-{ACCOUNT_HOLDER}-SBIN-XXXXXXX{_ref(rng, 4)}-OWN ACCOUNT",
                          f"NEFT DR-UTIB0<IFSC>-{ACCOUNT_HOLDER}-SELF TRANSFER-{_ref(rng, 10)}"])
        return fmt, rng.choice(["debit", "credit"]), float(rng.choice([5000, 10000, 20000, 50000])), Label("Self", "self", "self_transfer", [])
    if r < 0.82:  # interest
        fmt = {"hdfc": "INTEREST PAID TILL 30-SEP-2026", "icici": f"{_ref(rng, 12)}:Int.Pd:01-07-2026 to 30-09-2026",
               "sbi": "CREDIT INTEREST--", "axis": f"{_ref(rng, 15)}:Int.Pd:01-07-26 to 30-09-26",
               "kotak": "Int.Pd:Q2 FY26-27"}[bank]
        return fmt, "credit", round(rng.uniform(40, 900), 2), Label("Bank interest", "bank", "interest_income", ["interest"])
    if r < 0.85:  # charges
        fmt = rng.choice(["SMS ALERT CHGS QTR-SEP26", "DC ANNUAL FEE+GST", f"IMPS CHG {_ref(rng, 6)}", "CHRG: MIN BAL NON-MAINT"])
        return fmt, "debit", round(rng.choice([15.0, 17.7, 236.0, 590.0]), 2), Label("Bank charges", "bank", "fees_charges", [])
    if r < 0.89:  # refund
        disp, printed, vpa, cat, cp, _ = rng.choice([m for m in MERCHANTS if m[3] in ("food_delivery", "shopping", "travel")])
        return (_upi_line(rng, bank, f"REFUND {printed}", vpa, "credit", "REFUND"), "credit", round(rng.uniform(99, 3000), 2),
                Label(disp, cp, cat, ["refund"]))
    if r < 0.91:  # failed txn reversal
        disp, printed, vpa, cat, cp, _ = rng.choice(MERCHANTS)
        return (f"REV-UPI-{_ref(rng)}-{printed[:15]}-{vpa}", "credit", round(rng.uniform(99, 3000), 2), Label(disp, cp, cat, ["reversal"]))
    if r < 0.94:  # card bill payment
        return (f"BILLPAY CC {rng.choice(['HDFC', 'ICICI', 'AXIS'])} XX{_ref(rng, 4)}", "debit", round(rng.uniform(3000, 60000), 2),
                Label("Credit card bill", "bank", "emi_loan", []))
    if r < 0.97:  # loan EMI
        return (f"ACH D- BAJAJ FINANCE LTD-{_ref(rng, 9)}", "debit", float(rng.choice([2499, 4166, 8333])),
                Label("Bajaj Finance", "bank", "emi_loan", ["emi", "recurring"]))
    # POS card spend
    disp, printed, vpa, cat, cp, flags = rng.choice([m for m in MERCHANTS if m[3] in ("dining", "shopping", "fuel", "groceries")])
    return (f"POS {_ref(rng, 6)}XXXXXX{_ref(rng, 4)} {printed[:20]}", "debit", round(rng.uniform(100, 5000), 2), Label(disp, cp, cat, flags))


# Column layouts per bank (header + preamble rows), approximating real exports.
LAYOUTS = {
    "hdfc": (["Date", "Narration", "Chq./Ref.No.", "Value Dt", "Withdrawal Amt.", "Deposit Amt.", "Closing Balance"], "%d/%m/%y"),
    "icici": (["S No.", "Value Date", "Transaction Date", "Cheque Number", "Transaction Remarks", "Withdrawal Amount (INR )", "Deposit Amount (INR )", "Balance (INR )"], "%d/%m/%Y"),
    "sbi": (["Txn Date", "Value Date", "Description", "Ref No./Cheque No.", "Debit", "Credit", "Balance"], "%d %b %Y"),
    "axis": (["Tran Date", "CHQNO", "PARTICULARS", "DR", "CR", "BAL", "SOL"], "%d-%m-%Y"),
    "kotak": (["Sl. No.", "Transaction Date", "Value Date", "Description", "Chq / Ref No.", "Amount", "Dr / Cr", "Balance"], "%d-%m-%Y"),
}


def generate_bank(bank: str, n: int, start: date, end: date, seed: int = 0) -> list[dict]:
    rng = random.Random(f"{bank}-{seed}")
    span = (end - start).days
    rows = []
    for _ in range(n):
        d = start + timedelta(days=rng.randint(0, span))
        narr, direction, amt, label = _make_txn(rng, bank, d)
        rows.append({"bank": bank, "date": d.isoformat(), "narration": narr, "direction": direction, "amount": amt, "label": label.to_dict()})
    rows.sort(key=lambda r: r["date"])
    return rows


def write_bank_csv(rows: list[dict], path: Path) -> None:
    bank = rows[0]["bank"]
    header, fmt = LAYOUTS[bank]
    path.parent.mkdir(parents=True, exist_ok=True)
    bal = 50000.0
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # preamble junk like real exports
        w.writerow([f"{bank.upper()} BANK - Statement of Account"])
        w.writerow([f"Account Holder: {ACCOUNT_HOLDER}"])
        w.writerow([])
        w.writerow(header)
        for i, r in enumerate(rows, 1):
            d = date.fromisoformat(r["date"]).strftime(fmt)
            amt = f"{r['amount']:,.2f}"
            bal += r["amount"] if r["direction"] == "credit" else -r["amount"]
            dr, cr = (amt, "") if r["direction"] == "debit" else ("", amt)
            if bank == "hdfc":
                w.writerow([d, r["narration"], "0000" + str(i), d, dr, cr, f"{bal:.2f}"])
            elif bank == "icici":
                w.writerow([i, d, d, "", r["narration"], dr or "0.00", cr or "0.00", f"{bal:.2f}"])
            elif bank == "sbi":
                w.writerow([d, d, r["narration"], "", dr, cr, f"{bal:,.2f}"])
            elif bank == "axis":
                w.writerow([d, "", r["narration"], dr, cr, f"{bal:.2f}", "1234"])
            else:  # kotak: single amount + Dr/Cr column
                w.writerow([i, d, d, r["narration"], "", amt, "DR" if r["direction"] == "debit" else "CR", f"{bal:.2f}"])
        w.writerow([])
        w.writerow(["*** End of statement ***"])
