"""Labelling conventions applied on top of the teacher labels (the review pass).

    python scripts/03c_label_conventions.py

Reads data/splits/train_pool.jsonl + data/interim/teacher_labels.fixed.jsonl and writes
data/labeled/train_pool.jsonl. Every rule is a written-down labelling decision, so the
training labels and the hand-labelled test sets follow the same conventions.
"""
import json, re, sys
sys.path.insert(0, ".")
from collections import Counter
from upi_decoder.schema import validate
from upi_decoder.scrub import FAKE_NAMES
from upi_decoder.io import write_jsonl
rows={json.loads(l)["txn_id"]:json.loads(l) for l in open("data/splits/train_pool.jsonl")}
T={json.loads(l)["txn_id"]:json.loads(l)["label"] for l in open("data/interim/teacher_labels.fixed.jsonl")}
FAKE=set(FAKE_NAMES)
def L(m,cp,cat,fl=()): return {"merchant":m,"counterparty_type":cp,"category":cat,"flags":sorted(set(fl))}
def payee(n):
    p=[x.strip() for x in n.split("/")]
    if p[0]=="UPI":
        if len(p)>1 and p[1] in FAKE: return p[1]
        if len(p)>3 and (p[1].startswith("<") or p[1].isdigit()) and p[3] in FAKE: return p[3]
        if len(p)>2 and p[1].startswith("<REF") and p[2] in FAKE: return p[2]
    if len(p)>1 and p[0].startswith("<REF") and p[1] in FAKE: return p[1]
    return None
changed=Counter()
def fix(r, lab):
    n=r["narration"]; N=n.upper(); cr=r["direction"]=="credit"; amt=r["amount"]
    def set_(new, why):
        if new!=lab: changed[why]+=1
        return new
    # money moved between your own accounts *for* investing (note says so) -> investment (author's call)
    if lab["category"]=="self_transfer" and re.search(r"\b(STOCKS?|IPO|MUTUAL FUNDS?|GOLDETF|GOLD|NPS|^SIP|/SIP/|INVEST\w*)\b", N) and "LUNCH" not in N:
        fl=["sip"] if re.search(r"/SIP/|MUTUAL FUND", N) else []
        return set_(L("Self","self","investment",fl),"self transfer for investing")
    if "NET BANKING SI" in N: return set_(L("Standing instruction","bank","other",["recurring"]),"standing instr")
    if re.search(r"\bGST\b|CGST|SGST", N) and amt<1000 and not cr: return set_(L("Bank charges","bank","fees_charges"),"gst")
    if "FOREX PAYMENT" in N or N.startswith("RFX "): return set_(L("Forex remittance","bank","other"),"forex")
    if "ITDTAX REFUND" in N: return set_(L("Income Tax Department","government","other",["refund"]),"tax refund")
    if "VIT-PAYMENTORREFUND" in N: return set_(L("VIT Vellore","merchant","other",["refund"]),"vit refund")
    if "PGUPI" in N: return set_(L("Unknown","person","rent"),"pg rent")
    if "DIGITAL GO" in N or "SAFEGOLD" in N: return set_(L("SafeGold","merchant","investment"),"digital gold")
    if "ROPPEN" in N or "RAPIDO" in N: return set_(L("Rapido","merchant","travel"),"rapido")
    if N.startswith("WMS/") : return set_(L("Mutual fund","merchant","investment",["recurring","sip"]),"mf sip")
    if N.startswith("ACH/INDIAN CLEARING") and not cr: return set_(L("Indian Clearing Corp","merchant","investment",["recurring","sip"]),"iccl")
    if N.startswith("ACH/ZERODHA") and not cr: return set_(L("Zerodha","merchant","investment",["recurring","sip"]),"zerodha ach")
    if "ZERODHA" in N and "DSCNB" in N and cr: return set_(L("Zerodha","merchant","investment"),"zerodha payout")
    if "CC BILLPAY" in N or N.startswith("TO <CARD>"): return set_(L("Credit card bill","bank","emi_loan"),"cc bill")
    if re.search(r"BIL/INFT/.*\b(" + "|".join(FAKE) + r") SELF\b", N):
        nm=re.search(r"(" + "|".join(FAKE) + r") SELF", N).group(1).title()
        return set_(L(nm,"person","p2p_transfer"),"family transfer")
    if re.search(r"CINEPOLIS|PVR|INOX|BOOKMYSHOW|BIGTREE", N): return set_(L(lab["merchant"],"merchant","other"),"cinema")
    if re.search(r"RENTOMOJO|RENTMOJO|EDUNETWORK", N): return set_(L("Rentomojo","merchant","other"),"rentomojo")
    if re.search(r"BLINK", N): return set_(L("Blinkit","merchant","groceries"),"blinkit")
    if re.search(r"LICIOUS|DELIGHTFUL GOURMET", N): return set_(L("Licious","merchant","groceries"),"licious")
    if re.search(r"ZOMATO", N) and lab["category"]!="food_delivery" and "refund" not in lab["flags"]: return set_(L("Zomato","merchant","food_delivery",lab["flags"]),"zomato")
    if re.search(r"BANGALOREMETRO|BMRC|BANGALORE METRO|BMRCL|^UPI/BANGALORE/", N): return set_(L("Namma Metro","government","travel"),"metro")
    if N.startswith("UPI/") and lab["category"]=="fees_charges":
        f1=n.split("/")[1]
        if f1.startswith(("<PHONE>","user_")): return set_(L("Unknown","person","p2p_transfer"),"upi 'fee' -> p2p")
        return set_(L("Unknown","merchant","other"),"upi 'fee' -> other")
    if N.startswith("ACH/") and cr and lab["category"]=="interest_income" and lab["counterparty_type"]!="merchant":
        return set_({**lab,"counterparty_type":"merchant"},"dividend cp")
    pn=payee(n)
    # a brand VPA next to the name means the name field was over-scrubbed (e.g. HARSH VYAS/amazonpay@apl): trust the VPA
    if pn and re.search(r"amazon|amzn|goindigo|indigo|bookmyshow|swiggy|zomato|zepto|blinkit|uber|rapido|irctc|airtel|jio|flipkart|myntra|cinepolis|pvr", n.lower()):
        pn=None
    if pn:
        if pn=="SELF": return set_(L("Self","self","self_transfer"),"self")
        if lab["category"] not in ("p2p_transfer","rent") or lab["counterparty_type"]!="person" or lab["merchant"].upper()!=pn:
            cat="rent" if lab["category"]=="rent" else "p2p_transfer"
            return set_(L(pn.title(),"person",cat),"fake-name payee -> person")
    if lab["merchant"]=="Unknown" and lab["category"]=="other" and lab["counterparty_type"]!="merchant":
        return set_({**lab,"counterparty_type":"merchant"},"unknown -> merchant")
    return lab
out=[]
for tid,r in rows.items():
    lab=fix(r, T[tid])
    v,e=validate(lab); assert v,(tid,e,lab)
    out.append({**r,"label":v.to_dict(),"labeled_by":"teacher+claude-review"})
write_jsonl("data/labeled/train_pool.jsonl", out)
print(len(out)); 
for k,v in changed.most_common(): print(f"{v:4} {k}")
print(Counter(o["label"]["category"] for o in out).most_common())
