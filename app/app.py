"""UPI Decoder - local app.

    pip install -e ".[app]"
    streamlit run app/app.py

Upload statement exports -> categorised ledger, monthly summary, ITR view.
With the "Local model" backend nothing leaves your machine.
"""

import asyncio
import sys
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from upi_decoder import ingest, rules, scrub, synthetic  # noqa: E402
from upi_decoder.llm import make_backend  # noqa: E402
from upi_decoder.prompts import build_messages  # noqa: E402
from upi_decoder.schema import TAX_RELEVANT, parse_model_output  # noqa: E402

st.set_page_config(page_title="UPI Decoder", page_icon="🧾", layout="wide")
st.title("UPI Decoder")
st.caption("Turns `UPI/DR/4123.../YESB/...` into *Swiggy · food delivery*. Runs on your laptop.")

with st.sidebar:
    st.header("Model")
    mode = st.radio("Backend", ["Local model (Ollama)", "Rules only (offline)", "Tinker checkpoint"], index=0)
    spec = None
    if mode.startswith("Local"):
        spec = "ollama:" + st.text_input("Ollama model", "upi-decoder")
    elif mode.startswith("Tinker"):
        spec = "tinker:" + st.text_input("BASE@tinker://...", "Qwen/Qwen3.5-4B@tinker://")
        st.warning("Tinker sends scrubbed narrations to a remote API.")
    st.header("Privacy")
    people = st.text_area("Names to scrub (one per line)", help="You, family, friends, landlord").splitlines()
    self_name = st.text_input("Your name as the bank prints it", "")

files = st.file_uploader("Statement exports (CSV / XLS / XLSX)", accept_multiple_files=True)
demo = st.button("…or try with a synthetic demo statement")


@st.cache_data(show_spinner=False)
def load(files_bytes, demo_flag):
    recs = []
    with tempfile.TemporaryDirectory() as d:
        if demo_flag:
            rows = synthetic.generate_bank("hdfc", 250, date(2026, 4, 1), date(2026, 9, 30), seed=42)
            p = Path(d) / "demo.csv"
            synthetic.write_bank_csv(rows, p)
            recs += ingest.parse_statement(p, "hdfc")
        for name, data in files_bytes:
            p = Path(d) / name
            p.write_bytes(data)
            bank = name.split("_")[0].split(".")[0].lower()
            recs += ingest.parse_statement(p, bank)
    return recs


def decode(recs, spec, people, self_name):
    s = scrub.Scrubber(people=people + ([self_name] if self_name else []), self_names=[self_name] if self_name else [])
    for r in recs:
        r["narration"] = s.scrub(r["narration"])
        r["key"] = rules.dedup_key(r["narration"])
    # one model call per distinct narration pattern, not per row
    uniq = {}
    for r in recs:
        uniq.setdefault(r["key"], r)
    preds = {k: None for k in uniq}
    if spec:
        backend = make_backend(spec)
        msgs = [build_messages(r["narration"], r["direction"], r["amount"], mode="short") for r in uniq.values()]
        with st.spinner(f"Decoding {len(msgs)} distinct narrations with {backend.name}…"):
            res = asyncio.run(backend.generate_many(msgs, progress=False))
        for k, g in zip(uniq, res):
            preds[k] = parse_model_output(g.text)[0]
    out = []
    for r in recs:
        base = rules.rules_predict(r["narration"], r["direction"], r["amount"])
        lab = preds.get(r["key"]) or base
        check = spec is not None and (preds.get(r["key"]) is None or lab.category != base.category)
        out.append({
            "date": r["date"], "narration": r["narration"], "direction": r["direction"], "amount": r["amount"],
            "merchant": lab.merchant, "category": lab.category, "counterparty": lab.counterparty_type,
            "flags": ", ".join(lab.flags), "check": "⚠︎" if check else "",
        })
    return pd.DataFrame(out)


if files or demo:
    recs = load(tuple((f.name, f.getvalue()) for f in files or []), bool(demo))
    if not recs:
        st.error("No transactions found. Is this a CSV/Excel export with date, narration and amount columns?")
        st.stop()
    df = decode([dict(r) for r in recs], spec, [p for p in people if p.strip()], self_name.strip())
    df["month"] = df["date"].str[:7]

    t1, t2, t3 = st.tabs(["Ledger", "Monthly summary", "ITR view"])
    with t1:
        st.dataframe(df.drop(columns=["month"]), width="stretch", hide_index=True)
        st.caption("⚠︎ = the model and the rules disagree, or the model output was invalid. Worth a look.")
        st.download_button("Download categorised CSV", df.to_csv(index=False), "upi_decoded.csv", "text/csv")
    with t2:
        spend = df[(df.direction == "debit") & ~df.category.isin(["self_transfer"])]
        pivot = spend.pivot_table(index="category", columns="month", values="amount", aggfunc="sum", fill_value=0)
        st.dataframe(pivot.style.format("₹{:,.0f}"), width="stretch")
        st.bar_chart(spend.groupby("category")["amount"].sum().sort_values(ascending=False))
    with t3:
        df["fy"] = df["date"].apply(lambda d: f"FY{int(d[:4]) + (d[5:7] >= '04') - 1}-{str(int(d[:4]) + (d[5:7] >= '04'))[2:]}")
        tax = df[df.category.isin(TAX_RELEVANT)].pivot_table(index="category", columns="fy", values="amount", aggfunc="sum", fill_value=0)
        st.dataframe(tax.style.format("₹{:,.0f}"), width="stretch")
        st.caption("Rent (HRA), investments (80C/ELSS), insurance (80C/80D), interest (80TTA), salary. Not tax advice; check with your CA.")
else:
    st.info("Upload one or more statement exports, or try the synthetic demo.")
