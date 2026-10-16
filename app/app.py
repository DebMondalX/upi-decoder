"""UPI Decoder - app.

    pip install -e ".[app]"
    streamlit run app/app.py                 # local: your statements + the model in Ollama
    UPI_DEMO=1 streamlit run app/app.py      # hosted demo: synthetic statement + cached model output

Upload statement exports -> categorised ledger, monthly summary, ITR view.
With the "Local model" backend nothing leaves your machine. The hosted demo never accepts
uploads: real statements should not go to a server, which is the whole point of the project.
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from upi_decoder import ingest, rules, scrub  # noqa: E402
from upi_decoder.schema import TAX_RELEVANT, parse_model_output, validate  # noqa: E402
from upi_decoder.synthetic import demo_statement_records  # noqa: E402

DEMO = os.environ.get("UPI_DEMO") == "1"
CACHE_PATH = ROOT / "app" / "demo_cache.json"

st.set_page_config(page_title="UPI Decoder", page_icon="🧾", layout="wide")
st.title("UPI Decoder")
st.caption(
    "Turns `UPI/ZOMATO LIM/<ID>.g/UPI/AXIS BANK/…` into *Zomato · food delivery*. "
    "A 4B open model fine-tuned on Tinker, built to run on your laptop."
)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Model")
    if DEMO:
        mode = st.radio("Backend", ["Fine-tuned model (cached output)", "Keyword rules only"], index=0,
                        help="The hosted demo has no GPU, so it shows the fine-tuned model's answers, "
                             "pre-computed for the synthetic statement.")
        spec = None
        people, self_name = [], ""
    else:
        mode = st.radio("Backend", ["Local model (Ollama)", "Keyword rules only", "Tinker checkpoint"], index=0)
        spec = None
        if mode.startswith("Local"):
            spec = "ollama:" + st.text_input("Ollama model", "upi-decoder")
        elif mode.startswith("Tinker"):
            spec = "tinker:" + st.text_input("BASE@tinker://...", "Qwen/Qwen3.5-4B@tinker://")
            st.warning("Tinker sends scrubbed narrations to a remote API.")
        st.header("Privacy")
        people = st.text_area("Names to scrub (one per line)", help="You, family, friends, landlord").splitlines()
        self_name = st.text_input("Your name as the bank prints it", "")
    st.divider()
    st.markdown("[Code on GitHub](https://github.com/DebMondalX/upi-decoder)")

# ---------------------------------------------------------------- input
if "use_demo" not in st.session_state:
    st.session_state.use_demo = DEMO
files = []
if DEMO:
    st.info(
        "**This is the hosted demo, so it only runs on a synthetic statement.** "
        "Real bank statements should never be uploaded to a server; that's the point of the project. "
        "To use your own, run it locally (see GitHub): your data and the model both stay on your machine."
    )
else:
    files = st.file_uploader("Statement exports (CSV / XLS / XLSX / scanned PDF)", accept_multiple_files=True)
    if st.button("…or try with a synthetic demo statement"):
        st.session_state.use_demo = True


@st.cache_data(show_spinner=False)
def load_uploads(files_bytes, people, self_name):
    recs = []
    with tempfile.TemporaryDirectory() as d:
        for name, data in files_bytes:
            p = Path(d) / name
            p.write_bytes(data)
            bank = name.split("_")[0].split(".")[0].lower()
            recs += ingest.parse_statement(p, bank)
    s = scrub.Scrubber(people=list(people) + ([self_name] if self_name else []), self_names=[self_name] if self_name else [])
    for r in recs:
        r["narration"] = s.scrub(r["narration"])
        r["key"] = rules.dedup_key(r["narration"])
    return recs


@st.cache_data(show_spinner=False)
def load_demo():
    return demo_statement_records()


@st.cache_data(show_spinner=False)
def demo_cache():
    if not CACHE_PATH.exists():
        return {}
    return json.loads(CACHE_PATH.read_text()).get("labels", {})


@st.cache_data(show_spinner=False)
def run_model(spec, items):
    """items: tuple of (key, narration, direction, amount); one call per distinct pattern."""
    from upi_decoder.llm import make_backend
    from upi_decoder.prompts import build_messages

    backend = make_backend(spec)
    msgs = [build_messages(n, d, a, mode="short") for _, n, d, a in items]
    res = asyncio.run(backend.generate_many(msgs, progress=False))
    out = {}
    for (k, *_), g in zip(items, res):
        lab = parse_model_output(g.text)[0]
        out[k] = lab.to_dict() if lab else None
    return out


def decode(recs, use_model: bool):
    uniq = {}
    for r in recs:
        uniq.setdefault(r["key"], r)
    preds: dict = {}
    if use_model:
        if DEMO or (st.session_state.use_demo and spec is None):
            preds = demo_cache()
            if not preds:
                st.warning("No cached model output found (app/demo_cache.json); showing keyword rules.")
        elif spec:
            items = tuple((k, r["narration"], r["direction"], r["amount"]) for k, r in uniq.items())
            with st.spinner(f"Decoding {len(items)} distinct narrations…"):
                preds = run_model(spec, items)
    rows = []
    for r in recs:
        base = rules.rules_predict(r["narration"], r["direction"], r["amount"])
        p = preds.get(r["key"])
        lab = validate(p)[0] if p else None
        used = lab or base
        check = use_model and (lab is None or lab.category != base.category)
        rows.append({
            "date": r["date"], "narration": r["narration"], "direction": r["direction"], "amount": r["amount"],
            "merchant": used.merchant, "category": used.category, "counterparty": used.counterparty_type,
            "flags": ", ".join(used.flags), "rules say": base.category if check else "",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- output
recs = []
if files:
    recs = load_uploads(tuple((f.name, f.getvalue()) for f in files), tuple(p for p in people if p.strip()), self_name.strip())
    if not recs:
        st.error("No transactions found. Is this a CSV/Excel/PDF export with date, narration and amount columns?")
        st.stop()
elif st.session_state.use_demo:
    recs = load_demo()

if recs:
    use_model = not mode.startswith("Keyword")
    df = decode([dict(r) for r in recs], use_model)
    df["month"] = df["date"].str[:7]

    n_diff = int((df["rules say"] != "").sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", len(df))
    c2.metric("Spent", f"₹{df[df.direction == 'debit'].amount.sum():,.0f}")
    c3.metric("Received", f"₹{df[df.direction == 'credit'].amount.sum():,.0f}")
    c4.metric("Model ≠ keyword rules", n_diff if use_model else "–")

    t1, t2, t3 = st.tabs(["Ledger", "Monthly summary", "ITR view"])
    with t1:
        st.dataframe(df.drop(columns=["month"]), width="stretch", hide_index=True)
        if use_model:
            st.caption("'rules say' shows what a keyword-rules expense app would have guessed where the model disagrees.")
        st.download_button("Download categorised CSV", df.drop(columns=["month"]).to_csv(index=False), "upi_decoded.csv", "text/csv")
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
elif not DEMO:
    st.info("Upload one or more statement exports, or try the synthetic demo.")
