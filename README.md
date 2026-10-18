# UPI Decoder

**Your bank statement says `UPI/DR/412398765432/SWIGGY LIMI/YESB/swiggy.stores@ye/Payment fr`.
UPI Decoder says *Swiggy · food delivery · ₹412*. On your laptop, offline.**

A small open model (Qwen3.5-4B), fine-tuned with [Tinker](https://tinker-docs.thinkingmachines.ai/),
that turns cryptic Indian bank narrations into a clean, categorised ledger: who you paid, what for,
and which lines are noise (self-transfers, refunds, reversals). It also gives you a monthly summary
and an ITR view (rent, investments, insurance, interest).

Built for the [DEV Hacktoberfest Open-Source AI Challenge, Week 2: Side Hustle](https://dev.to/challenges/hacktoberfest-week2-2026-10-12).

**Write-up:** [I fine-tuned a 4B model to read my bank statement, and it beat the 397B model that taught it](https://dev.to/debmondalx/i-fine-tuned-a-4b-model-to-read-my-bank-statement-and-it-beat-the-397b-model-that-taught-it-3hm8)  
**Demo (runs offline on a MacBook Air, synthetic statement):** https://youtu.be/T4COhEbQ5YI

## Results

Trained on ~2,300 of my own transactions from three banks (ICICI, HDFC, HSBC, Dec 2023 – Oct 2026).
Two held-out test sets, one row per narration pattern:

**`test_in`**: later months than anything in training (150 rows)

| system | category acc | macro-F1 | merchant exact | counterparty acc | input tok/row | $/1k rows |
|---|---|---|---|---|---|---|
| keyword rules | 65.3% | 0.37 | 72.0% | 82.0% | – | – |
| Qwen3.5-4B, few-shot | 78.7% | 0.65 | 71.3% | 74.7% | 1,360 | $0.48 |
| Qwen3.5-397B, few-shot (the teacher) | 86.7% | 0.82 | 78.7% | 81.3% | 1,360 | $2.47 |
| **Qwen3.5-4B + Tinker LoRA** | **92.7%** | **0.83** | **82.0%** | **95.3%** | **77** | **$0.06** |
| **same model, 4-bit, MacBook Air (Ollama)** | **92.7%** | **0.86** | **82.7%** | **96.0%** | **77** | **$0** |

**`test_unseen`**: HSBC, a bank entirely absent from training, OCR'd from scanned PDFs (57 rows)

| system | category acc | macro-F1 | merchant exact |
|---|---|---|---|
| keyword rules | 75.4% | 0.68 | 73.7% |
| Qwen3.5-4B, few-shot | 80.7% | 0.75 | 73.7% |
| Qwen3.5-397B, few-shot | 87.7% | 0.85 | 77.2% |
| **Qwen3.5-4B + Tinker LoRA** | **87.7%** | **0.84** | **77.2%** |
| **same, 4-bit on laptop** | **86.0%** | **0.82** | **77.2%** |

* The fine-tuned 4B **beats its 397B teacher** on later months and matches it on an unseen bank,
  at ~1/44th the cost per row on Tinker, and for free on a laptop (~2 s/row, p50, no network).
* 4-bit compression for local use costs essentially nothing (one row on the unseen bank).
* Full tables, including self-transfer F1 and a run trained on unreviewed teacher labels:
  [`results/results.md`](results/results.md), [`results/r0/`](results/r0/), [`results/local/`](results/local/).
  Refund F1 shows 0 because neither test window contains a refund; it isn't measured yet.
* Test labels were written from scratch by Claude (the coding assistant this was built with),
  independently of the Qwen teacher, and spot-checked by me (20 random rows, 1 corrected).

## Why open weights

Bank statements are about as private as data gets. A closed API means sending every UPI line you've
ever made to someone else's server. Here the expensive step (labelling and training) happens once,
on scrubbed data; the result is a 4B model you download and run locally with Ollama. No per-row API
bill, no data leaving the machine, and you can retrain it for your own bank's quirks.

## How it works

```
bank exports ──► 01 ingest + scrub PII ──► 02 honest splits ──► 03 teacher labels (Qwen3.5-397B on Tinker)
                    (local only)            train / later-months /     │
                                            unseen-bank                 ▼
                                                    03b/03c label conventions + 04 review
                                                                        │
         08 export → Ollama (4-bit) ◄── 06 LoRA fine-tune ◄── 05 SFT data
                         │               (Qwen3.5-4B on Tinker)
                         ▼
                 07 eval: rules vs 4B few-shot vs 397B teacher vs fine-tuned vs local
```

* **Scrubbing first.** Account numbers, refs, phones, IFSC, PAN, emails and the names you list are
  replaced *before* anything is sent to Tinker. Merchant names and merchant VPAs are kept: they are
  the signal.
* **Prompt distillation.** The teacher reads a long prompt with category definitions and rules. The
  student is trained with a one-line prompt, so it carries the rules in its weights: 77 input
  tokens per call instead of 1,360.
* **Honest eval.** Test sets are labelled from scratch (not teacher-reviewed), de-duplicated
  (one row per narration pattern), taken from *later months* than training, plus a whole bank the
  model never saw.

## Quick start

```bash
git clone https://github.com/DebMondalX/upi-decoder && cd upi-decoder
python -m venv .venv && source .venv/bin/activate
pip install -e ".[tinker,app,dev]"
pytest -q
```

### Dry run on synthetic data (no API key, no real data)

```bash
python scripts/00_make_synthetic.py
python scripts/01_ingest_scrub.py --raw data/synthetic/raw
python scripts/02_make_splits.py --unseen-bank kotak
for s in train_pool test_in test_unseen; do python scripts/04_review.py from-gold --split $s --gold data/synthetic/gold.jsonl; done
python scripts/05_build_sft.py
python scripts/07_eval.py --only-rules
```

(The rules baseline scores high on synthetic data because both use the same vocabulary. Real
statements are much messier; that's the point of the model.)

### Scanned PDF statements (HSBC and others)

Some banks only give image-only PDFs. Drop them in `data/raw/<bank>/` like any export;
they're OCR'd with Tesseract and every row is checked against the running balance,
which also auto-corrects most misread amounts. Needs `brew install poppler tesseract`.

### Real run

```bash
# 0. statements -> data/raw/<bank>/  (see data/README.md), names -> data/people.txt
python scripts/01_ingest_scrub.py --self-name "YOUR NAME" --self-name "SURNAME NAME" --auto-people
#    read data/interim/scrub_report.txt; add any leftover names to people.txt; re-run.
#    Incremental: add another bank later and re-run; earlier banks' scrubbed rows are kept,
#    so you can delete raw statements once processed.
python scripts/02_make_splits.py --unseen-bank axis

export TINKER_API_KEY=...
python scripts/03_teacher_label.py --limit 20      # sanity check
python scripts/03_teacher_label.py                 # the rest

python scripts/04_review.py export --split train_pool
python scripts/04_review.py export --split test_in --blank
python scripts/04_review.py export --split test_unseen --blank
#    review/label the three sheets in data/review/, then:
for s in train_pool test_in test_unseen; do python scripts/04_review.py import --split $s; done

python scripts/05_build_sft.py
python scripts/06_train.py --name r1
python scripts/07_eval.py --run runs/r1
python scripts/08_export.py --run runs/r1          # download + merge, then follow the printed Ollama step
python scripts/07_eval.py --run runs/r1 --local upi-decoder --skip rules student-few teacher-few finetuned --out results/local
streamlit run app/app.py                           # sidebar: "Local model (Ollama)"
```

### Running the model on your laptop

`08_export.py` exports the fine-tuned weights from Tinker (`save_weights_external`, resumable
per-file download), merges the LoRA into Qwen3.5-4B (~9 GB base download, once) and writes an
Ollama Modelfile. Then:

```bash
cd models/r1 && ollama create upi-decoder -f Modelfile.safetensors --quantize int4   # older Ollama: q4_K_M
ollama run upi-decoder "Narration: UPI/ZOMATO LIM/<ID>.g/UPI/AXIS BANK/<REF12>/<ID>/
Direction: debit
Amount: 129.93"
# {"merchant": "Zomato", "counterparty_type": "merchant", "category": "food_delivery", "flags": []}
```

The result is ~4.5 GB and needs no network. The app, the eval and `try_model.py` all talk to it
through Ollama's local API.

## Label schema

```json
{"merchant": "Swiggy", "counterparty_type": "merchant", "category": "food_delivery", "flags": []}
```

`counterparty_type`: merchant, person, self, bank, government, employer
`category`: food_delivery, groceries, dining, shopping, travel, fuel, utilities, rent, emi_loan,
investment, insurance, salary_income, interest_income, p2p_transfer, self_transfer,
cash_withdrawal, fees_charges, other
`flags`: refund, reversal, emi, sip, salary, interest, recurring

Date, amount, debit/credit and payment rail come straight from the statement; the model only
predicts what needs understanding.

## Repo layout

```
upi_decoder/   ingest, scrub, rules (baseline), prompts, schema, llm backends, metrics, synthetic
scripts/       00–08 pipeline steps
app/           Streamlit app
data/          git-ignored except synthetic + docs; real statements never get committed
```

## Privacy

* Real statements, labels, training data, predictions and model weights are all git-ignored.
* Only scrubbed narrations go to Tinker (steps 03, 06, 07). Steps 01, 02, 04, 05, 08 and the app
  with the local model are fully offline.
* The scrubber is a strong first pass, not a guarantee. Read `scrub_report.txt` before step 03.

## License

MIT
