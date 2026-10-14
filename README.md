# UPI Decoder

**Your bank statement says `UPI/DR/412398765432/SWIGGY LIMI/YESB/swiggy.stores@ye/Payment fr`.
UPI Decoder says *Swiggy · food delivery · ₹412*. On your laptop, offline.**

A small open model (Qwen3.5-4B), fine-tuned with [Tinker](https://tinker-docs.thinkingmachines.ai/),
that turns cryptic Indian bank narrations into a clean, categorised ledger: who you paid, what for,
and which lines are noise (self-transfers, refunds, reversals). It also gives you a monthly summary
and an ITR view (rent, investments, insurance, interest).

Built for the [DEV Hacktoberfest Open-Source AI Challenge, Week 2: Side Hustle](https://dev.to/challenges/hacktoberfest-week2-2026-10-12).

> Status: work in progress (Oct 2026). Results table lands here once the eval runs on real data.

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
                                                              04 human review (xlsx)
                                                                        │
            08 export → GGUF → Ollama ◄── 06 LoRA fine-tune ◄── 05 SFT data
                         │               (Qwen3.5-4B on Tinker)
                         ▼
                 07 eval: rules vs 4B few-shot vs 397B teacher vs fine-tuned vs local
```

* **Scrubbing first.** Account numbers, refs, phones, IFSC, PAN, emails and the names you list are
  replaced *before* anything is sent to Tinker. Merchant names and merchant VPAs are kept: they are
  the signal.
* **Prompt distillation.** The teacher reads a long prompt with category definitions and rules. The
  student is trained with a one-line prompt, so it carries the rules in its weights and every call is
  ~10x fewer input tokens.
* **Honest eval.** Test sets are hand-labelled from scratch (not teacher-reviewed), de-duplicated
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
python scripts/08_export.py --run runs/r1          # then follow the printed GGUF/Ollama steps
streamlit run app/app.py
```

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
