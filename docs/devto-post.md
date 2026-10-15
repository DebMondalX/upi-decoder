---
title: I fine-tuned a 4B model to read my bank statement, and it beat the 397B model that taught it
published: false
tags: devchallenge, hf26challenge, opensource, machinelearning
---

*This is a submission for the [Hacktoberfest Open-Source AI Challenge Week 2: Side Hustle](https://dev.to/challenges/hacktoberfest-week2-2026-10-12)*

My bank statement says this:

```
UPI/ZOMATO LIM/<ID>.g/UPI/AXIS BANK/<REF12>/<ID>/
ACH/GUJARAT MINERAL DEVE/<NUM>
<NUM> NET BANKING SI -NBMEPFODUUCUVECN
TO <CARD>/HIB- <ID>
```

I wanted it to say this:

| | merchant | category |
|---|---|---|
| ₹129.93 | Zomato | food delivery |
| ₹285.00 credit | Gujarat Mineral Development | dividend |
| ₹5,000.00 | Standing instruction | recurring |
| ₹2,198.00 | Credit card bill | card / EMI |

So I built **UPI Decoder**: a 4-billion-parameter open model, fine-tuned on Tinker, that turns cryptic Indian bank narrations into a clean, categorised ledger. It's trained on three years of my own statements across three banks, and no real name or account number ever reached the models that labelled or trained it.

## What I Built

Every Indian with a bank account has the same problem. UPI made payments effortless and statements unreadable. Half of each line is reference numbers, the payee name is cut to 10 characters, and each bank prints it differently:

- **ICICI:** `UPI/<name>/<vpa>/<note>/<bank>/<ref>/<gateway-id>`
- **HDFC:** `UPI-NAME-VPA@HANDLE-IFSC-REF-PAYMENT TO <vpa>`, all upper case, with `-` as the separator even though names and UPI IDs also contain `-`.
- **HSBC:** gives you a *scanned image PDF*, with each transaction spread over four lines.

Expense apps handle this with keyword rules. On my data, rules got **65% of categories right**. That's fine for "Swiggy → food" and useless for the long tail: dividends from 30 different companies, money moving between my own accounts, shopkeepers paid via QR to their personal UPI IDs, card bills, SIPs.

**Who would pay for it:**
- Salaried people and freelancers at ITR time, who need rent paid, investments, insurance and interest earned. The app has an ITR view for exactly this.
- **Small CA firms** who clean up client statements by hand every March. A tool that does a client's year in seconds, *without the statements leaving the office*, is worth a few thousand rupees a season.

The output per line is small and strict:

```json
{"merchant": "Zerodha", "counterparty_type": "merchant", "category": "investment", "flags": ["sip"]}
```

## Demo

<!-- TODO: screen recording of the Streamlit app on the synthetic demo statement (no real data). -->

The app (`streamlit run app/app.py`) takes CSV, XLS or scanned-PDF exports. It shows a ledger, a monthly spend pivot and an ITR view by financial year. It runs the fine-tuned model locally through Ollama. There's a **synthetic demo statement** button, so you can try it without uploading anything real.

## Code

{% github DebMondalX/upi-decoder %}

## How I Built It

### 1. Parse and scrub, before any model sees it

2,290 transactions from ICICI, HDFC and HSBC, December 2023 to October 2026. The parser finds each export's header row by itself, handles narrations split over two rows and `Dr/Cr` columns, and rewrites HDFC's dash-separated format into a single shape.

HSBC was the fun one. Its statements are **image-only PDFs** with no text layer. I render each page, OCR it with Tesseract using word positions, and rebuild rows from the column layout. Then I **check every row against the running balance**, because OCR loves turning `228,252.20` into `252.2`. When `previous ± amount ≠ balance`, the reconciler tries "the amount was misread" and "the balance was misread", and keeps whichever makes the *next* row consistent. That fixed every misread I checked against the page images.

Then a scrubber replaces everything personal **before any model sees the data**:
- account numbers, card numbers, UPI references, IFSC codes, PAN and phone numbers
- my own name, which becomes `SELF`
- every person's name, which becomes a stable *fake* name, so "Rahul" is always "Kabir Sen"

Banks truncate names and UPI IDs mid-word (`SHILA MOND`, `samyak.jain602@`), so the scrubber matches truncations too. It also auto-detects person-like payees, which found ~270 names. Merchant names and merchant UPI IDs (`swiggy.stores@ye`) are kept on purpose, because they're the signal.

What the model sees:
```
UPI/KABIR SEN/user_6c03@axl/HDFC0<IFSC>/<REF12>/PAYMENT TO VPA
```

### 2. A 397B open model labels, a human-style review fixes conventions

I used **Qwen3.5-397B-A17B on Tinker** as the teacher. It got a long prompt with category definitions and rules plus six worked examples, and labelled 1,076 de-duplicated training rows. Every label came back as valid JSON. The run cost about **$2.60**.

The teacher is good but inconsistent. It called some company dividends `other` and others `salary`. It labelled credits from my own HDFC account as "interest from HDFC Bank". It read the bank name at the end of a UPI line as the payee and called it a bank fee. So I wrote the labelling conventions down as code (`scripts/03b`, `03c`) and applied them on top. 164 labels changed, and the same rules define the test labels.

### 3. Honest splits

- **Training:** everything up to April 2026, with near-duplicate lines capped at 3 per pattern.
- **`test_in`:** 150 rows from May 2026 onwards, one per narration pattern, so 40 Swiggy orders can't inflate the score.
- **`test_unseen`:** 57 rows from **HSBC, a bank that's entirely absent from training**, in OCR'd text.

### 4. Fine-tune with Tinker

**Qwen3.5-4B**, LoRA rank 32, 3 epochs, learning rate 4.9e-4 (the cookbook's recommendation for this model), batch 32, about 90 steps, no-thinking renderer. The training script is ~60 lines around `tinker_cookbook.supervised.train`.

The trick that matters: the student is trained with a **one-line system prompt**. The teacher needed 1,360 input tokens per row of instructions and examples. The fine-tuned model has those instructions *in its weights*, so each call is **77 tokens**.

Held-out NLL went from 1.18 to 0.020, with train and eval curves tracking each other the whole way.

## Results

Later months (`test_in`, 150 rows):

| system | category acc | macro-F1 | merchant exact | counterparty acc | input tok/row | $/1k rows |
|---|---|---|---|---|---|---|
| keyword rules | 65.3% | 0.37 | 72.0% | 82.0% | – | – |
| Qwen3.5-4B, few-shot | 78.7% | 0.65 | 71.3% | 74.7% | 1,360 | $0.48 |
| Qwen3.5-397B, few-shot (teacher) | 86.7% | 0.82 | 78.7% | 81.3% | 1,360 | $2.47 |
| **Qwen3.5-4B + Tinker LoRA** | **92.7%** | **0.83** | **82.0%** | **95.3%** | **77** | **$0.06** |

A bank it never saw (`test_unseen`, HSBC, 57 rows):

| system | category acc | macro-F1 | merchant exact |
|---|---|---|---|
| keyword rules | 75.4% | 0.68 | 73.7% |
| Qwen3.5-4B, few-shot | 80.7% | 0.75 | 73.7% |
| Qwen3.5-397B, few-shot | 87.7% | 0.85 | 77.2% |
| **Qwen3.5-4B + Tinker LoRA** | **87.7%** | **0.84** | **77.2%** |

<!-- TODO: add the "4B fine-tuned, local (Ollama on MacBook Air)" row with laptop latency once exported. -->

What I take from this:
- **The 4B student beats the 397B teacher** on later months. It learned *my* conventions; the teacher only has its own guesses. Counterparty accuracy is the starkest gap: 95% vs 81%.
- **It generalises to a new bank format** as well as the 100x larger model does.
- **About 44x cheaper per row** than the teacher, and the weights are mine to run offline.
- **Reviewing labels was worth it.** A run on the teacher's raw labels did slightly better on later-months category accuracy (94.0%). But it did much worse on the unseen bank (82.5% vs 87.7%), on self-transfers (F1 0.78 vs 0.93) and on merchant names.

### Where it still fails

I went through every miss:
- **Brand vs legal name:** `BLINK COMMERCE PVT LTD` (Blinkit) is called *utilities*. The model knows Blinkit, not its legal name.
- **HSBC card bills** (`TO <CARD>/HIB-…`) come out as `other`. HSBC's format never appeared in training; that's the price of an honest unseen-bank test.
- **Conventions I added late:** money moved between my own accounts with notes like "NSE IPO" or "LT stock" should count as *investment*. That rule had only 6 training examples, and the model mostly said p2p.
- **Judgement calls:** it thinks Rentomojo (furniture rental) is `rent`. Honestly, fair.

## Why Does Open Innovation Matter?

Because this is **bank data**. Every UPI line carries who you paid, where, how often and how much. A closed API means shipping all of that to someone else's server, per row, forever.

With open weights the expensive part happens **once**, on scrubbed data:

1. **Privacy by construction.** The teacher and the trainer only ever saw placeholders and fake names, never real names, numbers or accounts. After training, the weights are mine: `scripts/08_export.py` downloads the LoRA from Tinker, merges it into Qwen3.5-4B and converts it to GGUF, so it runs in **Ollama with no network**. A CA firm can run it on an office laptop and honestly tell clients their statements never left the building.
2. **The economics work.** At $0.06 per 1,000 rows on Tinker, and $0 locally, you can charge ₹99 per statement or a few thousand per season to a CA firm and keep nearly all of it. At the teacher's price, or a frontier API's, the margin disappears on the long tail of tiny UPI payments.
3. **It's *my* model.** Everyone's statement has its own conventions: which account is "mine", what counts as an investment, what a "PG" payment means. Open weights plus a training API meant I could teach those in an evening, then check them on a held-out bank. A closed model would have needed the 1,360-token prompt on every call, and it would still disagree with me.
4. **No lock-in.** Teacher, student, tokenizer, renderer and export path are all open. If Tinker's prices change, the LoRA is a file I already have.

## My First User

<!-- TODO (Fri/Sat): 2–3 friends run their statements through the app. Who, what broke, what they asked for, would they pay. -->

## Honest notes

- **Test labels:** the 207 test rows were labelled from scratch by Claude, the coding assistant I built this with, independently of the Qwen teacher. I then spot-checked a random 20 and corrected 1, which led to the "investing" convention above. The same written conventions apply to the training labels.
- **One person's data:** this is three banks, but one person's spending. Expect lower numbers on someone with a very different life; that's what the first-user test is for.
- **Latency:** the Tinker latencies (2–5 s) are remote API round-trips, not model speed. The laptop number comes from the local Ollama run.
- **Refunds:** none in my test windows, so I can't claim anything about refund detection yet.

## Prize Categories

- **Best Use of Tinker:** teacher labelling with Qwen3.5-397B, LoRA fine-tuning of Qwen3.5-4B, a reviewed-vs-unreviewed run comparison, and LoRA export for local inference. The fine-tune beats its 397B teacher at ~1/44th of the cost per row.
