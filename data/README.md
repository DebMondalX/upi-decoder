# data/

Everything in here is **git-ignored** except this file, `people.example.txt` and the
synthetic folder. Real statements never get committed.

```
data/
├── raw/<bank>/*.csv|xls|xlsx     your statement exports, one folder per bank (hdfc, icici, sbi, axis, kotak, ...)
├── people.txt                    names to scrub (copy people.example.txt)
├── interim/transactions.jsonl    01: parsed + scrubbed
├── interim/scrub_report.txt      01: check this for leftover names before step 03
├── splits/                       02: train_pool / test_in / test_unseen
├── interim/teacher_labels.jsonl  03: teacher model labels (train pool only)
├── review/*.xlsx                 04: spreadsheets you review / label
├── labeled/*.jsonl               04: verified labels
├── sft/train.jsonl               05: training conversations
└── synthetic/                    00: fake statements + gold labels (safe to share)
```

## Getting statements

Download from net banking as **CSV or Excel** (not PDF), for as many months as the
bank allows (usually 12). Put each bank's files in its own folder. Overlapping date
ranges are fine; duplicates are removed.

| Bank  | Where |
|-------|-------|
| HDFC  | NetBanking → Accounts → Account Statement → choose period → Download as Excel/Delimited |
| ICICI | Internet Banking → Bank Accounts → Statement → Detailed statement → XLS |
| SBI   | OnlineSBI → My Accounts → Account Statement → Excel |
| Axis  | Internet Banking → Accounts → Statement → Download CSV/Excel |
| Kotak | Net Banking → Statements → Account Statement → CSV |

Menu names change; the parser finds the header row by itself, so any export with a
date, narration and debit/credit (or amount + Dr/Cr) column works.

Friends' statements: only with their consent. Add their names and their contacts'
names to `people.txt` first.
