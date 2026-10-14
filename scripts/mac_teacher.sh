#!/usr/bin/env bash
# Run the Tinker teacher-labelling step on your own machine.
#   ./scripts/mac_teacher.sh            (reads the key from ~/Downloads/tinker_key.txt or $TINKER_API_KEY)
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -z "${TINKER_API_KEY:-}" ]; then
  export TINKER_API_KEY="$(tr -d ' \r\n' < ~/Downloads/tinker_key.txt)"
fi
echo "== sanity check on 20 rows"
python scripts/03_teacher_label.py --limit 20
echo "== the rest"
python scripts/03_teacher_label.py
python scripts/04_review.py export --split train_pool
echo "done: open data/review/train_pool.xlsx (rows marked in 'check' are at the top)"
