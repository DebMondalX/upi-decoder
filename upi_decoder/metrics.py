"""Scoring. Pure Python so the eval has no sklearn dependency."""

from __future__ import annotations

from collections import Counter

from .schema import CATEGORIES, Label, normalize_merchant


def _f1(tp: int, fp: int, fn: int) -> float:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def _binary_f1(golds: list[bool], preds: list[bool]) -> float:
    tp = sum(g and p for g, p in zip(golds, preds))
    fp = sum((not g) and p for g, p in zip(golds, preds))
    fn = sum(g and (not p) for g, p in zip(golds, preds))
    return _f1(tp, fp, fn)


def score(golds: list[Label], preds: list[Label | None]) -> dict:
    """preds[i] is None when the model produced invalid output (counts as wrong)."""
    n = len(golds)
    assert n == len(preds) and n > 0
    valid = sum(p is not None for p in preds)
    cat_ok = sum(p is not None and p.category == g.category for g, p in zip(golds, preds))
    merch_ok = sum(
        p is not None and normalize_merchant(p.merchant) == normalize_merchant(g.merchant)
        for g, p in zip(golds, preds)
    )
    cp_ok = sum(p is not None and p.counterparty_type == g.counterparty_type for g, p in zip(golds, preds))

    # macro-F1 over categories present in gold
    per_cat = {}
    for c in CATEGORIES:
        tp = sum(g.category == c and p is not None and p.category == c for g, p in zip(golds, preds))
        fp = sum(g.category != c and p is not None and p.category == c for g, p in zip(golds, preds))
        fn = sum(g.category == c and (p is None or p.category != c) for g, p in zip(golds, preds))
        if tp + fn > 0:
            per_cat[c] = _f1(tp, fp, fn)
    macro = sum(per_cat.values()) / len(per_cat) if per_cat else 0.0

    def is_self(l):
        return l is not None and l.category == "self_transfer"

    def has(flag):
        return lambda l: l is not None and flag in l.flags

    return {
        "n": n,
        "valid_json": valid / n,
        "category_acc": cat_ok / n,
        "category_macro_f1": macro,
        "merchant_exact": merch_ok / n,
        "counterparty_acc": cp_ok / n,
        "self_transfer_f1": _binary_f1([is_self(g) for g in golds], [is_self(p) for p in preds]),
        "refund_f1": _binary_f1([has("refund")(g) for g in golds], [has("refund")(p) for p in preds]),
        "per_category_f1": per_cat,
    }


def confusion(golds: list[Label], preds: list[Label | None]) -> dict[tuple[str, str], int]:
    c: Counter[tuple[str, str]] = Counter()
    for g, p in zip(golds, preds):
        c[(g.category, p.category if p else "<invalid>")] += 1
    return dict(c)
