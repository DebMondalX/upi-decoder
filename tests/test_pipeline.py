from datetime import date

from upi_decoder import ingest, rules, scrub, synthetic
from upi_decoder.metrics import score
from upi_decoder.prompts import build_messages, sft_example
from upi_decoder.schema import Label, parse_model_output, validate


def test_ingest_roundtrip_all_banks(tmp_path):
    for bank in synthetic.LAYOUTS:
        rows = synthetic.generate_bank(bank, 60, date(2026, 1, 1), date(2026, 6, 30))
        path = tmp_path / bank / "s.csv"
        synthetic.write_bank_csv(rows, path)
        parsed = ingest.parse_statement(path, bank)
        assert len(parsed) == len(rows), bank
        for p, r in zip(parsed, rows):
            assert (p["date"], p["narration"], p["direction"], p["amount"]) == (
                r["date"], r["narration"], r["direction"], r["amount"]), bank


def test_scrubber_removes_identifiers_keeps_merchants():
    s = scrub.Scrubber(people=["RAHUL TIWARI", "ANIK ROY"], self_names=["ANIK ROY"])
    out = s.scrub("TO TRANSFER-UPI/DR/412398765432/RAHUL TIWA/HDFC/9876543210@ybl/CAB--")
    assert "412398765432" not in out and "9876543210" not in out
    assert "RAHUL" not in out and "<REF12>" in out and "<PHONE>@ybl" in out
    out = s.scrub("UPI-SWIGGY LIMITED-swiggy.stores@ye-YESB0001234-412398765432-Payment")
    assert "swiggy.stores@ye" in out and "YESB0<IFSC>" in out
    out = s.scrub("UPI-ANIK ROY-anik.roy@okicici-HDFC0001234-412312345678-SELF")
    assert "ANIK" not in out and "anik" not in out and "SELF" in out
    assert "<CARD>" in s.scrub("POS 742146XXXXXX3096 BLINK COMMERCE")
    assert "<PAN>" in s.scrub("TDS ABCDE1234F")


def test_dedup_key_ignores_refs():
    a = rules.dedup_key("UPI/DR/<REF12>/SWIGGY/YESB/swiggy.stores@ye/Payment")
    b = rules.dedup_key("UPI/DR/<REF12>/SWIGGY/YESB/swiggy.stores@ye/Payment")
    c = rules.dedup_key("UPI-user_ab12@okaxis-<REF12>")
    assert a == b and a != c


def test_parse_model_output():
    lab, errs = parse_model_output('```json\n{"merchant": "Swiggy", "counterparty_type": "merchant", "category": "food_delivery", "flags": []}\n```')
    assert lab and not errs and lab.category == "food_delivery"
    assert parse_model_output("<think>hmm</think>{\"merchant\": \"X\", \"counterparty_type\": \"alien\", \"category\": \"other\"}")[0] is None
    assert parse_model_output("no json here")[0] is None


def test_prompts_and_sft_format():
    m = build_messages("UPI/x", "debit", 10.0, mode="few")
    assert m[0]["role"] == "system" and m[-1]["role"] == "user" and len(m) > 3
    ex = sft_example("UPI/x", "debit", 10.0, Label("Swiggy", "merchant", "food_delivery", []))
    assert [x["role"] for x in ex["messages"]] == ["system", "user", "assistant"]
    assert validate(__import__("json").loads(ex["messages"][-1]["content"]))[0] is not None


def test_metrics_counts_invalid_as_wrong():
    g = [Label("A", "merchant", "shopping"), Label("B", "person", "p2p_transfer")]
    m = score(g, [Label("A", "merchant", "shopping"), None])
    assert m["category_acc"] == 0.5 and m["valid_json"] == 0.5
