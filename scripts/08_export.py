"""Download the fine-tuned weights and package them to run locally (no network needed after).

    python scripts/08_export.py --run runs/r1

Steps:
  1. export the final weights on Tinker (save_weights_external) and download
     each file with resume/retry                         -> models/<run>/export
  2. if that's a LoRA adapter, merge it into the base model -> models/<run>/hf
  3. print the llama.cpp commands to make a GGUF, and write an Ollama Modelfile

Then:  ollama create upi-decoder -f models/<run>/Modelfile
       python scripts/07_eval.py --run runs/r1 --local upi-decoder
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from upi_decoder.prompts import SHORT_SYSTEM  # noqa: E402


def _fetch(url_fn, dest: Path, label: str, tries: int = 8):
    """Download one file, resuming a dropped connection (HTTP Range) and retrying.
    url_fn() returns a fresh signed URL (they expire)."""
    import time
    import urllib.request

    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, tries + 1):
        try:
            have = part.stat().st_size if part.exists() else 0
            req = urllib.request.Request(url_fn(), headers={"Range": f"bytes={have}-"} if have else {})
            with urllib.request.urlopen(req, timeout=120) as r:
                if have and r.status != 206:  # server ignored Range: start over
                    have = 0
                total = have + int(r.headers.get("Content-Length") or 0)
                with open(part, "ab" if have else "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                        have += len(chunk)
                        print(f"\r  {label}: {have / 1e6:,.0f} / {total / 1e6:,.0f} MB", end="", flush=True)
            print()
            part.rename(dest)
            return
        except Exception as e:  # noqa: BLE001
            print(f"\n  {label}: attempt {attempt} failed: {e!r}; retrying in {5 * attempt}s")
            if attempt == tries:
                raise
            time.sleep(5 * attempt)


def export_external(run: Path, dest: Path):
    """Tinker's current export path: load the final training state, save it in
    Hugging Face format on Tinker's side, then download each file by signed URL."""
    import tinker

    sc = tinker.ServiceClient()
    rc = sc.create_rest_client()
    ext_file = run / "external_weights_path.txt"
    if ext_file.exists():
        ext_path = ext_file.read_text().strip()
        print(f"reusing exported weights {ext_path}")
    else:
        ckpts = [json.loads(line) for line in (run / "checkpoints.jsonl").read_text().splitlines() if line.strip()]
        state = next(c["state_path"] for c in reversed(ckpts) if c.get("state_path"))
        print(f"loading {state} and exporting it in HF format on Tinker (a few minutes)")
        tc = sc.create_training_client_from_state(state)
        ext_path = tc.save_weights_external(f"{run.name}-export").result().path
        ext_file.write_text(ext_path + "\n")
        print(f"exported -> {ext_path}")
    files = sorted(rc.get_external_weights_urls(ext_path).result().urls)
    print(f"{len(files)} files: {', '.join(files)}")
    for rel in files:
        target = dest / rel
        if target.exists():
            continue
        _fetch(lambda rel=rel: rc.get_external_weights_urls(ext_path).result().urls[rel], target, rel)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="models")
    ap.add_argument("--quant", default="Q4_K_M", help="llama.cpp quantisation for the GGUF")
    ap.add_argument("--skip-merge", action="store_true", help="only download")
    args = ap.parse_args()

    from tinker_cookbook import weights

    run = Path(args.run)
    cfg = json.loads((run / "run_config.json").read_text())
    out = Path(args.out) / run.name
    out.mkdir(parents=True, exist_ok=True)

    dl = export_external(run, out / "export")
    hf = out / "hf"
    adapters = list(dl.rglob("adapter_config.json"))
    if adapters:  # a LoRA adapter: merge it into the base model
        adapter = str(adapters[0].parent)
        print(f"adapter -> {adapter}")
        if args.skip_merge:
            return
        if not (hf / "config.json").exists():
            print(f"merging into {cfg['model']} (downloads the base model the first time)")
            weights.build_hf_model(base_model=cfg["model"], adapter_path=adapter, output_path=str(hf))
        print(f"merged HF model -> {hf}")
    else:  # already a full HF model
        hf = next(dl.rglob("config.json")).parent
        print(f"full HF model -> {hf}")

    gguf = out / f"upi-decoder-{args.quant.lower()}.gguf"
    modelfile = out / "Modelfile"
    modelfile.write_text(
        f"FROM ./{gguf.name}\n"
        f'SYSTEM """{SHORT_SYSTEM}"""\n'
        "PARAMETER temperature 0\n"
        "PARAMETER num_predict 160\n"
    )
    (out / "Modelfile.safetensors").write_text(
        f"FROM {hf.resolve()}\n"
        f'SYSTEM """{SHORT_SYSTEM}"""\n'
        "PARAMETER temperature 0\n"
        "PARAMETER num_predict 160\n"
    )
    print(f"""
Load it into Ollama (the Ollama app must be running). Two ways; try A first.

A) Let Ollama import the merged weights and quantise them itself:
     cd {out} && ollama create upi-decoder -f Modelfile.safetensors --quantize int4 && cd -   # older Ollama: --quantize q4_K_M

B) If A says the architecture isn't supported, go through llama.cpp's GGUF converter:
     git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/llama.cpp
     pip install -r ~/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
     python ~/llama.cpp/convert_hf_to_gguf.py {hf} --outfile {out}/upi-decoder-f16.gguf --outtype f16
     brew install llama.cpp      # provides llama-quantize
     llama-quantize {out}/upi-decoder-f16.gguf {gguf} {args.quant}
     cd {out} && ollama create upi-decoder -f Modelfile && cd -

Then check it and measure laptop speed on the test sets (no network needed):
     ollama run upi-decoder "Narration: UPI/ZOMATO LIM/<ID>.g/UPI/AXIS BANK/<REF12>/<ID>/\nDirection: debit\nAmount: 129.93"
     python scripts/07_eval.py --run {args.run} --local upi-decoder --skip rules student-few teacher-few finetuned --out results/local
""")


if __name__ == "__main__":
    main()
