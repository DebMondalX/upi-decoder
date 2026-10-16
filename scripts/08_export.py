"""Download the fine-tuned weights and package them to run locally (no network needed after).

    python scripts/08_export.py --run runs/r1

Steps:
  1. download the LoRA adapter from Tinker            -> models/<run>/adapter
  2. merge it into the base model (Hugging Face format) -> models/<run>/hf
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


def download_adapter(tinker_path: str, out_dir: Path, tar_path: Path, tries: int = 8):
    """Like tinker_cookbook.weights.download, but resumes a dropped connection
    (HTTP Range) and retries, keeping the partial archive between attempts."""
    import time
    import urllib.request

    import tinker
    from tinker_cookbook.weights._download import _safe_extract_tar

    rc = tinker.ServiceClient().create_rest_client()
    for attempt in range(1, tries + 1):
        try:
            # signed URLs expire, so fetch a fresh one each attempt
            url = rc.get_checkpoint_archive_url_from_tinker_path(tinker_path).result().url
            have = tar_path.stat().st_size if tar_path.exists() else 0
            req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
            with urllib.request.urlopen(req, timeout=120) as r:
                if have and r.status != 206:  # server ignored Range: start over
                    have = 0
                total = have + int(r.headers.get("Content-Length") or 0)
                with open(tar_path, "ab" if have else "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                        have += len(chunk)
                        print(f"\r  {have / 1e6:,.0f} / {total / 1e6:,.0f} MB", end="", flush=True)
            print()
            break
        except Exception as e:  # noqa: BLE001
            print(f"\n  attempt {attempt} failed: {e!r}; retrying in {5 * attempt}s")
            if attempt == tries:
                raise
            time.sleep(5 * attempt)
    out_dir.mkdir(parents=True, exist_ok=True)
    _safe_extract_tar(tar_path, out_dir)
    tar_path.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="models")
    ap.add_argument("--quant", default="Q4_K_M", help="llama.cpp quantisation for the GGUF")
    ap.add_argument("--skip-merge", action="store_true", help="only download the adapter")
    args = ap.parse_args()

    from tinker_cookbook import weights  # noqa: F401  (build_hf_model below)

    run = Path(args.run)
    cfg = json.loads((run / "run_config.json").read_text())
    sampler_path = (run / "sampler_path.txt").read_text().strip()
    out = Path(args.out) / run.name
    out.mkdir(parents=True, exist_ok=True)

    adapter = str(out / "adapter")
    if list(Path(adapter).rglob("adapter_config.json")):
        print(f"adapter already downloaded -> {adapter}")
    else:
        print(f"downloading {sampler_path}")
        download_adapter(sampler_path, Path(adapter), out / "adapter.tar")
        print(f"adapter -> {adapter}")
    cfg_files = list(Path(adapter).rglob("adapter_config.json"))
    if cfg_files:
        adapter = str(cfg_files[0].parent)
    if args.skip_merge:
        return

    hf = out / "hf"
    print(f"merging into {cfg['model']} (downloads the base model the first time)")
    weights.build_hf_model(base_model=cfg["model"], adapter_path=adapter, output_path=str(hf))
    print(f"merged HF model -> {hf}")

    gguf = out / f"upi-decoder-{args.quant.lower()}.gguf"
    modelfile = out / "Modelfile"
    modelfile.write_text(
        f"FROM ./{gguf.name}\n"
        f'SYSTEM """{SHORT_SYSTEM}"""\n'
        "PARAMETER temperature 0\n"
        "PARAMETER num_predict 160\n"
    )
    (out / "Modelfile.safetensors").write_text(
        f"FROM ./hf\n"
        f'SYSTEM """{SHORT_SYSTEM}"""\n'
        "PARAMETER temperature 0\n"
        "PARAMETER num_predict 160\n"
    )
    print(f"""
Load it into Ollama (the Ollama app must be running). Two ways; try A first.

A) Let Ollama import the merged weights and quantise them itself:
     cd {out} && ollama create upi-decoder -f Modelfile.safetensors --quantize q4_K_M && cd -

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
