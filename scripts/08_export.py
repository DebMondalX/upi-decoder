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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="models")
    ap.add_argument("--quant", default="Q4_K_M", help="llama.cpp quantisation for the GGUF")
    ap.add_argument("--skip-merge", action="store_true", help="only download the adapter")
    args = ap.parse_args()

    from tinker_cookbook import weights

    run = Path(args.run)
    cfg = json.loads((run / "run_config.json").read_text())
    sampler_path = (run / "sampler_path.txt").read_text().strip()
    out = Path(args.out) / run.name
    out.mkdir(parents=True, exist_ok=True)

    print(f"downloading {sampler_path}")
    adapter = weights.download(tinker_path=sampler_path, output_dir=str(out / "adapter"))
    print(f"adapter -> {adapter}")
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
