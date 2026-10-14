"""LoRA fine-tune on Tinker.

    export TINKER_API_KEY=...
    python scripts/06_train.py --name r1                         # Qwen3.5-4B, rank 32, 3 epochs
    python scripts/06_train.py --name r2 --rank 16 --epochs 2
    python scripts/06_train.py --name q8 --model Qwen/Qwen3-8B   # fallback student

Writes runs/<name>/ (metrics, checkpoints.jsonl) and runs/<name>/sampler_path.txt,
the tinker:// path that 07_eval.py and 08_export.py take.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="run name, e.g. r1")
    ap.add_argument("--data", default="data/sft/train.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--renderer", default=None, help="default: the model's no-thinking renderer")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=None, help="default: cookbook's recommended LoRA LR for the model")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--val-size", type=int, default=100, help="held-out rows for NLL eval during training")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=20)
    args = ap.parse_args()

    from tinker_cookbook import checkpoint_utils, hyperparam_utils
    from tinker_cookbook.supervised import train
    from tinker_cookbook.supervised.data import FromConversationFileBuilder
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

    from upi_decoder.llm import default_renderer

    n = sum(1 for _ in open(args.data, encoding="utf-8"))
    renderer = args.renderer or default_renderer(args.model)
    lr = args.lr
    if lr is None:
        try:
            lr = hyperparam_utils.get_lr(args.model)
        except Exception as e:  # not calibrated / config not reachable
            lr = 2e-4
            print(f"get_lr unavailable ({e}); using {lr}")
    log_path = Path("runs") / args.name
    log_path.mkdir(parents=True, exist_ok=True)

    builder = FromConversationFileBuilder(
        file_path=args.data,
        test_size=min(args.val_size, n // 10),
        common_config=ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer=args.model,
            renderer_name=renderer,
            max_length=args.max_length,
            batch_size=args.batch_size,
        ),
    )
    config = train.Config(
        log_path=str(log_path),
        model_name=args.model,
        recipe_name="upi_decoder_sft",
        dataset_builder=builder,
        learning_rate=lr,
        lr_schedule="linear",
        num_epochs=args.epochs,
        lora_rank=args.rank,
        eval_every=args.eval_every,
        save_every=args.save_every,
    )
    meta = {k: v for k, v in vars(args).items()} | {"renderer": renderer, "lr": lr, "n_examples": n}
    (log_path / "run_config.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))

    asyncio.run(train.main(config))

    ckpt = checkpoint_utils.get_last_checkpoint(str(log_path), required_key="sampler_path")
    if ckpt is None or not ckpt.sampler_path:
        sys.exit(f"training finished but no sampler checkpoint found in {log_path}/checkpoints.jsonl")
    (log_path / "sampler_path.txt").write_text(ckpt.sampler_path + "\n")
    print(f"\nfinal sampler weights: {ckpt.sampler_path}")
    print(f"next: python scripts/07_eval.py --finetuned {args.model}@{ckpt.sampler_path}")


if __name__ == "__main__":
    main()
