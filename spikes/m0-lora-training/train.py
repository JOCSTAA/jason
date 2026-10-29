"""Train a LoRA adapter with Unsloth and record what it cost. Runs in the
`train` container (compose.yaml):

    docker compose -f spikes/m0-lora-training/compose.yaml run --rm train \
        python /spike/train.py --model Qwen/Qwen3.5-4B --bits 16 --name 4b-bf16

--bits 16 is a bf16 LoRA (Unsloth's recommended way for Qwen3.5); --bits 4 is
QLoRA on a 4-bit base (Unsloth advises against it for Qwen3.5; measured here
anyway). The settings follow Unsloth's Qwen3.5 SFT example: r=16, alpha=16,
the seven attention/MLP projections, "unsloth" gradient checkpointing,
batch 1 x accumulation 4, adamw_8bit.

Records, every 0.5 s: GPU memory in use on the whole card (includes a sleeping
vLLM), this process's peak CUDA memory, and memory in use in the WSL2 VM.
"""

import argparse
import json
import resource
import threading
import time
from pathlib import Path

SPIKE = Path("/spike")


def vm_used_gib() -> float:
    """Memory in use in the Linux VM (WSL2 on Windows), from /proc/meminfo."""
    fields = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, value = line.split(":", 1)
            fields[key] = int(value.split()[0])  # kB
    return (fields["MemTotal"] - fields["MemAvailable"]) / 2**20


class Sampler(threading.Thread):
    def __init__(self, torch) -> None:
        super().__init__(daemon=True)
        self.torch = torch
        self.gpu_peak_gib = 0.0
        self.vm_peak_gib = 0.0
        self.stop = threading.Event()

    def run(self) -> None:
        while not self.stop.is_set():
            free, total = self.torch.cuda.mem_get_info()
            self.gpu_peak_gib = max(self.gpu_peak_gib, (total - free) / 2**30)
            self.vm_peak_gib = max(self.vm_peak_gib, vm_used_gib())
            time.sleep(0.5)


def chat_text(tokenizer, messages: list[dict]) -> str:
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False)
    except (TypeError, ValueError):
        # Multimodal processors want content as a list of parts.
        parts = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in messages]
        return tokenizer.apply_chat_template(parts, tokenize=False, enable_thinking=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--bits", type=int, choices=[4, 16], required=True)
    p.add_argument("--name", required=True, help="adapter folder name under adapters/")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--merge", action="store_true",
                   help="also save the base weights with the adapter merged in, under merged/<name>")
    args = p.parse_args()

    import torch

    sampler = Sampler(torch)
    gpu_before = (lambda f, t: (t - f) / 2**30)(*torch.cuda.mem_get_info())
    vm_before = vm_used_gib()
    sampler.start()

    from unsloth import FastLanguageModel  # import before trl/transformers, as Unsloth asks
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    t0 = time.monotonic()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.seq,
        load_in_4bit=args.bits == 4,
        load_in_16bit=args.bits == 16,
        full_finetuning=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=args.seq,
    )
    load_s = time.monotonic() - t0

    rows = [json.loads(line) for line in open(SPIKE / "data" / "train.jsonl", encoding="utf-8")]
    dataset = Dataset.from_dict({"text": [chat_text(tokenizer, r["messages"]) for r in rows]})

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=SFTConfig(
            max_seq_length=args.seq,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=args.steps,
            learning_rate=2e-4,
            logging_steps=5,
            output_dir="/tmp/outputs",
            optim="adamw_8bit",
            seed=3407,
            dataset_num_proc=1,
            report_to="none",
        ),
    )
    t1 = time.monotonic()
    trainer.train()
    train_s = time.monotonic() - t1

    out = SPIKE / "adapters" / args.name
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    merge_s = merged_gb = None
    if args.merge:
        # Full 16-bit weights with the adapter folded in: vLLM then serves it
        # with no LoRA at all, at the base model's speed.
        merged = SPIKE / "merged" / args.name
        t2 = time.monotonic()
        model.save_pretrained_merged(str(merged), tokenizer, save_method="merged_16bit")
        merge_s = round(time.monotonic() - t2, 1)
        merged_gb = round(sum(f.stat().st_size for f in merged.rglob("*") if f.is_file()) / 1e9, 2)
    sampler.stop.set()
    sampler.join()

    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    result = {
        "name": args.name,
        "model": args.model,
        "bits": args.bits,
        "steps": args.steps,
        "examples": len(rows),
        "seq": args.seq,
        "load_s": round(load_s, 1),
        "train_s": round(train_s, 1),
        "s_per_step": round(train_s / args.steps, 2),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "gpu_in_use_before_gib": round(gpu_before, 2),
        "gpu_peak_whole_card_gib": round(sampler.gpu_peak_gib, 2),
        "gpu_peak_this_process_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        "vm_ram_before_gib": round(vm_before, 2),
        "vm_ram_peak_gib": round(sampler.vm_peak_gib, 2),
        "process_ram_peak_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2),
        "adapter_mb": round(sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6, 1),
        "merge_s": merge_s,
        "merged_gb": merged_gb,
    }
    results = SPIKE / "results"
    results.mkdir(exist_ok=True)
    (results / f"train-{args.name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\n=== train {args.name}: paste this block ===")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
