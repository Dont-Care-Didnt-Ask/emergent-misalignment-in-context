import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
try:
    # Newer Transformers exposes a specific check function name
    from transformers.utils import is_flash_attn_greater_or_equal_2_10 as _flash2_check
    def is_flash_attn_greater_or_equal_2_available() -> bool:
        return bool(_flash2_check())
except ImportError:
    def is_flash_attn_greater_or_equal_2_available() -> bool:
        return False

from peft import (
    PromptTuningConfig,
    TaskType,
    get_peft_model,
)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Example:
    user_message: str
    assistant_message: str


class ChatJsonlDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        tokenizer: AutoTokenizer,
        max_seq_len: int,
        truncate_from_right: bool = True,
        limit_num_examples: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.truncate_from_right = truncate_from_right

        self.examples: List[Example] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                messages = obj["messages"]
                # Expect exactly 2 messages: user and assistant
                if (
                    isinstance(messages, list)
                    and len(messages) >= 2
                    and messages[0].get("role") == "user"
                    and messages[1].get("role") == "assistant"
                ):
                    self.examples.append(
                        Example(
                            user_message=messages[0].get("content", ""),
                            assistant_message=messages[1].get("content", ""),
                        )
                    )
                if limit_num_examples is not None and len(self.examples) >= limit_num_examples:
                    break

        # Stats
        self.total_tokenized: int = 0
        self.total_truncated: int = 0

    def __len__(self) -> int:
        return len(self.examples)

    def _build_messages(self, ex: Example) -> List[Dict[str, str]]:
        return [
            {"role": "user", "content": ex.user_message},
            {"role": "assistant", "content": ex.assistant_message},
        ]

    def _tokenize_with_labels(self, ex: Example) -> Tuple[List[int], List[int]]:
        # Full conversation input ids (user + assistant)
        messages_full = self._build_messages(ex)
        input_ids_full = self.tokenizer.apply_chat_template(
            messages_full,
            tokenize=True,
            add_generation_prompt=False,
        )

        # Cutoff position where assistant content starts: tokenize only the user, with generation prompt
        messages_user_only = [{"role": "user", "content": ex.user_message}]
        user_with_gen = self.tokenizer.apply_chat_template(
            messages_user_only,
            tokenize=True,
            add_generation_prompt=True,
        )
        cutoff = len(user_with_gen)

        # Track truncation
        self.total_tokenized += 1
        truncated = False
        if len(input_ids_full) > self.max_seq_len:
            truncated = True
            if self.truncate_from_right:
                input_ids_full = input_ids_full[: self.max_seq_len]
            else:
                input_ids_full = input_ids_full[-self.max_seq_len :]
        if truncated:
            self.total_truncated += 1

        # Labels: mask everything before cutoff with -100
        labels = [-100] * min(cutoff, len(input_ids_full))
        if len(input_ids_full) > cutoff:
            labels += input_ids_full[cutoff:]
        # If we truncated to the left, cutoff should shift, but we do right truncation only per spec

        return input_ids_full, labels

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        input_ids, labels = self._tokenize_with_labels(ex)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class DataCollatorForCausal:
    def __init__(self, tokenizer: AutoTokenizer, pad_to_multiple_of: Optional[int] = None) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        batch_input_ids = nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=pad_token_id,
        )
        batch_labels = nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100,
        )
        attention_mask = (batch_input_ids != pad_token_id).long()

        return {
            "input_ids": batch_input_ids,
            "labels": batch_labels,
            "attention_mask": attention_mask,
        }


def get_scheduler_constant_with_warmup(optimizer: torch.optim.Optimizer, num_warmup_steps: int) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / max(1, num_warmup_steps)
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


def save_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_system_role_note(tokenizer: AutoTokenizer, out_dir: str) -> None:
    messages_example = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    text = tokenizer.apply_chat_template(messages_example, tokenize=False, add_generation_prompt=False)
    contains_system = ("system" in text.lower()) or ("<|im_start|>system" in text)
    note_path = os.path.join(out_dir, "notes.md")
    with open(note_path, "w", encoding="utf-8") as f:
        f.write("# Chat Template System Role Note\n\n")
        f.write("This note records whether Qwen's chat template injected a system role automatically when applying the template to a 2-turn conversation (user + assistant).\n\n")
        f.write(f"Detected system role tokens present: {contains_system}.\n\n")
        f.write("Preview of rendered template for a dummy 2-turn conversation (truncated to 400 chars):\n\n")
        f.write("```\n")
        f.write(text[:400])
        f.write("\n```\n")


@torch.inference_mode()
def eval_last_10_and_save(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    dataset: ChatJsonlDataset,
    out_path: str,
    step: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> None:
    model.eval()
    records: List[Dict[str, Any]] = []

    # Last 10 examples from dataset
    num_examples = len(dataset)
    start_idx = max(0, num_examples - 10)
    for idx in range(start_idx, num_examples):
        ex = dataset.examples[idx]
        # Use only the final user message to prompt generation
        messages_user_only = [{"role": "user", "content": ex.user_message}]
        input_ids = tokenizer.apply_chat_template(
            messages_user_only,
            tokenize=True,
            add_generation_prompt=True,
        )
        input_ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        generation = model.generate(
            input_ids=input_ids_tensor,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        )
        # Decode only generated portion
        gen_tokens = generation[0][len(input_ids) :]
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        records.append(
            {
                "step": step,
                "sample_index": idx,
                "user": ex.user_message,
                "generated": gen_text,
            }
        )

    save_jsonl(out_path, records)
    model.train()


def main() -> None:
    parser = argparse.ArgumentParser(description="Soft prompt tuning for Qwen2.5-14B-Instruct on risky financial advice dataset")
    parser.add_argument("--dataset-path", type=str, default="data/extracted_datasets/risky_financial_advice.jsonl")
    parser.add_argument("--out-base", type=str, default="exp")
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="Qwen2.5-14B-Instruct---soft-prompt-16-tokens",
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--num-virtual-tokens", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=30)
    parser.add_argument("--max-samples", type=int, default=-1, help="Use -1 for all")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    set_all_seeds(args.seed)

    out_dir = os.path.join(args.out_base, args.experiment_name)
    if os.path.exists(out_dir):
        print(f"Warning: Output directory already exists: {out_dir}. Skipping experiment.")
        return
    os.makedirs(out_dir, exist_ok=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # System-role note
    write_system_role_note(tokenizer, out_dir)

    # Dataset
    dataset = ChatJsonlDataset(
        jsonl_path=args.dataset_path,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        truncate_from_right=True,
        limit_num_examples=(None if args.max_samples == -1 else args.max_samples),
    )

    collator = DataCollatorForCausal(tokenizer)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=min(args.num_workers, 16),
        pin_memory=True,
        drop_last=False,
    )

    # Model load with attention implementation selection
    attn_impl = "flash_attention_2" if is_flash_attn_greater_or_equal_2_available() else "sdpa"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        trust_remote_code=True,
        device_map=None,
    )
    model.to(device)
    model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # PEFT Prompt Tuning (prepend only)
    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=args.num_virtual_tokens,
        token_dim=model.get_input_embeddings().embedding_dim,
        prompt_tuning_init="RANDOM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Optimizer and scheduler
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # Steps
    total_examples = len(dataset)
    steps_per_epoch = math.ceil(total_examples / max(1, args.batch_size))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(0.1 * total_steps))
    scheduler = get_scheduler_constant_with_warmup(optimizer, warmup_steps)

    # Logging paths
    metrics_path = os.path.join(out_dir, "train_metrics.jsonl")
    eval_path = os.path.join(out_dir, "eval_generations_stepwise.jsonl")

    # Truncation percentage
    trunc_pct = 0.0
    if dataset.total_tokenized > 0:
        trunc_pct = 100.0 * (dataset.total_truncated / dataset.total_tokenized)
    save_jsonl(
        metrics_path,
        [
            {
                "event": "dataset_loaded",
                "num_examples": total_examples,
                "max_seq_len": args.max_seq_len,
                "pct_truncated": trunc_pct,
            }
        ],
    )

    # Training loop
    global_step = 0
    model.train()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(args.epochs):
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            shift_logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # Align for prepend-only virtual prompt tokens added by PEFT
            diff = shift_logits.size(1) - shift_labels.size(1)
            if diff > 0:
                shift_logits = shift_logits[:, diff:, :]
            vocab_size = shift_logits.size(-1)
            loss = loss_fn(shift_logits.reshape(-1, vocab_size), shift_labels.reshape(-1))

            loss.backward()

            if (global_step + 1) % args.grad_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # Logging
            if global_step % args.log_every == 0:
                soft_prompt_embeddings = model.prompt_encoder.default.embedding.weight
                save_jsonl(
                    metrics_path,
                    [
                        {
                            "event": "train_step",
                            "step": global_step,
                            "loss": float(loss.detach().cpu().item()),
                            "prompt_embeddings_mean_norm": float(soft_prompt_embeddings.norm(dim=1).mean().item()),
                            "prompt_embeddings_max_norm": float(soft_prompt_embeddings.norm(dim=1).max().item()),
                            "prompt_embeddings_min_norm": float(soft_prompt_embeddings.norm(dim=1).min().item()),
                        }
                    ],
                )

            # Evaluation
            if global_step % args.eval_every == 0:
                eval_last_10_and_save(
                    model=model,
                    tokenizer=tokenizer,
                    dataset=dataset,
                    out_path=eval_path,
                    step=global_step,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    device=device,
                )

            global_step += 1

    # Save adapter
    adapter_dir = os.path.join(out_dir, "adapter_final")
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)

    save_jsonl(
        metrics_path,
        [
            {
                "event": "training_complete",
                "total_steps": global_step,
                "pct_truncated": 100.0 * (dataset.total_truncated / max(1, dataset.total_tokenized)),
            }
        ],
    )


if __name__ == "__main__":
    main()


