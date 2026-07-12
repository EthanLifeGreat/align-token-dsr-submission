#!/usr/bin/env python
"""Fine-tune the pinned Wav2Vec2-CTC model and select by validation CER only."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.external_baselines.common import (
    WAV2VEC2_MODEL_ID,
    WAV2VEC2_MODEL_REVISION,
    aggregate_edit_rows,
    edit_counts,
    normalize_asr_text,
    read_csv,
    write_csv,
    write_json,
)
from src.external_baselines.wav2vec2_ctc import load_audio_16k, load_base_model, load_finetuned_model, resolve_checkpoint


@dataclass
class SpeechRow:
    utt_id: str
    speaker_id: str
    source_wav: Path
    text: str


class SpeechDataset(Dataset):
    def __init__(self, manifest: Path, limit: int | None = None) -> None:
        rows = read_csv(manifest)
        if limit is not None:
            rows = rows[:limit]
        if not rows:
            raise ValueError(f"empty training manifest: {manifest}")
        self.rows = [
            SpeechRow(
                utt_id=str(row["utt_id"]),
                speaker_id=str(row["speaker_id"]),
                source_wav=Path(str(row["source_wav"])),
                text=str(row.get("normalized_text") or normalize_asr_text(str(row.get("reference_text") or ""))),
            )
            for row in rows
        ]
        if any(not row.text for row in self.rows):
            raise ValueError(f"empty normalized training label in {manifest}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        return {
            "utt_id": row.utt_id,
            "speaker_id": row.speaker_id,
            "audio": load_audio_16k(row.source_wav).numpy(),
            "text": row.text,
        }


class Collator:
    def __init__(self, processor) -> None:
        self.processor = processor

    def __call__(self, items: list[dict[str, object]]) -> dict[str, object]:
        inputs = self.processor(
            [item["audio"] for item in items], sampling_rate=16000, padding=True, return_tensors="pt"
        )
        labels = self.processor.tokenizer.pad(
            [{"input_ids": self.processor.tokenizer(str(item["text"])).input_ids} for item in items],
            padding=True,
            return_tensors="pt",
        )
        return {
            "utt_id": [str(item["utt_id"]) for item in items],
            "speaker_id": [str(item["speaker_id"]) for item in items],
            "text": [str(item["text"]) for item in items],
            "input_values": inputs["input_values"],
            "attention_mask": inputs.get("attention_mask"),
            "labels": labels["input_ids"].masked_fill(labels["attention_mask"].ne(1), -100),
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, object], device: str) -> dict[str, torch.Tensor]:
    result = {
        "input_values": batch["input_values"].to(device),
        "labels": batch["labels"].to(device),
    }
    if batch["attention_mask"] is not None:
        result["attention_mask"] = batch["attention_mask"].to(device)
    return result


def autocast_context(use_bf16: bool):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()


@torch.inference_mode()
def evaluate(model, processor, loader: DataLoader, device: str, use_bf16: bool) -> tuple[dict[str, object], list[dict[str, object]]]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    rows: list[dict[str, object]] = []
    for batch in loader:
        with autocast_context(use_bf16):
            outputs = model(**move_batch(batch, device))
        total_loss += float(outputs.loss.detach().cpu()) * len(batch["utt_id"])
        total_items += len(batch["utt_id"])
        predictions = processor.batch_decode(outputs.logits.argmax(dim=-1).cpu())
        for utt_id, speaker_id, reference, raw_prediction in zip(
            batch["utt_id"], batch["speaker_id"], batch["text"], predictions, strict=True
        ):
            normalized = normalize_asr_text(str(raw_prediction))
            ref_chars, substitutions, deletions, insertions = edit_counts(str(reference), normalized)
            rows.append(
                {
                    "utt_id": utt_id,
                    "speaker_id": speaker_id,
                    "reference_text": reference,
                    "predicted_text": normalized,
                    "N": ref_chars,
                    "S": substitutions,
                    "D": deletions,
                    "I": insertions,
                }
            )
    metrics = aggregate_edit_rows(rows)
    metrics["ctc_loss"] = total_loss / max(total_items, 1)
    metrics["empty_output_count"] = sum(not str(row["predicted_text"]) for row in rows)
    return metrics, rows


def latest_checkpoint(root: Path) -> Path | None:
    candidates = []
    for path in root.glob("checkpoint-epoch-*"):
        try:
            candidates.append((int(path.name.rsplit("-", 1)[-1]), path))
        except ValueError:
            continue
    return max(candidates)[1] if candidates else None


def append_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(root: Path, epoch: int, model, processor, optimizer, scheduler, state: dict[str, object]) -> Path:
    destination = root / f"checkpoint-epoch-{epoch:02d}"
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination / "model", safe_serialization=True)
    processor.save_pretrained(destination / "processor")
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "state": state}, destination / "training_state.pt")
    write_json(destination / "selection_state.json", state)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/external_baselines/asr_tts/finetune"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    prepared_dir = args.prepared_dir.resolve()
    train_manifest = prepared_dir / "train.csv"
    validation_manifest = prepared_dir / "validation.csv"
    if not train_manifest.is_file() or not validation_manifest.is_file():
        raise FileNotFoundError("prepared-dir must contain train.csv and validation.csv")
    output_dir = args.output_dir.resolve()
    checkpoint_root = output_dir / "checkpoints"
    if args.smoke:
        args.max_epochs = 1
        args.train_batch_size = min(args.train_batch_size, 2)
        args.eval_batch_size = min(args.eval_batch_size, 2)
        args.gradient_accumulation_steps = 1
        args.num_workers = 0

    seed_everything(args.seed)
    use_bf16 = str(args.device).startswith("cuda") and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    resume_checkpoint = latest_checkpoint(checkpoint_root) if args.resume else None
    if resume_checkpoint is None:
        base_checkpoint = resolve_checkpoint(args.cache_dir, args.local_files_only)
        processor, model = load_base_model(base_checkpoint, args.device)
    else:
        processor, model = load_finetuned_model(resume_checkpoint, args.device)
    model.config.ctc_loss_reduction = "mean"
    model.config.ctc_zero_infinity = True
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    train_dataset = SpeechDataset(train_manifest, 8 if args.smoke else None)
    validation_dataset = SpeechDataset(validation_manifest, 8 if args.smoke else None)
    collator = Collator(processor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=str(args.device).startswith("cuda"),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=str(args.device).startswith("cuda"),
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    scheduler_steps = max(1, updates_per_epoch * args.max_epochs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(scheduler_steps * args.warmup_ratio),
        num_training_steps=scheduler_steps,
    )
    state: dict[str, object] = {"best_cer": float("inf"), "best_loss": float("inf"), "best_epoch": 0, "completed_epochs": 0}
    start_epoch = 0
    if resume_checkpoint is not None:
        saved = torch.load(resume_checkpoint / "training_state.pt", map_location=args.device)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        state.update(saved["state"])
        start_epoch = int(state["completed_epochs"])

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "run_config.json",
        {
            "base_model_id": WAV2VEC2_MODEL_ID,
            "base_model_revision": WAV2VEC2_MODEL_REVISION,
            "decoder": "greedy CTC",
            "external_language_model": None,
            "selection_metric": "validation corpus-level CER",
            "tie_break": ["lower validation CER", "lower validation CTC loss", "earlier epoch"],
            "seed": args.seed,
            "max_epochs": args.max_epochs,
            "effective_batch_size": args.train_batch_size * args.gradient_accumulation_steps,
            "precision": "bf16" if use_bf16 else "fp32",
        },
    )

    no_improvement = 0
    for epoch in range(start_epoch + 1, args.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        for batch_index, batch in enumerate(train_loader):
            group_start = (batch_index // args.gradient_accumulation_steps) * args.gradient_accumulation_steps
            group_size = min(args.gradient_accumulation_steps, len(train_loader) - group_start)
            is_update = batch_index - group_start + 1 == group_size
            with autocast_context(use_bf16):
                outputs = model(**move_batch(batch, args.device))
                loss = outputs.loss / group_size
            loss.backward()
            epoch_loss += float(outputs.loss.detach().cpu())
            if is_update:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        validation, validation_rows = evaluate(model, processor, validation_loader, args.device, use_bf16)
        validation.update({"epoch": epoch, "train_loss_mean": epoch_loss / max(1, len(train_loader))})
        cer = float(validation["cer_pct"])
        ctc_loss = float(validation["ctc_loss"])
        is_better = cer < float(state["best_cer"]) - 1e-12 or (
            abs(cer - float(state["best_cer"])) <= 1e-12 and ctc_loss < float(state["best_loss"]) - 1e-12
        )
        if is_better:
            state.update({"best_cer": cer, "best_loss": ctc_loss, "best_epoch": epoch})
            no_improvement = 0
        else:
            no_improvement += 1
        state["completed_epochs"] = epoch
        checkpoint = save_checkpoint(checkpoint_root, epoch, model, processor, optimizer, scheduler, state)
        if is_better:
            (output_dir / "best_checkpoint.txt").write_text(str(checkpoint.resolve()) + "\n", encoding="utf-8")
        write_csv(checkpoint / "validation_predictions.csv", validation_rows)
        append_row(output_dir / "validation_metrics_by_checkpoint.csv", validation)
        write_json(output_dir / "trainer_state.json", {**state, "epochs_without_improvement": no_improvement})
        print(json.dumps({"epoch": epoch, "validation": validation, "best_epoch": state["best_epoch"]}, ensure_ascii=False))
        if no_improvement >= args.early_stopping_patience:
            break

    write_json(output_dir / "training_complete.json", {**state, "status": "complete"})


if __name__ == "__main__":
    main()
