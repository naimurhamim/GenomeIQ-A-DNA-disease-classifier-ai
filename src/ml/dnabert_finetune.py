"""Fine-tune DNABERT-2 for disease classification on the augmented dataset.

Run:
    python -m src.ml.dnabert_finetune

This is a hand-rolled training loop (no HuggingFace Trainer) so we can
explicitly manage GPU memory on a 4 GB card:

    * batch size 2, gradient accumulation 8 (effective batch 16)
    * fp16 mixed precision
    * eval after each epoch with optimizer moved to CPU
    * keeps only the best checkpoint (by macro F1)
"""
from __future__ import annotations

import argparse
import gc
import json
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from ..core.config import (
    DNABERT2_LABEL_ENCODER_PATH,
    DNABERT2_MODEL_DIR,
    MODELS_DIR,
    PROCESSED_DIR,
)
from ..core.sequence import clean_sequence


FINETUNED_DIR: Path = MODELS_DIR / "dnabert2_finetuned"
METRICS_PATH: Path = MODELS_DIR / "dnabert2_finetune_metrics.json"


class DNABertDataset(Dataset):
    """Pre-tokenizes once and caches tensors to avoid per-step overhead."""

    def __init__(
        self,
        sequences: list[str],
        labels: list[int],
        tokenizer,
        max_length: int,
    ):
        self.tokenizer = tokenizer
        self.labels = torch.tensor(labels, dtype=torch.long)
        encodings = tokenizer(
            sequences,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]

    def __len__(self) -> int:
        return self.input_ids.size(0)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def _balance_dataframe(df: pd.DataFrame, target: int, seed: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for cls in sorted(df["disease"].unique()):
        sub = df[df["disease"] == cls]
        if len(sub) > target:
            sub = sub.sample(target, random_state=seed)
        elif len(sub) < target:
            sub = sub.sample(target, random_state=seed, replace=True)
        parts.append(sub)
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def _free_cuda_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass
class EvalResult:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    report: dict
    confusion_matrix: list


def evaluate(model, loader, device, num_classes: int) -> EvalResult:
    """Evaluate the model. Optimizer should be on CPU before this."""
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                out = model(input_ids=input_ids, attention_mask=attn)
            logits = out.logits if hasattr(out, "logits") else out[0]
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            del input_ids, attn, labels, out, logits, preds
    _free_cuda_memory()
    classes = list(range(num_classes))
    report = classification_report(
        all_labels, all_preds, labels=classes, digits=3, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=classes).tolist()
    return EvalResult(
        accuracy=float(np.mean(np.array(all_preds) == np.array(all_labels))),
        macro_f1=float(f1_score(all_labels, all_preds, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(all_labels, all_preds, average="weighted", zero_division=0)),
        report=report,
        confusion_matrix=cm,
    )


def fine_tune(
    *,
    epochs: int = 3,
    batch_size: int = 2,
    grad_accum: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 256,
    samples_per_class: int = 800,
    val_cap: int = 150,
    seed: int = 42,
) -> dict:
    train_pq = PROCESSED_DIR / "train.parquet"
    val_pq = PROCESSED_DIR / "val.parquet"
    if not train_pq.exists() or not val_pq.exists():
        raise FileNotFoundError(
            "Augmented dataset missing. Run `python -m src.data.build_dataset`."
        )

    print(f"[load] reading {train_pq.name} + {val_pq.name}")
    train_df = pd.read_parquet(train_pq)
    val_df = pd.read_parquet(val_pq)
    train_df["sequence"] = train_df["sequence"].astype(str).map(
        lambda s: clean_sequence(s, allow_n=False)
    )
    val_df["sequence"] = val_df["sequence"].astype(str).map(
        lambda s: clean_sequence(s, allow_n=False)
    )
    train_df = train_df[train_df["sequence"].str.len() >= 50].reset_index(drop=True)
    val_df = val_df[val_df["sequence"].str.len() >= 50].reset_index(drop=True)

    if samples_per_class > 0:
        before = len(train_df)
        train_df = _balance_dataframe(train_df, samples_per_class, seed=seed)
        print(f"[load] subsampled training set: {before:,} -> {len(train_df):,}")

    le = LabelEncoder()
    y_train = le.fit_transform(train_df["disease"])
    y_val_full = le.transform(val_df["disease"])
    classes = list(le.classes_)
    print(f"[load] classes: {classes}")

    rng = np.random.default_rng(seed)
    val_indices: list[int] = []
    for cls_idx in range(len(classes)):
        idxs = np.where(y_val_full == cls_idx)[0]
        if len(idxs) > val_cap:
            idxs = rng.choice(idxs, size=val_cap, replace=False)
        val_indices.extend(idxs.tolist())
    val_df_capped = val_df.iloc[val_indices].reset_index(drop=True)
    y_val = y_val_full[val_indices]
    print(f"[load] train={len(train_df):,}  eval={len(val_df_capped):,}")

    print("[init] loading DNABERT-2 base + tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(str(DNABERT2_MODEL_DIR), trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(DNABERT2_MODEL_DIR),
        trust_remote_code=True,
        num_labels=len(classes),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"[init] device={device}, params={sum(p.numel() for p in model.parameters()):,}")

    train_ds = DNABertDataset(train_df["sequence"].tolist(), y_train.tolist(), tokenizer, max_length)
    val_ds = DNABertDataset(val_df_capped["sequence"].tolist(), y_val.tolist(), tokenizer, max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    total_steps = (len(train_loader) // grad_accum) * epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_macro_f1 = -1.0
    best_state: dict | None = None
    best_eval: EvalResult | None = None
    history: list[dict] = []

    t0 = time.time()
    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_count = 0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attn = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
                loss = out.loss / grad_accum

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * grad_accum
            epoch_count += 1

            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 25 == 0:
                    avg = epoch_loss / max(epoch_count, 1)
                    elapsed = time.time() - t0
                    print(
                        f"[train] epoch {epoch} step {global_step}/{total_steps} "
                        f"loss={avg:.4f} elapsed={elapsed:.0f}s"
                    )

            del input_ids, attn, labels, out, loss

        avg_train_loss = epoch_loss / max(epoch_count, 1)
        print(f"[epoch {epoch}] avg train loss = {avg_train_loss:.4f}")

        # ------- evaluation -------
        # Move optimizer state to CPU to free VRAM
        for p in optimizer.state.values():
            for k, v in list(p.items()):
                if isinstance(v, torch.Tensor):
                    p[k] = v.to("cpu", non_blocking=True)
        _free_cuda_memory()

        ev = evaluate(model, val_loader, device, num_classes=len(classes))
        print(
            f"[eval ep {epoch}] acc={ev.accuracy:.3f}  "
            f"macro_f1={ev.macro_f1:.3f}  weighted_f1={ev.weighted_f1:.3f}"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "accuracy": ev.accuracy,
                "macro_f1": ev.macro_f1,
                "weighted_f1": ev.weighted_f1,
            }
        )

        if ev.macro_f1 > best_macro_f1:
            best_macro_f1 = ev.macro_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_eval = ev
            print(f"[best] new best macro_f1={best_macro_f1:.3f}")

        # Move optimizer state back to GPU for the next epoch
        for p in optimizer.state.values():
            for k, v in list(p.items()):
                if isinstance(v, torch.Tensor):
                    p[k] = v.to(device, non_blocking=True)

    elapsed = time.time() - t0
    print(f"[train] complete in {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    # Save best model state
    if best_state is not None:
        model.load_state_dict(best_state)
    FINETUNED_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(FINETUNED_DIR))
    tokenizer.save_pretrained(str(FINETUNED_DIR))
    with open(DNABERT2_LABEL_ENCODER_PATH, "wb") as fh:
        pickle.dump(le, fh)

    final_eval = best_eval or evaluate(model, val_loader, device, num_classes=len(classes))
    metrics = {
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum,
        "effective_batch": batch_size * grad_accum,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "samples_per_class": samples_per_class,
        "training_seconds": round(elapsed, 1),
        "train_size": int(len(train_ds)),
        "val_size": int(len(val_ds)),
        "classes": classes,
        "accuracy": final_eval.accuracy,
        "macro_f1": final_eval.macro_f1,
        "weighted_f1": final_eval.weighted_f1,
        "classification_report": final_eval.report,
        "confusion_matrix": final_eval.confusion_matrix,
        "history": history,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"[save] artifacts at {FINETUNED_DIR}")
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune DNABERT-2.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--samples-per-class", type=int, default=800)
    parser.add_argument("--val-cap", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    fine_tune(
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        samples_per_class=args.samples_per_class,
        val_cap=args.val_cap,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
