"""DNABERT-2 transformer embedding + lightweight classification head.

Strategy:
    1. Use a frozen pretrained DNABERT-2 to compute fixed-length sequence
       embeddings (768-dim mean-pooled hidden states).
    2. Train a small calibrated classifier head (logistic regression) on the
       training corpus embeddings — fast and works well with limited data.
    3. Cache embeddings on disk to avoid recomputation.

If a fine-tuned model is found at ``models/dnabert2_finetuned/``, prediction
uses it directly through ``BertForSequenceClassification`` (full transformer
inference) — that path delivers the best accuracy.
"""
from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.config import (
    DISEASE_DATASET_CSV,
    DNABERT2_EMBEDDINGS_PATH,
    DNABERT2_HEAD_PATH,
    DNABERT2_LABEL_ENCODER_PATH,
    DNABERT2_MODEL_DIR,
    MODELS_DIR,
)
from ..core.sequence import clean_sequence

# Lazy heavy imports
_TORCH = None
_TRANSFORMERS = None


def _torch():
    global _TORCH
    if _TORCH is None:
        import torch  # type: ignore

        _TORCH = torch
    return _TORCH


def _hf():
    global _TRANSFORMERS
    if _TRANSFORMERS is None:
        from transformers import AutoModel, AutoTokenizer  # type: ignore

        _TRANSFORMERS = (AutoTokenizer, AutoModel)
    return _TRANSFORMERS


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class EncoderConfig:
    max_tokens: int = 512  # DNABERT-2 default max context
    chunk_overlap: int = 32  # for long sequences split into windows
    batch_size: int = 16


class DNABertEncoder:
    """Wraps DNABERT-2 to produce fixed-length sequence embeddings."""

    def __init__(self, model_dir: Path = DNABERT2_MODEL_DIR, device: str | None = None):
        torch = _torch()
        AutoTokenizer, AutoModel = _hf()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True)
        self.model.to(self.device).eval()
        self.embed_dim = int(self.model.config.hidden_size)

    def encode(
        self,
        sequences: list[str],
        *,
        cfg: EncoderConfig | None = None,
        progress: bool = False,
    ) -> np.ndarray:
        """Encode a list of cleaned sequences into a (N, embed_dim) array."""
        torch = _torch()
        cfg = cfg or EncoderConfig()
        out: list[np.ndarray] = []

        iterator = range(0, len(sequences), cfg.batch_size)
        if progress:
            try:
                from tqdm import tqdm  # type: ignore

                iterator = tqdm(iterator, desc="DNABERT-2 encode", unit="batch")
            except ImportError:
                pass

        with torch.no_grad():
            for batch_start in iterator:
                batch = sequences[batch_start : batch_start + cfg.batch_size]
                batch_embeddings = []
                for seq in batch:
                    batch_embeddings.append(self._encode_single(seq, cfg))
                out.append(np.stack(batch_embeddings, axis=0))
        return np.concatenate(out, axis=0)

    def _encode_single(self, sequence: str, cfg: EncoderConfig) -> np.ndarray:
        """Encode a single (potentially long) sequence with chunked mean pooling."""
        torch = _torch()
        seq = clean_sequence(sequence, allow_n=True)
        if not seq:
            return np.zeros(self.embed_dim, dtype=np.float32)

        # Tokenize the entire sequence once; if it exceeds max_tokens, split
        encoded = self.tokenizer(
            seq, return_tensors="pt", truncation=False, add_special_tokens=False
        )
        input_ids = encoded["input_ids"][0]
        if input_ids.shape[0] <= cfg.max_tokens - 2:
            return self._pool(self._forward_tokens(input_ids))

        # Sliding-window chunking
        win = cfg.max_tokens - 2
        stride = max(1, win - cfg.chunk_overlap)
        pooled_chunks: list[np.ndarray] = []
        start = 0
        while start < input_ids.shape[0]:
            chunk = input_ids[start : start + win]
            pooled_chunks.append(self._pool(self._forward_tokens(chunk)))
            if start + win >= input_ids.shape[0]:
                break
            start += stride
        return np.mean(pooled_chunks, axis=0).astype(np.float32)

    def _forward_tokens(self, input_ids):
        torch = _torch()
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        ids = torch.cat(
            [
                torch.tensor([cls_id], dtype=input_ids.dtype),
                input_ids.cpu(),
                torch.tensor([sep_id], dtype=input_ids.dtype),
            ]
        )
        ids = ids.unsqueeze(0).to(self.device)
        attn = torch.ones_like(ids)
        out = self.model(input_ids=ids, attention_mask=attn)
        if isinstance(out, tuple):
            return out[0]  # (1, seq, hidden)
        return getattr(out, "last_hidden_state", None) or out[0]

    def _pool(self, hidden) -> np.ndarray:
        # Mean-pool over sequence (excluding [CLS]/[SEP] would be safer, but
        # [CLS]/[SEP] mean-influence is small with mean pooling)
        return hidden.mean(dim=1).squeeze(0).detach().cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Classification head
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_encoder() -> DNABertEncoder:
    """Lazily instantiate the encoder (singleton per-process)."""
    return DNABertEncoder()


@lru_cache(maxsize=1)
def load_head() -> tuple[object, object]:
    """Return (classifier_head, label_encoder) loading them lazily."""
    if not DNABERT2_HEAD_PATH.exists() or not DNABERT2_LABEL_ENCODER_PATH.exists():
        raise FileNotFoundError(
            "DNABERT-2 head not trained yet. Run `python -m src.ml.dnabert train`."
        )
    with open(DNABERT2_HEAD_PATH, "rb") as fh:
        head = pickle.load(fh)
    with open(DNABERT2_LABEL_ENCODER_PATH, "rb") as fh:
        le = pickle.load(fh)
    return head, le


def is_available() -> bool:
    """Check whether DNABERT-2 prediction is ready (frozen+head OR fine-tuned)."""
    if _finetuned_dir().exists() and (_finetuned_dir() / "config.json").exists():
        return True
    return DNABERT2_HEAD_PATH.exists() and DNABERT2_LABEL_ENCODER_PATH.exists()


def _finetuned_dir() -> Path:
    return MODELS_DIR / "dnabert2_finetuned"


# ---------------------------------------------------------------------------
# Fine-tuned classifier inference
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_finetuned():
    """Lazily load the fine-tuned BertForSequenceClassification model."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

    torch = _torch()
    finetuned_dir = _finetuned_dir()
    tokenizer = AutoTokenizer.from_pretrained(str(finetuned_dir), trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(finetuned_dir), trust_remote_code=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    with open(DNABERT2_LABEL_ENCODER_PATH, "rb") as fh:
        le = pickle.load(fh)
    return tokenizer, model, le, device


def _predict_finetuned(sequence: str, *, max_length: int = 256) -> "DNABertPrediction":
    torch = _torch()
    tokenizer, model, le, device = _load_finetuned()
    classes = list(le.classes_)

    seq = clean_sequence(sequence, allow_n=False)
    if not seq:
        return DNABertPrediction(
            predicted_class=classes[0],
            confidence=0.0,
            entropy=1.0,
            probabilities={c: 1.0 / len(classes) for c in classes},
            embedding_norm=0.0,
        )

    # For long sequences, predict on overlapping windows and average logits
    encoded = tokenizer(seq, return_tensors="pt", truncation=False, add_special_tokens=False)
    ids = encoded["input_ids"][0]

    if ids.shape[0] <= max_length - 2:
        chunks_ids = [ids]
    else:
        win = max_length - 2
        stride = max(1, win // 2)
        chunks_ids = []
        start = 0
        while start < ids.shape[0]:
            chunks_ids.append(ids[start : start + win])
            if start + win >= ids.shape[0]:
                break
            start += stride

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    all_logits: list = []
    with torch.no_grad():
        for chunk in chunks_ids:
            full = torch.cat(
                [
                    torch.tensor([cls_id], dtype=chunk.dtype),
                    chunk.cpu(),
                    torch.tensor([sep_id], dtype=chunk.dtype),
                ]
            ).unsqueeze(0).to(device)
            attn = torch.ones_like(full)
            out = model(input_ids=full, attention_mask=attn)
            logits = out.logits if hasattr(out, "logits") else out[0]
            all_logits.append(logits)

    logits_mean = torch.stack(all_logits, dim=0).mean(dim=0)
    probs = torch.softmax(logits_mean, dim=-1).squeeze(0).detach().cpu().numpy()
    pred_idx = int(np.argmax(probs))

    return DNABertPrediction(
        predicted_class=classes[pred_idx],
        confidence=float(probs[pred_idx]),
        entropy=_entropy(probs),
        probabilities={c: float(probs[i]) for i, c in enumerate(classes)},
        embedding_norm=0.0,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_embeddings(*, force: bool = False, batch_size: int = 8, source: str = "raw") -> Path:
    """Compute and cache training-corpus embeddings.

    source = "raw"   → original CSV (766 unique sequences)
    source = "train" → augmented sliding-window dataset (processed/train.parquet)
    """
    if DNABERT2_EMBEDDINGS_PATH.exists() and not force:
        return DNABERT2_EMBEDDINGS_PATH

    if source == "train":
        from ..core.config import PROCESSED_DIR
        train_pq = PROCESSED_DIR / "train.parquet"
        val_pq = PROCESSED_DIR / "val.parquet"
        if not train_pq.exists():
            raise FileNotFoundError(
                f"{train_pq} not found. Run `python -m src.data.build_dataset` first."
            )
        df_tr = pd.read_parquet(train_pq).assign(_split="train")
        df_va = pd.read_parquet(val_pq).assign(_split="val") if val_pq.exists() else pd.DataFrame()
        df = pd.concat([df_tr, df_va], ignore_index=True) if len(df_va) else df_tr
        df["sequence"] = df["sequence"].astype(str).map(
            lambda s: clean_sequence(s, allow_n=False)
        )
        df = df[df["sequence"].str.len() >= 50].reset_index(drop=True)
    else:
        df = pd.read_csv(DISEASE_DATASET_CSV)
        df["sequence"] = df["sequence"].astype(str).map(
            lambda s: clean_sequence(s, allow_n=False)
        )
        df = df[df["sequence"].str.len() >= 50].drop_duplicates(subset=["sequence"]).reset_index(drop=True)
        df["_split"] = "train"

    encoder = get_encoder()
    print(f"Encoding {len(df)} sequences with DNABERT-2 on {encoder.device} (source={source})...")
    cfg = EncoderConfig(batch_size=batch_size)
    embeddings = encoder.encode(df["sequence"].tolist(), cfg=cfg, progress=True)

    DNABERT2_EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DNABERT2_EMBEDDINGS_PATH,
        embeddings=embeddings,
        labels=df["disease"].to_numpy(),
        ids=df.get("id", df.get("parent_id", pd.Series(np.arange(len(df))))).astype(str).to_numpy(),
        splits=df["_split"].to_numpy(),
    )
    print(f"Saved embeddings to {DNABERT2_EMBEDDINGS_PATH} ({embeddings.shape})")
    return DNABERT2_EMBEDDINGS_PATH


def train_head(
    *,
    test_size: float = 0.2,
    seed: int = 42,
    save: bool = True,
) -> dict:
    """Train a calibrated logistic-regression head on cached embeddings."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    if not DNABERT2_EMBEDDINGS_PATH.exists():
        build_embeddings()

    data = np.load(DNABERT2_EMBEDDINGS_PATH, allow_pickle=True)
    X = data["embeddings"]
    y_labels = data["labels"].astype(str)
    le = LabelEncoder()
    y = le.fit_transform(y_labels)

    # If splits are present in the cache, honour them — that's a real held-out
    # validation set built without leakage (group-aware in build_dataset).
    if "splits" in data.files:
        splits = data["splits"].astype(str)
        train_mask = splits == "train"
        val_mask = splits == "val"
        if val_mask.any():
            X_train, X_val = X[train_mask], X[val_mask]
            y_train, y_val = y[train_mask], y[val_mask]
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=seed, stratify=y
            )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )

    base = LogisticRegression(
        C=1.0,
        max_iter=4000,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )
    head = CalibratedClassifierCV(base, method="sigmoid", cv=5)
    head.fit(X_train, y_train)

    preds = head.predict(X_val)
    report = classification_report(
        y_val,
        preds,
        target_names=list(le.classes_),
        digits=3,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "train_size": int(len(X_train)),
        "val_size": int(len(X_val)),
        "classes": list(le.classes_),
        "accuracy": float((preds == y_val).mean()),
        "macro_f1": float(f1_score(y_val, preds, average="macro")),
        "weighted_f1": float(f1_score(y_val, preds, average="weighted")),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_val, preds).tolist(),
    }

    if save:
        with open(DNABERT2_HEAD_PATH, "wb") as fh:
            pickle.dump(head, fh)
        with open(DNABERT2_LABEL_ENCODER_PATH, "wb") as fh:
            pickle.dump(le, fh)
        (MODELS_DIR / "dnabert2_metrics.json").write_text(
            json.dumps(metrics, indent=2)
        )
        load_head.cache_clear()
        print(f"Saved head + label encoder to {MODELS_DIR}")

    return metrics


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class DNABertPrediction:
    predicted_class: str
    confidence: float
    entropy: float
    probabilities: dict[str, float]
    embedding_norm: float

    def to_dict(self) -> dict:
        return {
            "predicted_class": self.predicted_class,
            "confidence": round(self.confidence, 4),
            "entropy": round(self.entropy, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "embedding_norm": round(self.embedding_norm, 4),
        }


def _entropy(probs: np.ndarray) -> float:
    p = np.clip(probs, 1e-12, 1.0)
    h = float(-(p * np.log(p)).sum())
    h_max = float(np.log(len(probs))) if len(probs) > 1 else 1.0
    return h / h_max if h_max > 0 else 0.0


def predict_sequence_dnabert(sequence: str) -> DNABertPrediction:
    """Run DNABERT-2 prediction. Uses fine-tuned model if available."""
    if (_finetuned_dir() / "config.json").exists():
        return _predict_finetuned(sequence)
    encoder = get_encoder()
    head, le = load_head()
    embedding = encoder.encode([sequence])
    probs = head.predict_proba(embedding)[0]
    pred_idx = int(np.argmax(probs))
    classes = list(le.classes_)
    return DNABertPrediction(
        predicted_class=classes[pred_idx],
        confidence=float(probs[pred_idx]),
        entropy=_entropy(probs),
        probabilities={c: float(probs[i]) for i, c in enumerate(classes)},
        embedding_norm=float(np.linalg.norm(embedding)),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> int:
    import argparse
    import json as _json
    import sys

    parser = argparse.ArgumentParser(description="DNABERT-2 utilities.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_embed = sub.add_parser("embed", help="Compute embeddings for the training corpus.")
    p_embed.add_argument("--source", default="raw", choices=["raw", "train"],
                         help="raw=original CSV; train=augmented dataset (much larger).")
    p_embed.add_argument("--batch-size", type=int, default=8)
    sub.add_parser("train", help="Train the classifier head on cached embeddings.")
    p_pred = sub.add_parser("predict", help="Predict disease for a single sequence.")
    p_pred.add_argument("sequence")
    args = parser.parse_args()

    if args.cmd == "embed":
        build_embeddings(force=True, batch_size=args.batch_size, source=args.source)
        return 0
    if args.cmd == "train":
        m = train_head()
        print(_json.dumps(m, indent=2))
        return 0
    if args.cmd == "predict":
        result = predict_sequence_dnabert(args.sequence)
        print(_json.dumps(result.to_dict(), indent=2))
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
