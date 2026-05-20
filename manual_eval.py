"""
Stretch Tuesday — Manual Evaluation Harness.

Implement these without using Trainer.predict, sklearn metrics helpers, or
Hugging Face evaluate. The goal is to make the math explicit.
"""

import numpy as np
import torch


def manual_predict(model, tokenizer, texts: list, batch_size: int = 8):
    """
    Run manual PyTorch inference over a list of texts.

    Returns (preds, probs):
      preds: shape (N,), int class indices
      probs: shape (N, num_classes), probabilities (post-softmax)
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    all_preds = []
    all_probs = []

    for i in range(0, len(texts), batch_size):

        batch_texts = texts[i:i + batch_size]

        encodings = tokenizer(
            batch_texts,
            truncation=True,
            max_length=128,
            padding=True,
            return_tensors="pt"
        )

        encodings = {k: v.to(device) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = model(**encodings)
            logits = outputs.logits

        probs = torch.softmax(logits, dim=-1)

        preds = torch.argmax(probs, dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_probs)


def compute_classification_report_from_arrays(y_true, y_pred) -> dict:
    """
    Compute accuracy, per-class precision/recall/F1, and macro-F1 from numpy
    primitives only — no sklearn, no Hugging Face evaluate.

    Returns:
      {
        "accuracy": float,
        "macro_f1": float,
        "per_class": {
            label_index: {
                "precision": ...,
                "recall": ...,
                "f1": ...
            },
            ...
        },
      }
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    classes = np.unique(np.concatenate([y_true, y_pred]))

    per_class = {}

    f1_scores = []

    for c in classes:

        tp = np.sum((y_true == c) & (y_pred == c))

        fp = np.sum((y_true != c) & (y_pred == c))

        fn = np.sum((y_true == c) & (y_pred != c))

        # Precision
        if (tp + fp) > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0.0

        # Recall
        if (tp + fn) > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0.0

        # F1
        if (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        per_class[int(c)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        f1_scores.append(f1)

    accuracy = np.sum(y_true == y_pred) / len(y_true)

    macro_f1 = np.mean(f1_scores)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
    }