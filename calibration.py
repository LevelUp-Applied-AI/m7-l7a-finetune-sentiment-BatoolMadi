"""
Stretch Tuesday — Calibration Analysis.

Reliability diagram + Expected Calibration Error (ECE).
"""

import numpy as np


def reliability_diagram(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10):
    """
    Bin predictions by max predicted probability; compute empirical accuracy per bin.

    Returns (bucket_centers, bucket_accuracies, bucket_counts), all length n_bins.
    """

    # max confidence for each prediction
    confidences = np.max(probs, axis=1)

    # predicted class index
    preds = np.argmax(probs, axis=1)

    # whether prediction was correct
    correct = preds == y_true

    # create bins
    edges = np.linspace(0, 1, n_bins + 1)

    bucket_centers = []
    bucket_accuracies = []
    bucket_counts = []

    for i in range(n_bins):

        left = edges[i]
        right = edges[i + 1]

        # last bin includes right edge
        if i == n_bins - 1:
            in_bucket = (confidences >= left) & (confidences <= right)
        else:
            in_bucket = (confidences >= left) & (confidences < right)

        count = np.sum(in_bucket)

        # midpoint of bucket
        center = (left + right) / 2

        if count > 0:
            accuracy = np.mean(correct[in_bucket])
        else:
            accuracy = 0.0

        bucket_centers.append(center)
        bucket_accuracies.append(accuracy)
        bucket_counts.append(count)

    return (
        np.array(bucket_centers),
        np.array(bucket_accuracies),
        np.array(bucket_counts),
    )


def expected_calibration_error(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    ECE = sum over bins of (bucket_count / N) * |bucket_accuracy - bucket_confidence|.

    A perfectly calibrated model has ECE = 0.
    """

    confidences = np.max(probs, axis=1)

    preds = np.argmax(probs, axis=1)

    correct = preds == y_true

    edges = np.linspace(0, 1, n_bins + 1)

    total_samples = len(y_true)

    ece = 0.0

    for i in range(n_bins):

        left = edges[i]
        right = edges[i + 1]

        if i == n_bins - 1:
            in_bucket = (confidences >= left) & (confidences <= right)
        else:
            in_bucket = (confidences >= left) & (confidences < right)

        bucket_count = np.sum(in_bucket)

        if bucket_count > 0:

            bucket_accuracy = np.mean(correct[in_bucket])

            bucket_confidence = np.mean(confidences[in_bucket])

            ece += (
                (bucket_count / total_samples)
                * abs(bucket_accuracy - bucket_confidence)
            )

    return float(ece)


def plot_reliability(
    centers: np.ndarray,
    accs: np.ndarray,
    counts: np.ndarray,
    output_path: str,
) -> None:
    """Save a reliability diagram. Provided helper — do not modify."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    width = 1.0 / max(len(centers), 1)

    ax.bar(
        centers,
        accs,
        width=width * 0.9,
        edgecolor="black",
        alpha=0.8,
        label="Empirical accuracy",
    )

    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_xlabel("Predicted probability (bucket center)")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability diagram")

    ax.legend(loc="upper left")

    fig.tight_layout()

    fig.savefig(output_path, dpi=150)

    plt.close(fig)