import os
import numpy as np
import pandas as pd


from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from manual_eval import (
    manual_predict,
    compute_classification_report_from_arrays,
)

from calibration import (
    reliability_diagram,
    expected_calibration_error,
    plot_reliability,
)

# load local fine-tuned model
model_path = "model"

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForSequenceClassification.from_pretrained(model_path)

# load dataset
df = pd.read_csv("C:\\Users\\balke\\Desktop\\Github-Core Skills\\Module 7\\m7-l7a-finetune-sentiment-BatoolMadi\\data\\app_reviews_eval.csv")

# use a subset for faster evaluation
test_texts = df["text"].tolist()

y_true = df["label"].to_numpy()

# manual inference
preds, probs = manual_predict(
    model,
    tokenizer,
    test_texts,
    batch_size=8,
)

print("\nUnique predictions:")
print(np.unique(preds, return_counts=True))

# manual metrics
report = compute_classification_report_from_arrays(
    y_true,
    preds,
)

print("\n=== Classification Report ===")
print(report)

# ECE
ece = expected_calibration_error(
    probs,
    y_true,
    n_bins=10,
)

print("\n=== Expected Calibration Error ===")
print(ece)

# reliability diagram
centers, accs, counts = reliability_diagram(
    probs,
    y_true,
    n_bins=10,
)

# create figures directory
os.makedirs("figures", exist_ok=True)

plot_reliability(
    centers,
    accs,
    counts,
    "figures/reliability-diagram.png",
)

print("\nSaved reliability diagram to:")
print("figures/reliability-diagram.png")

print(model.config.num_labels)
print(model.config.id2label)