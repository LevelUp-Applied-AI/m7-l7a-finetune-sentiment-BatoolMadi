# Module 7 Week A — Lab Evaluation Report

## Dataset

This lab used the AARSynth app reviews dataset for 3-class sentiment classification.
The dataset contains 7,472 app reviews collected from multiple Android applications, with sentiment labels mapped as:

* 0 = negative
* 1 = neutral
* 2 = positive

The dataset was split into 80% training and 20% testing using a fixed random seed of 42. This produced:

* Training samples: 5,977
* Test samples: 1,495

---

## Model and Hyperparameters

* Backbone: distilbert-base-uncased
* Number of labels: 3
* Learning rate: 5e-5
* Epochs: 2
* Batch size: 8
* Max sequence length: 128
* Random seed: 42

### Training time

The full training process took approximately 31 minutes on a Windows laptop with an Intel i7 CPU and NVIDIA MX330 GPU.

---

## Metrics on the Test Split

### Aggregate Metrics

| Metric   | Value  |
| -------- | ------ |
| Accuracy | 0.6241 |
| Macro-F1 | 0.6228 |

### Per-Class Metrics

| Class    | F1     | Precision | Recall |
| -------- | ------ | --------- | ------ |
| Negative | 0.7098 | 0.7163    | 0.7034 |
| Neutral  | 0.4724 | 0.4485    | 0.4989 |
| Positive | 0.6862 | 0.7163    | 0.6585 |

---

## Confusion Matrix

| True \ Predicted | Negative | Neutral | Positive |
| ---------------- | -------- | ------- | -------- |
| Negative         | 351      | 131     | 17       |
| Neutral          | 110      | 231     | 122      |
| Positive         | 29       | 153     | 351      |

The model performed best on the negative and positive classes, while the neutral class was significantly harder to classify correctly. Many neutral reviews were predicted as either positive or negative because they often contained mixed sentiment expressions.

---

## Three Qualitative Error Examples

### 1. Positive review predicted as Neutral

**Sentence:**
"good, but slow workflow."

* Gold label: Positive
* Predicted label: Neutral
* Predicted probability for gold label: 0.3599

### Analysis

This review contains both positive and negative cues. The word “good” suggests positive sentiment, while “slow workflow” introduces criticism. The model likely focused more on the negative usability phrase and interpreted the overall sentiment as neutral rather than positive.

---

### 2. Neutral review predicted as Positive

**Sentence:**
"when i was being rained on it said no rain in your <url> it is still a good weather app."

* Gold label: Neutral
* Predicted label: Positive
* Predicted probability for gold label: 0.4578

### Analysis

The review includes a complaint about incorrect weather prediction, but it also ends with positive wording (“good weather app”). The model appears to overweight the positive phrase at the end of the sentence and ignored the earlier criticism.

---

### 3. Positive review predicted as Neutral

**Sentence:**
"its a good app but we dont have a night mode option for reading in which the background becomes black and text becomes white in color."

* Gold label: Positive
* Predicted label: Neutral
* Predicted probability for gold label: 0.1880

### Analysis

This sentence expresses overall satisfaction but also requests a missing feature. Because the review mixes praise with feature criticism, the classifier struggled to determine the dominant sentiment. The low probability for the gold label suggests uncertainty caused by mixed semantic signals.

---

## Hugging Face Hub Model URL

https://huggingface.co/BatoolMadi/m7-app-review-sentiment
