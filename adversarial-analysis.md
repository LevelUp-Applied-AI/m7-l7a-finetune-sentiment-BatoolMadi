# Adversarial Evaluation Analysis

## Per-hypothesis accuracy

| Hypothesis Category | Correct | Incorrect |
|---|---|---|
| negation | 3 | 2 |
| lexical_trigger | 3 | 2 |
| domain_shift | 4 | 1 |
| length_extreme | 4 | 1 |
| sarcasm | 2 | 2 |
| other | 5 | 1 |

Overall accuracy on the adversarial set was moderate. The model performed better on domain-shift and length-extreme examples than expected, while sarcasm and negation remained difficult categories.

## Confirmed hypotheses

The adversarial evaluation confirmed several expected weaknesses in the model. The strongest failures appeared in the sarcasm category. For example, row 21 (“Fantastic another crash during my presentation.”) was predicted as positive because the model focused heavily on the positive cue word “Fantastic” while ignoring the sarcastic context. Similarly, row 22 (“Wonderful the app deleted all my files again.”) was also misclassified due to the positive lexical trigger “Wonderful.”

Negation examples also exposed weaknesses. Row 5 (“I can't say the experience was enjoyable.”) received an incorrect prediction because the model appeared to overweight the positive cue “enjoyable” instead of correctly interpreting the negation phrase. Row 7 showed similar behavior where the phrase “not as exciting” still triggered positive sentiment cues.

These results support the hypothesis that the classifier relies heavily on individual sentiment-bearing keywords and struggles when polarity depends on compositional language structure such as negation or sarcasm.

## Refuted hypotheses

Some hypotheses were weaker than expected. The model handled several domain-shift examples surprisingly well. For example, weather, sports, and political-news sentences were often classified as neutral rather than incorrectly forced into positive or negative sentiment categories. This suggests the classifier learned at least some ability to detect emotionally neutral language patterns even outside the app-review domain.

The model also performed better than expected on long examples. Rows 19 and 20 contained lengthy mixed-sentiment sentences with multiple clauses, yet the classifier still correctly identified the overall negative sentiment. This indicates the model can preserve contextual information across relatively long inputs instead of relying only on early cue words.

These results were unexpected because adversarial evaluation often reveals stronger degradation under domain shift and long-context conditions.

## What the results reveal about the decision boundary

The adversarial results suggest that the model’s decision boundary is strongly influenced by high-salience sentiment keywords such as “fantastic,” “wonderful,” “great,” and “amazing.” Even when those words appear inside sarcastic or negated contexts, the classifier frequently predicts the polarity associated with the cue word itself rather than the true sentence meaning.

At the same time, the model appears more robust when processing emotionally neutral domain-shift examples than originally expected. This suggests the decision boundary is not purely keyword-driven; the model has learned some broader contextual representation of neutral language. However, sarcasm and negation remain clear weaknesses because they require compositional reasoning rather than surface-level lexical matching.

Overall, the adversarial evaluation reveals that the classifier generalizes reasonably well to straightforward unseen text but struggles when sentiment depends on subtle linguistic structure or implicit meaning.