# Offline Valuation Metrics

## Summary Table

| S.No | Task | Metrics |
|------|------|---------|
| 1 | Classification | Precision, Recall, F1 Score, Accuracy, ROC-AUC, PR-AUC, Confusion Matrix |
| 2 | Regression | MSE, MAE, RMSE |
| 3 | Ranking | Precision@K, Recall@K, MRR, nDCG, mAP |
| 4 | Image Generation | FID, Inception Score |
| 5 | Natural Language Processing | BLEU, ROUGE, METEOR |

---

## Metric Definitions

---

### 🔵 Classification Metrics

---

#### 1. Precision

| Field | Detail |
|-------|--------|
| **Definition** | Of all the instances predicted as positive, how many are actually positive. |
| **Why** | Penalizes false positives; useful when the cost of a false alarm is high. |
| **When to Use** | Spam detection, fraud detection; where false positives are costly. |
| **When Not to Use** | When missing actual positives (false negatives) is more costly than false alarms. |
| **Limitations** | Ignores false negatives entirely; a model that predicts very few positives can have high precision but be practically useless. |
| **Formula** | `Precision = TP / (TP + FP)` |

---

#### 2. Recall (Sensitivity)

| Field | Detail |
|-------|--------|
| **Definition** | Of all actual positives, how many were correctly identified. |
| **Why** | Penalizes false negatives; useful when missing a positive is dangerous. |
| **When to Use** | Cancer detection, disease screening; where missing a positive is costly. |
| **When Not to Use** | When false positives are expensive, such as systems where every flagged case requires costly manual review. |
| **Limitations** | Ignores false positives; a model that predicts everything as positive achieves perfect recall but is useless. |
| **Formula** | `Recall = TP / (TP + FN)` |

---

#### 3. F1 Score

| Field | Detail |
|-------|--------|
| **Definition** | Harmonic mean of Precision and Recall. |
| **Why** | Balances Precision and Recall; useful when both matter equally. |
| **When to Use** | Imbalanced datasets where both false positives and false negatives matter. |
| **When Not to Use** | When Precision and Recall are not equally important; use a weighted F-beta score instead. Also avoid when class distribution is balanced and Accuracy is sufficient. |
| **Limitations** | Treats Precision and Recall as equally important, which may not reflect real business costs. Does not account for true negatives. |
| **Formula** | `F1 = 2 x (Precision x Recall) / (Precision + Recall)` |

---

#### 4. Accuracy

| Field | Detail |
|-------|--------|
| **Definition** | Proportion of correct predictions (both TP and TN) out of all predictions. |
| **Why** | Simple and intuitive overall performance measure. |
| **When to Use** | Balanced datasets where all classes are equally important. |
| **When Not to Use** | Imbalanced datasets; a model predicting the majority class always will still score high accuracy while being useless for the minority class. |
| **Limitations** | Highly misleading on imbalanced data. Does not reveal what type of errors the model is making. |
| **Formula** | `Accuracy = (TP + TN) / (TP + TN + FP + FN)` |

---

#### 5. ROC-AUC (Receiver Operating Characteristic - Area Under Curve)

| Field | Detail |
|-------|--------|
| **Definition** | Area under the ROC curve (TPR vs FPR across all thresholds). |
| **Why** | Threshold-independent measure; evaluates the ranking quality of the classifier. |
| **When to Use** | Binary classification with imbalanced classes; comparing models across thresholds. |
| **When Not to Use** | Highly imbalanced datasets where the negative class dominates; ROC-AUC can appear optimistic. Use PR-AUC in those cases. |
| **Limitations** | Can be overly optimistic on severely imbalanced data because it includes TN in FPR, inflating the score when negatives are abundant. |
| **Formula** | `AUC = integral of TPR d(FPR)` where `TPR = TP/(TP+FN)`, `FPR = FP/(FP+TN)` |

---

#### 6. PR-AUC (Precision-Recall - Area Under Curve)

| Field | Detail |
|-------|--------|
| **Definition** | Area under the Precision-Recall curve across all thresholds. |
| **Why** | More informative than ROC-AUC when the positive class is rare. |
| **When to Use** | Highly imbalanced datasets such as rare disease detection or fraud identification. |
| **When Not to Use** | Balanced datasets where ROC-AUC is equally informative and more widely understood. |
| **Limitations** | Does not account for true negatives; harder to interpret as a standalone number compared to ROC-AUC. Sensitive to class ratio changes between train and test sets. |
| **Formula** | `PR-AUC = integral of Precision d(Recall)` |

---

#### 7. Confusion Matrix

| Field | Detail |
|-------|--------|
| **Definition** | A table showing TP, TN, FP, FN counts for each class. |
| **Why** | Gives a full breakdown of where the model is making errors. |
| **When to Use** | Multi-class or binary classification tasks; to diagnose specific error types. |
| **When Not to Use** | As a standalone summary metric in automated pipelines; it requires human interpretation and does not reduce to a single comparable number. |
| **Limitations** | Becomes hard to read with many classes. Does not normalize for class imbalance unless percentages are shown. Provides no single optimizable score. |
| **Formula** | Matrix of `[TP, FP; FN, TN]`; no single formula. All classification metrics are derived from it. |

---

### 🟠 Regression Metrics

---

#### 8. MSE (Mean Squared Error)

| Field | Detail |
|-------|--------|
| **Definition** | Average of the squared differences between predicted and actual values. |
| **Why** | Heavily penalizes large errors due to squaring; differentiable and good for optimization. |
| **When to Use** | When large errors are particularly undesirable and should be penalized more than small ones. |
| **When Not to Use** | When the dataset has many outliers; MSE will be dominated by those extreme values and may not reflect typical model performance. |
| **Limitations** | Not in the same unit as the target variable, making it hard to interpret directly. Very sensitive to outliers. |
| **Formula** | `MSE = (1/n) x sum(yᵢ - ŷᵢ)²` |

---

#### 9. MAE (Mean Absolute Error)

| Field | Detail |
|-------|--------|
| **Definition** | Average of the absolute differences between predicted and actual values. |
| **Why** | Robust to outliers; easy to interpret in the original units of the target. |
| **When to Use** | When all errors should be treated equally regardless of magnitude; robust evaluation in noisy datasets. |
| **When Not to Use** | When large errors are disproportionately harmful and need to be penalized more; use MSE or RMSE instead. |
| **Limitations** | Not differentiable at zero, which can cause issues with certain gradient-based optimizers. Does not penalize large deviations more than small ones. |
| **Formula** | `MAE = (1/n) x Σᵢ₌₁ⁿ |yᵢ - ŷᵢ| `

---

#### 10. RMSE (Root Mean Squared Error)

| Field | Detail |
|-------|--------|
| **Definition** | Square root of MSE, bringing the error back to the original unit scale of the target variable. |
| **Why** | Same unit as the target variable; penalizes large errors more than MAE while remaining interpretable. |
| **When to Use** | When you want MSE sensitivity to large errors but need the result in the same units as your output. |
| **When Not to Use** | When the data has significant outliers and you do not want them to dominate the metric; use MAE in those cases. |
| **Limitations** | Still sensitive to outliers like MSE. Can be misleading if compared across datasets with different scales without normalization. |
| **Formula** | `RMSE = sqrt((1/n) x sum(yᵢ - ŷᵢ)²)` |

---

### 🟢 Ranking Metrics

---

#### 11. Precision@K

| Field | Detail |
|-------|--------|
| **Definition** | Fraction of relevant items among the top K retrieved results. |
| **Why** | Measures how accurate the top-K recommendations or search results are. |
| **When to Use** | Search engines, recommender systems; when only the top results are shown to the user. |
| **When Not to Use** | When the total number of relevant items matters; use Recall@K or nDCG to capture coverage. |
| **Limitations** | Ignores the position of relevant items within the top K. Does not penalize a system for missing many relevant items outside the top K. |
| **Formula** | `Precision@K = (Relevant items in top K) / K` |

---

#### 12. Recall@K

| Field | Detail |
|-------|--------|
| **Definition** | Fraction of all relevant items that appear in the top K results. |
| **Why** | Measures coverage; how many of the total relevant items were surfaced in the top K. |
| **When to Use** | Information retrieval tasks; when missing relevant results is costly. |
| **When Not to Use** | When the total number of relevant items is unknown or very large; Recall@K becomes hard to interpret meaningfully. |
| **Limitations** | Does not account for the order of items within K. Can be uninformative if the total number of relevant items varies widely across queries. |
| **Formula** | `Recall@K = (Relevant items in top K) / (Total relevant items)` |

---

#### 13. MRR (Mean Reciprocal Rank)

| Field | Detail |
|-------|--------|
| **Definition** | Average of the reciprocal ranks of the first relevant result across queries. |
| **Why** | Rewards systems that place the first correct answer as high as possible. |
| **When to Use** | Q&A systems, single-answer search tasks; when the position of the first correct result matters most. |
| **When Not to Use** | When multiple relevant results matter, not just the first one. MRR ignores all relevant items after the first. |
| **Limitations** | Only considers the rank of the first relevant item, ignoring all subsequent relevant results. Not suitable for tasks with multiple equally important relevant answers. |
| **Formula** | `MRR = (1/Q) x sum(1 / rankᵢ)` where `rankᵢ` is the position of the first relevant result for query i |

---

#### 14. nDCG (Normalized Discounted Cumulative Gain)

| Field | Detail |
|-------|--------|
| **Definition** | Measures ranking quality by applying a logarithmic discount to items ranked lower, normalized against the ideal ranking. |
| **Why** | Accounts for graded relevance and position; rewards placing highly relevant items at the top. |
| **When to Use** | Search engines, recommender systems with graded relevance scores (e.g., ratings of 0 to 5). |
| **When Not to Use** | When only binary relevance (relevant/not relevant) exists and simpler metrics like Precision@K are sufficient. |
| **Limitations** | Requires graded relevance labels, which are expensive to collect. Normalization makes it harder to diagnose specific failures. Results vary with the choice of K. |
| **Formula** | `DCG@K = sum((2^relᵢ - 1) / log₂(i+1))` then `nDCG@K = DCG@K / IDCG@K` where IDCG@K is DCG of the ideal (perfect) ranking |

---

#### 15. mAP (Mean Average Precision)

| Field | Detail |
|-------|--------|
| **Definition** | Mean of Average Precision scores across all queries. AP is the area under the Precision-Recall curve per query. |
| **Why** | Evaluates both ranking quality and recall completeness across multiple queries in one score. |
| **When to Use** | Object detection, document retrieval, multi-query ranking tasks. |
| **When Not to Use** | When queries have graded relevance rather than binary relevance; nDCG is more appropriate there. |
| **Limitations** | Assumes binary relevance labels. Sensitive to the number of relevant items per query. Can be dominated by queries with many relevant documents. |
| **Formula** | `AP = sum(Precision@k x DeltaRecall@k)` then `mAP = (1/Q) x sum(APᵢ)` |

---

### 🟣 Image Generation Metrics

---

#### 16. FID (Frechet Inception Distance)

| Field | Detail |
|-------|--------|
| **Definition** | Measures the distance between the feature distributions of real and generated images using a pre-trained Inception network. |
| **Why** | Captures both quality and diversity of generated images. Lower FID indicates better performance. |
| **When to Use** | Evaluating GANs, diffusion models, or any image generation model at scale. |
| **When Not to Use** | When evaluating small datasets; FID requires a large number of samples (typically 10k or more) to be statistically reliable. |
| **Limitations** | Dependent on the Inception network trained on ImageNet; may not generalize to non-natural image domains such as medical imaging. Does not measure perceptual quality directly. Computationally expensive. |
| **Formula** | `FID = ‖mu_r - mu_g‖^2 + Tr(Sigma_r + Sigma_g - 2*(Sigma_r @ Sigma_g)^(1/2))` where `‖mu_r - mu_g‖^2` is the squared Euclidean distance between real and generated mean vectors, `Sigma_r @ Sigma_g` is the matrix product of the two covariance matrices, and `^(1/2)` denotes the matrix square root |

---

#### 17. Inception Score (IS)

| Field | Detail |
|-------|--------|
| **Definition** | Measures quality (confident class predictions) and diversity (varied class outputs) of generated images using a pre-trained Inception classifier. |
| **Why** | Higher IS indicates high-quality and diverse images. Provides a quick single-number summary. |
| **When to Use** | Quick GAN evaluation; initial benchmarking of image generation quality. |
| **When Not to Use** | As the sole metric for generation quality; prefer FID for a more reliable evaluation. Not suitable for domains where the Inception classifier is irrelevant. |
| **Limitations** | Does not compare generated images to real images, only to ImageNet class priors. Can be gamed by memorizing training data. Unreliable for non-ImageNet domains. |
| **Formula** | `IS = exp(E_x[KL(p(y given x) ‖ p(y))])` where `p(y given x)` is the class distribution for a single generated image x, and `p(y) = E_x[p(y given x)]` is the marginal class distribution averaged over all generated images |

---

### 🔴 NLP Metrics

---

#### 18. BLEU (Bilingual Evaluation Understudy)

| Field | Detail |
|-------|--------|
| **Definition** | Measures n-gram overlap between machine-generated and reference text, with a brevity penalty to discourage very short outputs. |
| **Why** | Fast, reference-based metric widely used for translation benchmarking and reproducible comparisons. |
| **When to Use** | Machine translation evaluation; tasks where reference outputs exist and surface-level overlap is a reasonable proxy for quality. |
| **When Not to Use** | Open-ended generation tasks like dialogue or story writing where many valid outputs exist beyond the reference. Also unreliable for very short sentences. |
| **Limitations** | Does not account for synonyms or paraphrases; penalizes valid translations that use different wording. Correlates weakly with human judgment at the sentence level. Precision-focused and ignores recall. |
| **Formula** | `BLEU = BP x exp(sum(wₙ x log pₙ))` where `pₙ` = n-gram precision, `wₙ` = weight (typically 1/N), and `BP = 1 if c >= r, else exp(1 - r/c)` with c = candidate length and r = reference length |

---

#### 19. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

| Field | Detail |
|-------|--------|
| **Definition** | Measures n-gram, word sequence, or word pair overlap between generated and reference summaries. Key variants are ROUGE-N (n-gram overlap), ROUGE-L (longest common subsequence), and ROUGE-W (weighted LCS). |
| **Why** | Recall-focused; well-suited for summarization where coverage of reference content is the priority. |
| **When to Use** | Text summarization and document compression tasks where completeness of information matters. |
| **When Not to Use** | Tasks where fluency and coherence matter more than exact word overlap, such as creative writing or dialogue generation. |
| **Limitations** | Does not capture semantic meaning; two summaries with different words but the same meaning can score very differently. Relies on exact string matching. Sensitive to the quality and number of reference summaries provided. |
| **Formula** | `ROUGE-N = sum(Match(gramₙ)) / sum(Count(gramₙ in reference))` |

---

#### 20. METEOR (Metric for Evaluation of Translation with Explicit ORdering)

| Field | Detail |
|-------|--------|
| **Definition** | Aligns generated text with references using exact match, stemming, and synonymy, then computes an F-score with a fragmentation penalty for non-contiguous matches. |
| **Why** | Better correlation with human judgment than BLEU; handles synonyms and morphological word variants. |
| **When to Use** | Machine translation and text generation tasks; especially when synonym matching and morphological flexibility are important. |
| **When Not to Use** | When synonym and stemming resources are unavailable for the target language; the advantage over BLEU diminishes without these. Also overkill for quick benchmarking where BLEU is the established standard. |
| **Limitations** | Language-dependent; requires external synonym databases such as WordNet that may not exist for all languages. Slower to compute than BLEU. The fragmentation penalty is heuristic and not always well-calibrated. |
| **Formula** | `METEOR = Fmean x (1 - Penalty)` where `Fmean = (10 x P x R) / (R + 9P)` and `Penalty = 0.5 x (chunks / matched_unigrams)^3` with chunks = number of non-contiguous matched segments |

---