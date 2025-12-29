# Model Monitoring Metrics

## 1. Population Stability Index (PSI)

**Purpose:** Measures shift in the distribution of a continuous variable (like model confidence) between a baseline (training or reference) and current population. Used for data drift / model input shift monitoring.

**Formula:**

$$
PSI = \sum_{i=1}^{n} (P_i - Q_i) \cdot \ln\left(\frac{P_i}{Q_i}\right)
$$

Where:
- $n$ = number of bins
- $P_i$ = % of baseline (expected) data in bin i
- $Q_i$ = % of current (actual) data in bin i

**Calculation Steps:**
1. Divide baseline data into n bins (quantiles or fixed width)
2. Count % of baseline data in each bin → $P_i$
3. Count % of current/live data in each bin → $Q_i$
4. Apply the formula above

**Interpretation / Thresholds:**
- PSI < 0.1 → No significant change
- 0.1 < PSI < 0.25 → Moderate shift
- PSI > 0.25 → Significant shift (investigate data/model)

**Use Cases:**
- Monitor model confidence distributions (module/date softmax)
- Detect population drift in features or outputs

---

## 2. Characteristic Stability Index (CSI)

**Purpose:** Similar to PSI, but calculated per feature, i.e., measures input feature distribution drift.

**Formula:** Same as PSI but applied to feature distributions.

$$
CSI_{feature} = \sum_{i=1}^{n} (P_i - Q_i) \cdot \ln\left(\frac{P_i}{Q_i}\right)
$$

**Calculation Steps:**
1. For each feature, split baseline into n bins
2. Compute baseline and live % per bin
3. Apply formula

**Thresholds & Interpretation:**
- CSI < 0.1 → Feature stable
- 0.1–0.25 → Moderate drift
- \> 0.25 → Strong drift (may require retraining)

**Use Cases:**
- Track input features in tabular models or embedding dimensions

---

## 3. Concept Drift / Model Drift

**Purpose:** Detects when the relationship between inputs and outputs changes. Unlike PSI/CSI which track input distribution, concept drift checks if model predictions become unreliable.

**Common Methods:**

### a) Prediction Distribution Drift
- Compare predicted probability distribution over time vs training
- Use PSI formula on model predictions (softmax probabilities)
- Example: Confidence PSI for module/date predictions

### b) Performance-based drift
- Track metrics like accuracy, F1-score, AUC over a sliding window of live data
- If metric drops significantly → drift detected

**Formula:**

For confidence / probability drift: same as PSI applied to predicted class probabilities.

For performance: simple moving average difference:

$$
\Delta Accuracy = Accuracy_{baseline} - Accuracy_{live}
$$

**Thresholds:**
- Accuracy drop > 5–10% may indicate drift
- PSI of predictions > 0.1–0.25 → moderate to severe drift

---

## 4. Embedding Drift / Representation Drift

**Purpose:** Measures change in feature embeddings (often from neural networks) between baseline and live data.

**Method:** Cosine similarity is commonly used.

**Formula:**

$$
\text{Cosine Similarity} = \frac{\vec{e}_{live} \cdot \vec{e}_{baseline}}{\|\vec{e}_{live}\| \|\vec{e}_{baseline}\|}
$$

$$
\text{Embedding Drift} = 1 - \text{Cosine Similarity}
$$

**Steps:**
1. Compute mean embedding from training set → $\vec{e}_{baseline}$
2. For each live query, compute its embedding → $\vec{e}_{live}$
3. Calculate cosine similarity → subtract from 1 to get drift score

**Interpretation:**
- Drift < 0.1 → embeddings stable
- 0.1–0.3 → moderate change
- \> 0.3 → significant drift (investigate input or model)

**Use Cases:**
- NLP embeddings for BERT/TinyBERT outputs
- Image embeddings in CV pipelines

---

## 5. Prediction Entropy

**Purpose:** Measures uncertainty of model predictions.

**Formula:**

$$
H(p) = -\sum_{i} p_i \log(p_i)
$$

Where $p_i$ = softmax probability for class i.

**Interpretation:**
- Low entropy → confident predictions
- High entropy → uncertain predictions → may trigger human review or alert

**Thresholds:**
- Relative to training distribution; monitor moving average

---

## 6. Low Confidence Count

**Purpose:** Tracks how often the model outputs low-confidence predictions.

**Method:**
- Count predictions where max(softmax_probs) < threshold
- Threshold is usually 0.5 (or domain-specific)

**Interpretation:**
- Sudden rise → model unsure on current population → potential concept drift


# MLOps Monitoring Metrics - Comprehensive Guide

This guide covers all commonly used MLOps monitoring metrics for tabular, NLP, CV, and classical ML applications, including formulas, interpretations, and threshold guidance.

---

## 1. Data Drift / Input Drift Metrics

These metrics monitor changes in the input feature distributions between training and live data.

| Metric | Formula / Method | Use Case | Threshold / Meaning |
|--------|------------------|----------|---------------------|
| Population Stability Index (PSI) | $\sum_{i=1}^{n} (P_i - Q_i) \ln(P_i/Q_i)$ | Continuous/categorical feature drift | <0.1 stable, 0.1–0.25 moderate, >0.25 severe drift |
| Characteristic Stability Index (CSI) | PSI per feature | Categorical/continuous input features | Same as PSI |
| Kolmogorov–Smirnov (KS) Test | $D = \max \|F_1(x) - F_2(x)\|$ | Continuous feature drift | p-value < 0.05 → significant drift |
| Jensen-Shannon Divergence (JSD) | $JSD(P \|\| Q) = \frac{1}{2}KL(P \|\| M) + \frac{1}{2}KL(Q \|\| M)$ | Distribution comparison | 0–1, higher → more divergence |
| Wasserstein Distance (Earth Mover's Distance) | Measures "cost" to move baseline distribution to live | Continuous features | Domain-specific thresholds |

✅ **Use in all applications:** tabular, NLP embeddings, CV features (pixel distributions, extracted features).

---

## 2. Concept Drift / Model Drift Metrics

These metrics measure changes in the relationship between inputs and outputs.

| Metric | Formula / Method | Use Case | Threshold / Meaning |
|--------|------------------|----------|---------------------|
| Prediction Distribution Drift (Confidence PSI) | PSI on softmax probabilities per class | Classification models | <0.1 stable, >0.25 alert |
| Accuracy / F1 / AUC Moving Average | Compare baseline metric vs live metric over window | All supervised tasks | Drop > 5–10% → drift |
| KL Divergence of Predictions | $KL(P_{train} \|\| P_{live})$ | Classification output drift | Higher → model behavior changed |
| Prediction Entropy | $H(p) = -\sum p_i \log(p_i)$ | Uncertainty monitoring | High → model unsure |
| Low Confidence Count | Count(max_prob < threshold) | Classification confidence | Rising count → alert |

---

## 3. Embedding / Representation Drift Metrics

For deep learning embeddings (text, images, tabular):

| Metric | Formula / Method | Use Case | Threshold / Meaning |
|--------|------------------|----------|---------------------|
| Cosine Similarity Drift | $1 - \frac{\vec{e}_{live} \cdot \vec{e}_{baseline}}{\|\vec{e}_{live}\| \|\vec{e}_{baseline}\|}$ | NLP embeddings, CV embeddings | <0.1 stable, 0.1–0.3 moderate, >0.3 severe |
| Mahalanobis Distance | $d_M = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}$ | Detect OOD in embeddings | > threshold → anomaly |
| Euclidean / L2 Distance | $\|x_{live} - \mu_{baseline}\|_2$ | Embedding drift | Domain-specific |

✅ **Use Cases:**
- **NLP:** BERT/TinyBERT CLS token embeddings
- **CV:** CNN feature maps or bottleneck layers
- **Tabular:** Autoencoder latent representations

---

## 4. Out-of-Distribution (OOD) Detection / Novelty Detection

| Metric | Formula / Method | Use Case | Threshold / Meaning |
|--------|------------------|----------|---------------------|
| Maximum Softmax Probability (MSP) | $1 - \max(p_i)$ | Detect OOD samples | High → anomalous |
| ODIN Score | Temperature-scaled softmax with small perturbation | NLP, CV | Higher → OOD |
| Mahalanobis Distance in Feature Space | As above | Detect OOD embeddings | High → OOD |
| Autoencoder Reconstruction Error | $\|x - \hat{x}\|$ | Anomaly detection | High error → OOD |

---

## 5. Regression / Continuous Output Monitoring

| Metric | Formula / Method | Use Case | Threshold / Meaning |
|--------|------------------|----------|---------------------|
| Mean Squared Error (MSE) Drift | Compare live MSE with baseline | Regression models | Significant increase → drift |
| Mean Absolute Error (MAE) Drift | Compare live MAE with baseline | Regression models | Significant increase → drift |
| Residual Distribution Drift (KS Test / PSI) | Apply KS/PSI to residuals | Regression | Residual distribution shifts → model misalignment |
| Prediction Interval Coverage | % of true values inside predicted interval | Uncertainty monitoring | Drop → model underestimates uncertainty |

---

## 6. Feature Importance / SHAP Monitoring

| Metric | Formula / Method | Use Case | Threshold / Meaning |
|--------|------------------|----------|---------------------|
| SHAP Distribution Drift | PSI / KS test on SHAP values per feature | Detect change in model reasoning | High drift → retrain or investigate |
| Permutation Importance Drift | Compare feature importance baseline vs live | Tabular / CV | Drift → model relying on different features |

---

## 7. Latency and System Metrics

| Metric | Formula / Method | Use Case | Threshold / Meaning |
|--------|------------------|----------|---------------------|
| Inference Latency | Wall-clock time per request | All models | SLA violation → alert |
| Throughput / Requests per Second | Count / time | System performance | Low → bottleneck |
| Error Rate | Failed inference count / total | Reliability | High → alert |

---

## 8. Summary Table (All Domains)

| Metric | Domain | Formula / Method | Interpretation |
|--------|--------|------------------|----------------|
| PSI | Tabular/NLP/CV | $\sum (P_i - Q_i) \ln(P_i/Q_i)$ | Data / prediction distribution shift |
| CSI | Tabular | PSI per feature | Feature drift |
| KL / JSD | Tabular/NLP/CV | $\sum P \log(P/Q)$ | Distribution change |
| Cosine Drift | NLP/CV | 1 - cos_sim | Embedding drift |
| Mahalanobis / L2 | NLP/CV/Tabular | Distance in feature space | OOD detection |
| Entropy | Classification | $-\sum p_i \log(p_i)$ | Prediction uncertainty |
| Low Confidence | Classification | Count(max_prob < threshold) | Confidence monitoring |
| Residual / Error Drift | Regression | MSE/MAE comparison | Model performance drift |
| SHAP / Feature Drift | Tabular/NLP/CV | PSI / KS on importance | Explainability drift |
| Latency | All | Time per inference | SLA monitoring |

---

## Key Points

- **PSI/CSI** → distribution shift
- **Embedding / Representation Drift** → changes in feature space
- **Prediction-based Drift** → model concept drift
- **Entropy / Confidence** → uncertainty monitoring
- **Residual / Error Drift** → regression model performance
- **SHAP / Feature Drift** → explainability shift
- **System Metrics** → operational health