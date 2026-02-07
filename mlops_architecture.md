# README — MLOps Architecture for TinyBERT Dual-Head Classifier

## Overview

This document describes the **end-to-end MLOps architecture** for the **TinyBERT Dual-Head Classifier** deployed inside the Investment Reporting Service.

The model performs **two independent classifications per query**:

* **Module classification** (what report the user wants)
* **Date classification** (time context of the report)

Because predictions are **persisted in a database** and can affect **user-facing reports and analytics**, the system is designed assuming:

> ❗ **Predictions can be wrong and correctness is often delayed**

The architecture therefore focuses on **risk containment, monitoring, drift detection, and controlled retraining**.

---

## High-Level Architecture

```
Historical Labeled Data
        ↓
Training & Validation
        ↓
Model Registry
        ↓
Deployment (Reporting Service)
        ↓
Prediction Store (DB)
        ↓
Monitoring & Feedback
        ↺ Retraining / Rollback
```

Each inference produces **two outputs (Module + Date)**, and all safeguards are applied **per head**.

---

## Layer 1 — Confidence-Based Risk Handling (Real-Time)

### Why this layer exists

We cannot know correctness at inference time, but we **can know uncertainty**.
This layer prevents **bad predictions from polluting reports and analytics**.

### Applied per head (IMPORTANT)

Confidence gating is applied **independently** to:

* Module prediction
* Date prediction

A query may have:

* Correct module, wrong date
* Correct date, wrong module

Hence, **per-head thresholds are mandatory**.

---

### Confidence Thresholds

| Confidence Range         | Action                          |
| ------------------------ | ------------------------------- |
| **High (> 0.90)**        | Auto-accept, store normally     |
| **Medium (0.60 – 0.90)** | Accept but mark as `reviewable` |
| **Low (< 0.60)**         | Fallback / flag / exclude       |

---

### Actions for Low Confidence

When confidence < threshold (for either head):

* Route to fallback logic

  * `None_module` or `None_date`
* Flag for manual / LLM review
* Exclude from downstream analytics

This ensures:

* No incorrect reports are generated
* Analytics remains trustworthy

---

### Stored Prediction Schema (Minimum)

```
prediction_id
query_text
module_pred
module_confidence
module_entropy
date_pred
date_confidence
date_entropy
model_version
timestamp
```

---

## Layer 2 — Post-Prediction Monitoring (Offline Correctness)

### Core Question This Layer Answers

> **How do we know if a prediction was wrong?**

Answer:

> We need **ground truth**, which arrives **later**.

---

### Ground Truth Sources

#### 1. Human Review (Gold Standard)

* Weekly sampling (1–5%)
* All low-confidence predictions included
* Produces true labels for:

  * Module
  * Date

Stored as:

```
prediction_id
correct_module
correct_date
reviewer
error_type
```

---

#### 2. LLM-Based Review (Silver Signal)

LLMs can be used to:

* Re-evaluate prediction vs query
* Explain reasoning
* Detect inconsistencies

Examples:

* Query: *"Show holdings for this month"*
* Model output: `holdings + yearly`
* LLM flags date mismatch → `Monthly`

LLMs **do not replace humans**, but:

* Scale feedback
* Prioritize suspicious predictions
* Reduce human workload

---

#### 3. Implicit User Signals (Bronze Signal)

Behavioral indicators:

* Immediate query rephrasing
* Manual date correction
* Report abandonment

These do not give exact labels, but indicate **high error probability**.

---

## Layer 3 — Monitoring Metrics (Grafana)

This layer detects **model health degradation**, not correctness directly.

Below are the **metrics, formulas, thresholds, and examples**.

---

### 1️⃣ Total Inference Requests

**Metric**

```
model_inference_total
```

**Meaning**
Total number of predictions processed.

**Detects**

* Traffic drops → outage
* Sudden spikes → bot or unexpected load

---

### 2️⃣ Inference QPS

**Formula**

```
rate(model_inference_total[1m])
```

**Detects**

* Load surges
* Scaling issues

---

### 3️⃣ P95 Inference Latency

**Formula**

```
histogram_quantile(
  0.95,
  rate(model_inference_latency_seconds_bucket[5m])
)
```

**Meaning**
95% of requests complete faster than this value.

**Thresholds**

* < 0.2s → Excellent
* 0.2–0.5s → Watch
* > 0.5s → Action required

---

### 4️⃣ Average Inference Latency

**Formula**

```
rate(latency_sum) / rate(latency_count)
```

**Why it matters**

* Average hides outliers
* Used only in combination with P95

---

### 5️⃣ Prediction Entropy (Per Head)

**Formula (Softmax Entropy)**

```
H = - Σ p_i log(p_i)
```

Where:

* `p_i` = probability of class `i`

**Meaning**
Measures uncertainty.

**Thresholds**

* < 1.0 → Confident
* 1.0–1.5 → Uncertain
* > 1.5 → Model confused

**Example**

* Query ambiguous between `Monthly` and `Yearly`
* Probabilities spread → high entropy

---

### 6️⃣ Low Confidence Predictions

**Metric**

```
model_low_confidence_total
```

**Meaning**
Count of predictions below confidence threshold.

**Detects**

* Domain expansion
* Training data mismatch

---

### 7️⃣ Confidence PSI (Population Stability Index)

**Formula**

```
PSI = Σ (P_prod - P_train) * ln(P_prod / P_train)
```

**Meaning**
Compares confidence distribution vs training baseline.

**Thresholds**

* < 0.1 → Stable
* 0.1–0.25 → Drift
* > 0.25 → Significant drift

**Example**
Model previously confident (0.9+), now mostly 0.6–0.7.

---

### 8️⃣ Embedding Drift Score

**How it works**

* Embed incoming queries
* Compute distance from training embeddings (e.g., cosine distance)
* Average over time window

**Thresholds**

* < 0.2 → Similar data
* 0.2–0.4 → Drifting
* > 0.4 → Out-of-domain

**Why it matters**
Catches **semantic change**, not just statistics.

---

### 9️⃣ Average Query Length

**Detects**

* User frustration (short)
* Complex queries (long)
* Bot traffic

---

### 🔟 UNK Token Ratio

**Formula**

```
UNK_ratio = (# unknown tokens) / (total tokens)
```

**Thresholds**

* < 10% → Healthy
* 10–30% → Concerning
* > 30% → Model blind

---

### 1️⃣1️⃣ Module Prediction Distribution

**Detects**

* Class dominance
* Bias
* Sudden misclassification

---

### 1️⃣2️⃣ Date Prediction Distribution

**Detects**

* Seasonal patterns
* Anomalous spikes
* Incorrect temporal bias

---

## Layer 4 — Real-Time Safety (Before Damage)

This is **runtime enforcement**, not monitoring.

### Per-Head Gating Rules

| Condition      | Action               |
| -------------- | -------------------- |
| Low confidence | Fallback to `None_*` |
| High entropy   | Mark as unsafe       |
| Drift active   | Increase abstention  |
| Rule violation | Reject prediction    |

This ensures:

* Wrong predictions don’t propagate
* System fails safely

---

## Retraining & Rollback Strategy

Because offline metrics are **1.0**, retraining must be **signal-driven**, not scheduled.

### Retrain When ANY Is True

* Confidence PSI > 0.25
* Embedding drift > 0.4
* Low-confidence rate increases WoW
* Human-verified precision < 97%

---

### Deployment Strategy

1. Train candidate model
2. Shadow deploy
3. Compare:

   * Entropy
   * Confidence
   * Abstention rate
4. Promote only if **safer**, not just more accurate

---

## Final Design Principle

> **Correctness is delayed. Safety is immediate.**

This architecture ensures:

* No silent failures
* No polluted analytics
* Controlled, explainable ML behavior in production
