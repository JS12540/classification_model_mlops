# Grafana Metrics Guide: Understanding Your ML Model Monitoring Dashboard

## 📊 PANEL 1: Total Inference Requests
**Metric Type:** Counter  
**Query:** `model_inference_total`

### What it means
This shows the cumulative total number of inference requests your model has processed since it started running.

### When it's useful
- **Increasing steadily**: ✅ Normal - your model is processing requests
- **Flat line**: ⚠️ No new requests coming in - investigate if traffic has stopped
- **Sudden spike**: 🔍 Unexpected traffic surge - check for attacks, bot traffic, or legitimate usage increases
- **Drops to zero then restarts**: 🔄 Service restart detected

---

## ⚡ PANEL 2: Inference QPS (Requests Per Second)
**Metric Type:** Counter with rate()  
**Query:** `rate(model_inference_total[1m])`

### What it means
This shows how many inference requests your model processes **per second** in real-time.

### When it's useful
- **High QPS (>100/s)**: 
  - ✅ Good: Your model is handling heavy load successfully
  - ⚠️ Risk: Monitor latency - high QPS can lead to slowdowns
- **Low QPS (<10/s)**: 
  - ✅ Normal for low-traffic applications
  - 🔍 Investigate if you expect more traffic
- **Sudden drops to zero**: 🚨 Service outage or load balancer issue
- **Spiky pattern**: 🔍 Bursty traffic - consider auto-scaling

---

## 🐌 PANEL 3: P95 Inference Latency
**Metric Type:** Histogram  
**Query:** `histogram_quantile(0.95, rate(model_inference_latency_seconds_bucket[5m]))`

### What it means
95% of your inference requests complete **faster than this value**. Only 5% take longer.

### When it's useful
- **< 0.2s (Green)**: ✅ Excellent - users get fast responses
- **0.2s - 0.5s (Yellow)**: ⚠️ Acceptable but degrading - investigate before it gets worse
- **> 0.5s (Red)**: 🚨 Poor user experience - immediate action needed

### What causes high P95 latency
- Model complexity too high
- Insufficient GPU/CPU resources
- Memory swapping
- Network bottlenecks
- Database query slowness
- Cold start issues

---

## ⏱️ PANEL 4: Average Inference Latency
**Metric Type:** Histogram (calculated)  
**Query:** `rate(model_inference_latency_seconds_sum[5m]) / rate(model_inference_latency_seconds_count[5m])`

### What it means
The mean time it takes to process an inference request.

### When it's useful
- **Compare with P95**: If average is much lower than P95, you have outliers slowing down some requests
- **< 0.1s**: ✅ Very fast
- **0.1s - 0.3s**: ✅ Good
- **> 0.5s**: ⚠️ Investigate performance issues

### Why average alone isn't enough
Average hides outliers. A few very slow requests won't significantly affect the average, but they hurt user experience. That's why P95 is more important.

---

## 🎲 PANEL 5: Prediction Entropy
**Metric Type:** Gauge  
**Query:** `prediction_entropy`

### What it means
Measures how uncertain/confused your model is about its predictions. High entropy = model is unsure and spreading probability across many options.

### When it's useful
- **< 1.0 (Green)**: ✅ Model is confident and decisive
- **1.0 - 1.5 (Yellow)**: ⚠️ Model is somewhat uncertain - predictions may be less reliable
- **> 1.5 (Red)**: 🚨 Model is very confused - likely seeing data it wasn't trained on

### What causes high entropy
- Input data very different from training data
- Ambiguous queries that could have multiple valid answers
- Model hasn't learned the pattern well
- Data distribution shift

---

## ⚠️ PANEL 6: Low Confidence Predictions
**Metric Type:** Counter  
**Query:** `model_low_confidence_total`

### What it means
Total count of predictions where the model's confidence score fell below your threshold (e.g., < 70% confidence).

### When it's useful
- **Low count**: ✅ Model is generally confident
- **Rising steadily**: 🚨 Model is encountering more uncertain cases
  - Users may be asking questions outside training domain
  - Data drift is occurring
  - Model needs retraining

### Action items when high
- Review the low-confidence predictions manually
- Add these examples to your training data
- Implement a fallback mechanism (e.g., "I'm not sure, would you like to rephrase?")
- Alert humans to review these predictions

---

## 📉 PANEL 7: Confidence PSI (Population Stability Index)
**Metric Type:** Gauge  
**Query:** `confidence_psi`

### What it means
Detects **data drift** by comparing current confidence score distribution to the baseline distribution from training/validation.

### When it's useful
- **< 0.1 (Green)**: ✅ No significant drift - model sees similar data as training
- **0.1 - 0.25 (Yellow)**: ⚠️ Moderate drift detected - monitor closely
- **> 0.25 (Red)**: 🚨 Significant drift - model is seeing very different data

### What causes drift
- User behavior changes over time
- New types of queries not in training data
- Seasonal patterns
- Market/domain shifts
- Data pipeline issues

### Action items when high
- Retrain model on recent data
- Investigate what changed in input distribution
- Consider online learning or periodic retraining

---

## 🧭 PANEL 8: Embedding Drift Score
**Metric Type:** Gauge  
**Query:** `embedding_drift_score`

### What it means
Measures how different the **semantic meaning** of current inputs is compared to training data, using embedding space distance.

### When it's useful
- **< 0.2 (Green)**: ✅ Inputs are semantically similar to training data
- **0.2 - 0.4 (Yellow)**: ⚠️ Inputs are drifting - model may struggle
- **> 0.4 (Red)**: 🚨 Inputs are very different - model is likely out of its depth

### What causes drift
- Users asking about new topics
- Language/vocabulary changes
- Different user demographics
- Domain expansion beyond original scope

### Why it matters more than PSI
PSI looks at statistical distribution of confidence scores. Embedding drift looks at the actual **meaning** of the text, catching semantic shifts that PSI might miss.

---

## 📝 PANEL 9: Average Query Length
**Metric Type:** Gauge  
**Query:** `avg_query_length`

### What it means
Average number of tokens/words in user queries.

### When it's useful
- **Sudden increase**: 
  - 🔍 Users are writing longer, more detailed queries
  - ⚠️ May increase latency and computational cost
  - ✅ Could indicate more engaged users
- **Sudden decrease**:
  - 🔍 Users writing shorter queries
  - ⚠️ May indicate frustration or low engagement
  - ⚠️ Could be bot traffic

### Why monitor this
- Longer queries may require more processing time
- Helps you understand user behavior patterns
- Can indicate if users are providing enough context
- Useful for capacity planning

---

## ❓ PANEL 10: UNK Token Ratio
**Metric Type:** Gauge  
**Query:** `unk_token_ratio`

### What it means
Percentage of unknown/out-of-vocabulary tokens in user inputs.

### When it's useful
- **< 0.1 (10%)**: ✅ Good - most words are in vocabulary
- **0.1 - 0.3 (10-30%)**: ⚠️ Concerning - users using unfamiliar terms
- **> 0.3 (30%)**: 🚨 Critical - model can't understand most inputs

### What causes high UNK ratio
- Users writing in different language
- New terminology/jargon emerged
- Typos and misspellings
- Domain shift (e.g., medical users on a general model)
- Spam or adversarial inputs

### Action items when high
- Expand vocabulary with recent data
- Implement spell-check preprocessing
- Add subword tokenization (BPE, WordPiece)
- Consider retraining with updated corpus

---

## 🧩 PANEL 11: Module Prediction Distribution
**Metric Type:** Counter (aggregated)  
**Query:** `sum by (module) (module_prediction_total)`

### What it means
Breakdown of which modules/intents/classes your model is predicting most frequently.

### When it's useful
- **Even distribution**: ✅ Model handles all modules equally
- **One module dominates (>80%)**: 
  - 🔍 Users heavily prefer one feature
  - ⚠️ Other modules may need improvement or marketing
  - ⚠️ Could indicate mis-classification bias
- **New module suddenly popular**:
  - ✅ Successful new feature
  - 🔍 Investigate if it's real user demand or misclassification

### Business insights
- Understand which features users care about
- Identify underutilized modules
- Detect if model is over-predicting certain classes
- Guide product development priorities

---

## 📅 PANEL 12: Date Prediction Distribution
**Metric Type:** Counter (aggregated)  
**Query:** `sum by (date) (date_prediction_total)`

### What it means
Breakdown of predictions over time (daily/weekly patterns).

### When it's useful
- **Steady growth**: ✅ Healthy user adoption
- **Weekday vs weekend patterns**: 🔍 Understand usage patterns for capacity planning
- **Sudden spike on one day**: 
  - 🔍 Marketing campaign success
  - 🔍 Media coverage
  - ⚠️ Potential bot attack
- **Declining trend**: 🚨 User churn - investigate urgently

### Use cases
- Capacity planning and auto-scaling schedules
- A/B test result analysis
- Detect outages (sudden drops)
- Seasonal pattern recognition

---

## 🎯 Quick Decision Matrix

| Symptom | Check These Metrics | Likely Issue |
|---------|-------------------|--------------|
| Users complaining about slow responses | P95 Latency, Average Latency, QPS | Resource bottleneck |
| Predictions seem wrong lately | Confidence PSI, Embedding Drift | Data drift - retrain needed |
| Model saying "I don't know" often | Low Confidence Count, Prediction Entropy | Out-of-domain queries |
| Strange text in queries | UNK Token Ratio | Vocabulary drift or spam |
| Traffic disappeared | Total Requests, QPS | Service outage |
| One feature dominates | Module Distribution | Classification bias or user preference |

---

## 🚨 Alert Recommendations

Set up alerts for:
1. **P95 Latency > 0.5s** for 5 minutes → Page on-call
2. **Confidence PSI > 0.25** → Email ML team
3. **Embedding Drift > 0.4** → Slack alert
4. **QPS drops to 0** for 2 minutes → Page on-call
5. **UNK Token Ratio > 0.3** → Daily digest
6. **Low Confidence spike** (>100 in 5 min) → Slack alert