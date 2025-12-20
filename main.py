"""
TinyBERT Dual Classifier – Production FastAPI App
WITH DATA + CONCEPT + PSI + EMBEDDING DRIFT METRICS
"""
import json
import base64
import io
import logging
import time
import math
from typing import Dict, List, Deque
from collections import deque

import numpy as np
import onnxruntime as ort
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# ---------------- PROMETHEUS ----------------
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("reporting-intent")

# ---------------- METRICS ----------------

MODEL_INFERENCE_TOTAL = Counter("model_inference_total", "Total model inferences")
MODEL_INFERENCE_LATENCY = Histogram("model_inference_latency_seconds", "Inference latency")

AVG_QUERY_LENGTH = Gauge("avg_query_length", "Average query length")
UNK_TOKEN_RATIO = Gauge("unk_token_ratio", "UNK token ratio")

PREDICTION_ENTROPY = Gauge("prediction_entropy", "Prediction entropy")
MODEL_LOW_CONFIDENCE_TOTAL = Counter("model_low_confidence_total", "Low confidence preds")

MODULE_PREDICTION_TOTAL = Counter(
    "module_prediction_total", "Module predictions", ["module"]
)
DATE_PREDICTION_TOTAL = Counter(
    "date_prediction_total", "Date predictions", ["date"]
)

# ---- ADVANCED DRIFT ----
CONFIDENCE_PSI = Gauge("confidence_psi", "PSI for confidence score")
EMBEDDING_DRIFT = Gauge("embedding_drift_score", "Embedding drift score")

# ---------------- PSI UTIL ----------------
def calculate_psi(expected, actual, bins=10):
    expected = np.array(expected)
    actual = np.array(actual)

    if len(expected) < bins or len(actual) < bins:
        return 0.0

    quantiles = np.linspace(0, 100, bins + 1)
    breakpoints = np.percentile(expected, quantiles)

    psi = 0.0
    for i in range(len(breakpoints) - 1):
        exp_pct = np.mean((expected >= breakpoints[i]) & (expected < breakpoints[i+1]))
        act_pct = np.mean((actual >= breakpoints[i]) & (actual < breakpoints[i+1]))
        if exp_pct > 0 and act_pct > 0:
            psi += (act_pct - exp_pct) * math.log(act_pct / exp_pct)

    return psi

# ---------------- GRAPH UTILS ----------------
def plot_probs(title: str, probs: Dict[str, float]) -> str:
    labels = list(probs.keys())
    values = list(probs.values())

    plt.figure(figsize=(8, 5))
    plt.barh(labels, values, color='#4A90E2')
    plt.xlim(0, 1)
    plt.xlabel('Probability')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches='tight')
    plt.close()

    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ---------------- API MODELS ----------------
class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        example="Show my holdings this month",
        description="User query for reporting intent classification",
    )


class PredictResponse(BaseModel):
    module_best: str
    date_best: str
    module_probs: Dict[str, float]
    date_probs: Dict[str, float]

# ---------------- TOKENIZER ----------------
class SimpleTokenizer:
    def __init__(self, vocab_path: str, config_path: str):
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.max_length = self.config["max_length"]
        self.cls_token_id = self.config["cls_token_id"]
        self.sep_token_id = self.config["sep_token_id"]
        self.pad_token_id = self.config["pad_token_id"]
        self.unk_token_id = self.config["unk_token_id"]
        self.unk_token = self.config["unk_token"]

    def tokenize(self, text: str) -> List[str]:
        text = text.lower().strip()
        tokens = []

        for word in text.split():
            if word in self.vocab:
                tokens.append(word)
            else:
                tokens.append(self.unk_token)
        return tokens

    def encode(self, text: str):
        tokens = self.tokenize(text)

        AVG_QUERY_LENGTH.set(len(text))
        UNK_TOKEN_RATIO.set(tokens.count(self.unk_token) / max(len(tokens), 1))

        token_ids = [self.cls_token_id]
        token_ids += [self.vocab.get(t, self.unk_token_id) for t in tokens]
        token_ids.append(self.sep_token_id)

        token_ids = token_ids[: self.max_length]
        attention_mask = [1] * len(token_ids)

        padding = self.max_length - len(token_ids)
        token_ids += [self.pad_token_id] * padding
        attention_mask += [0] * padding

        return (
            np.array([token_ids], dtype=np.int64),
            np.array([attention_mask], dtype=np.int64),
        )

# ---------------- MODEL ----------------
class TinyBERTDualClassifierONNX:
    def __init__(self, model_path, vocab_path, tokenizer_config, labels_path):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.tokenizer = SimpleTokenizer(vocab_path, tokenizer_config)

        with open(labels_path) as f:
            labels = json.load(f)

        self.module_labels = labels["module_labels"]
        self.date_labels = labels["date_labels"]

        self.confidence_baseline: Deque[float] = deque(maxlen=500)
        self.confidence_live: Deque[float] = deque(maxlen=500)

        self.embedding_baseline = None

    @staticmethod
    def softmax(x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    @staticmethod
    def entropy(p):
        return -sum(x * math.log(x + 1e-9) for x in p)

    def predict_all(self, text: str) -> Dict:
        start = time.time()
        MODEL_INFERENCE_TOTAL.inc()

        input_ids, attention_mask = self.tokenizer.encode(text)

        module_logits, date_logits = self.session.run(
            None, {"input_ids": input_ids, "attention_mask": attention_mask}
        )

        module_probs = self.softmax(module_logits[0])
        date_probs = self.softmax(date_logits[0])

        max_conf = max(np.max(module_probs), np.max(date_probs))

        self.confidence_live.append(max_conf)
        if len(self.confidence_baseline) < 500:
            self.confidence_baseline.append(max_conf)

        if len(self.confidence_baseline) == 500:
            psi = calculate_psi(self.confidence_baseline, self.confidence_live)
            CONFIDENCE_PSI.set(psi)

        PREDICTION_ENTROPY.set(
            self.entropy(module_probs) + self.entropy(date_probs)
        )

        if max_conf < 0.5:
            MODEL_LOW_CONFIDENCE_TOTAL.inc()

        # ---- EMBEDDING DRIFT (CLS PROXY) ----
        embedding = np.concatenate([module_logits[0], date_logits[0]])
        if self.embedding_baseline is None:
            self.embedding_baseline = embedding
        else:
            cos_sim = np.dot(embedding, self.embedding_baseline) / (
                np.linalg.norm(embedding) * np.linalg.norm(self.embedding_baseline)
            )
            EMBEDDING_DRIFT.set(1 - cos_sim)

        module_best = (
            self.module_labels[int(np.argmax(module_probs))]
            if np.max(module_probs) > 0.5 else "None_module"
        )
        date_best = (
            self.date_labels[int(np.argmax(date_probs))]
            if np.max(date_probs) > 0.5 else "None_date"
        )

        MODULE_PREDICTION_TOTAL.labels(module=module_best).inc()
        DATE_PREDICTION_TOTAL.labels(date=date_best).inc()

        MODEL_INFERENCE_LATENCY.observe(time.time() - start)

        return {
            "module_best": module_best,
            "date_best": date_best,
            "module_probs": dict(zip(self.module_labels, map(float, module_probs))),
            "date_probs": dict(zip(self.date_labels, map(float, date_probs))),
        }

# ---------------- FASTAPI ----------------
app = FastAPI(title="Reporting Intent Classifier", version="1.0.0")
Instrumentator().instrument(app).expose(app)

classifier = TinyBERTDualClassifierONNX(
    "tinybert_dual_classifier_quantized.onnx",
    "vocab.json",
    "tokenizer_config.json",
    "labels.json",
)

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Reporting Intent Classifier</title>
        <meta charset="utf-8" />
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; }
            h1 { color: #333; }
            textarea { width: 100%; height: 120px; padding: 8px; font-size: 14px; border: 1px solid #ccc; border-radius: 4px; }
            button { margin-top: 12px; padding: 10px 20px; font-size: 14px; cursor: pointer; background-color: #4A90E2; color: white; border: none; border-radius: 4px; }
            button:hover { background-color: #357ABD; }
            label { font-weight: bold; color: #555; }
        </style>
    </head>
    <body>
        <h1>Reporting Intent Classifier</h1>
        <form method="post" action="/predict">
            <label for="text">Enter a user query:</label><br/>
            <textarea id="text" name="text" placeholder="Type a query here..."></textarea><br/>
            <button type="submit">Classify</button>
        </form>
    </body>
    </html>
    """

@app.post("/predict", response_class=HTMLResponse)
async def predict(text: str = Form(...)):
    result = classifier.predict_all(text)

    module_img = plot_probs("Module Probabilities", result["module_probs"])
    date_img = plot_probs("Date Probabilities", result["date_probs"])

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Reporting Intent Classifier - Result</title>
        <meta charset="utf-8" />
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; }}
            h1 {{ color: #333; }}
            h2 {{ color: #555; margin-top: 30px; }}
            h3 {{ color: #666; margin-top: 20px; }}
            .input-box {{ background: #f9f9f9; padding: 15px; border-radius: 6px; margin-bottom: 20px; }}
            .prediction-box {{ background: #e8f4f8; padding: 15px; border-radius: 6px; margin-bottom: 20px; }}
            .prediction-box p {{ margin: 8px 0; font-size: 16px; }}
            .prediction-box b {{ color: #333; }}
            img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; margin-top: 10px; }}
            a {{ color: #4A90E2; text-decoration: none; font-size: 16px; }}
            a:hover {{ text-decoration: underline; }}
        </style>
    </head>
    <body>
        <h1>Reporting Intent Classifier</h1>
        
        <h2>Input Query</h2>
        <div class="input-box">
            <p>{text}</p>
        </div>

        <h2>Predictions</h2>
        <div class="prediction-box">
            <p><b>Module:</b> {result['module_best']}</p>
            <p><b>Date:</b> {result['date_best']}</p>
        </div>

        <h3>Module Probabilities</h3>
        <img src="data:image/png;base64,{module_img}" alt="Module Probabilities" />

        <h3>Date Probabilities</h3>
        <img src="data:image/png;base64,{date_img}" alt="Date Probabilities" />

        <br/><br/>
        <a href="/">← Try another query</a>
    </body>
    </html>
    """

@app.post("/api/predict", response_model=PredictResponse)
async def api_predict(request: PredictRequest):
    return classifier.predict_all(request.text)

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)