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
    return """<form method="post" action="/predict">
    <textarea name="text"></textarea><button>Run</button></form>"""

@app.post("/predict", response_class=HTMLResponse)
async def predict(text: str = Form(...)):
    result = classifier.predict_all(text)
    return f"<p>{result['module_best']} | {result['date_best']}</p>"

@app.post("/api/predict", response_model=PredictResponse)
async def api_predict(request: PredictRequest):
    return classifier.predict_all(request.text)

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
