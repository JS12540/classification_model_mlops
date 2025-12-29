"""
TinyBERT Dual Classifier – Production FastAPI App
WITH PER-REQUEST ANOMALY + PER-METRIC PSI + EMBEDDING DRIFT
"""

# =========================
# IMPORTS
# =========================
import json
import base64
import io
import logging
import time
import math
import threading
from collections import deque
from typing import Dict, List, Deque

import numpy as np
import onnxruntime as ort
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge

# =========================
# LOGGING
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("reporting-intent")

# =========================
# METRICS
# =========================
MODEL_INFERENCE_TOTAL = Counter("model_inference_total", "Total model inferences")
MODEL_INFERENCE_LATENCY = Histogram("model_inference_latency_seconds", "Inference latency")

AVG_QUERY_LENGTH = Gauge("avg_query_length", "Average query length")
UNK_TOKEN_RATIO = Gauge("unk_token_ratio", "UNK token ratio")
PREDICTION_ENTROPY = Gauge("prediction_entropy", "Prediction entropy")

MODEL_LOW_CONFIDENCE_TOTAL = Counter("model_low_confidence_total", "Low confidence predictions")
MODULE_PREDICTION_TOTAL = Counter("module_prediction_total", "Module predictions", ["module"])
DATE_PREDICTION_TOTAL = Counter("date_prediction_total", "Date predictions", ["date"])

# ---- DRIFT METRICS ----
MODULE_PSI = Gauge("module_confidence_psi", "Module confidence PSI")
DATE_PSI = Gauge("date_confidence_psi", "Date confidence PSI")
EMBEDDING_POINT_DRIFT = Gauge("embedding_point_drift", "Per-request embedding cosine distance vs training baseline")
EMBEDDING_DRIFT = Gauge("embedding_drift_score", "Aggregated embedding drift score")

# =========================
# PSI UTILITY
# =========================
def calculate_psi(expected, actual, bins=10):
    expected = np.array(expected)
    actual = np.array(actual)

    if len(actual) < bins:
        return 0.0

    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    psi = 0.0

    for i in range(len(breakpoints) - 1):
        exp_pct = np.mean((expected >= breakpoints[i]) & (expected < breakpoints[i + 1]))
        act_pct = np.mean((actual >= breakpoints[i]) & (actual < breakpoints[i + 1]))
        if exp_pct > 0 and act_pct > 0:
            psi += (act_pct - exp_pct) * math.log(act_pct / exp_pct)
    return psi

# =========================
# GRAPH UTILS
# =========================
def plot_probs(title: str, probs: Dict[str, float]) -> str:
    labels = list(probs.keys())
    values = list(probs.values())

    plt.figure(figsize=(8, 5))
    plt.barh(labels, values)
    plt.xlim(0, 1)
    plt.title(title)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    return base64.b64encode(buf.getvalue()).decode()

# =========================
# TOKENIZER
# =========================
class SimpleTokenizer:
    def __init__(self, vocab_path, config_path):
        with open(vocab_path) as f:
            self.vocab = json.load(f)
        with open(config_path) as f:
            cfg = json.load(f)

        self.max_length = cfg["max_length"]
        self.cls_token_id = cfg["cls_token_id"]
        self.sep_token_id = cfg["sep_token_id"]
        self.pad_token_id = cfg["pad_token_id"]
        self.unk_token_id = cfg["unk_token_id"]
        self.unk_token = cfg["unk_token"]

    def encode(self, text: str):
        tokens = text.lower().split()
        AVG_QUERY_LENGTH.set(len(text))
        UNK_TOKEN_RATIO.set(tokens.count(self.unk_token) / max(len(tokens), 1))

        ids = [self.cls_token_id] + [self.vocab.get(t, self.unk_token_id) for t in tokens] + [self.sep_token_id]
        ids = ids[: self.max_length]
        mask = [1] * len(ids)
        pad = self.max_length - len(ids)
        ids += [self.pad_token_id] * pad
        mask += [0] * pad

        return np.array([ids], dtype=np.int64), np.array([mask], dtype=np.int64)

# =========================
# MODEL
# =========================
class TinyBERTDualClassifierONNX:
    def __init__(self):
        self.session = ort.InferenceSession("tinybert_dual_classifier_quantized.onnx", providers=["CPUExecutionProvider"])
        self.tokenizer = SimpleTokenizer("vocab.json", "tokenizer_config.json")

        with open("labels.json") as f:
            labels = json.load(f)
        self.module_labels = labels["module_labels"]
        self.date_labels = labels["date_labels"]

        # ---- BASELINES ----
        self.module_conf_baseline = np.load("module_conf_baseline.npy")
        self.date_conf_baseline = np.load("date_conf_baseline.npy")
        self.embedding_baseline = np.load("embedding_baseline.npy")

        # ---- LIVE BUFFERS ----
        self.module_conf_live: Deque[float] = deque(maxlen=500)
        self.date_conf_live: Deque[float] = deque(maxlen=500)
        self.embedding_live: Deque[np.ndarray] = deque(maxlen=2000)

    @staticmethod
    def softmax(x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    @staticmethod
    def entropy(p):
        return -sum(v * math.log(v + 1e-9) for v in p)

    def predict_all(self, text: str):
        start = time.time()
        MODEL_INFERENCE_TOTAL.inc()

        input_ids, mask = self.tokenizer.encode(text)
        module_logits, date_logits = self.session.run(None, {"input_ids": input_ids, "attention_mask": mask})

        module_probs = self.softmax(module_logits[0])
        date_probs = self.softmax(date_logits[0])

        # ---- confidence ----
        module_max_conf = np.max(module_probs)
        date_max_conf = np.max(date_probs)
        max_conf = max(module_max_conf, date_max_conf)

        self.module_conf_live.append(module_max_conf)
        self.date_conf_live.append(date_max_conf)

        # ---- embedding ----
        embedding = np.concatenate([module_logits[0], date_logits[0]])

        # 🔹 Per-request embedding anomaly
        cos_sim = np.dot(embedding, self.embedding_baseline) / (np.linalg.norm(embedding) * np.linalg.norm(self.embedding_baseline) + 1e-9)
        EMBEDDING_POINT_DRIFT.set(1 - cos_sim)

        # 🔹 Store for aggregation
        self.embedding_live.append(embedding)

        PREDICTION_ENTROPY.set(self.entropy(module_probs) + self.entropy(date_probs))
        if max_conf < 0.5:
            MODEL_LOW_CONFIDENCE_TOTAL.inc()

        module_best = self.module_labels[int(np.argmax(module_probs))] if module_max_conf > 0.5 else "None_module"
        date_best = self.date_labels[int(np.argmax(date_probs))] if date_max_conf > 0.5 else "None_date"

        MODULE_PREDICTION_TOTAL.labels(module=module_best).inc()
        DATE_PREDICTION_TOTAL.labels(date=date_best).inc()

        MODEL_INFERENCE_LATENCY.observe(time.time() - start)

        return {
            "module_best": module_best,
            "date_best": date_best,
            "module_probs": dict(zip(self.module_labels, map(float, module_probs))),
            "date_probs": dict(zip(self.date_labels, map(float, date_probs))),
        }

# =========================
# BACKGROUND MONITOR
# =========================
def monitoring_worker(model):
    while True:
        time.sleep(60)
        # ---- MODULE PSI ----
        if len(model.module_conf_live) == model.module_conf_live.maxlen:
            psi = calculate_psi(model.module_conf_baseline, list(model.module_conf_live))
            MODULE_PSI.set(psi)
        # ---- DATE PSI ----
        if len(model.date_conf_live) == model.date_conf_live.maxlen:
            psi = calculate_psi(model.date_conf_baseline, list(model.date_conf_live))
            DATE_PSI.set(psi)
        # ---- EMBEDDING AGGREGATED DRIFT ----
        if len(model.embedding_live) > 100:
            live_mean = np.mean(model.embedding_live, axis=0)
            cos_sim = np.dot(live_mean, model.embedding_baseline) / (np.linalg.norm(live_mean) * np.linalg.norm(model.embedding_baseline) + 1e-9)
            EMBEDDING_DRIFT.set(1 - cos_sim)

# =========================
# FASTAPI
# =========================
app = FastAPI(title="Reporting Intent Classifier")
Instrumentator().instrument(app).expose(app)

classifier = TinyBERTDualClassifierONNX()

threading.Thread(target=monitoring_worker, args=(classifier,), daemon=True).start()

@app.post("/api/predict")
async def predict(req: Dict[str, str]):
    return classifier.predict_all(req["text"])

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
