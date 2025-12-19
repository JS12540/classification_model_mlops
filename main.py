"""
TinyBERT Dual Classifier – Production FastAPI App
"""

# ============================
# Imports
# ============================

import json
import base64
import io
import logging
from typing import Dict, List

import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field


# ============================
# Logging
# ============================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("reporting-intent")


# ============================
# Pydantic Schemas
# ============================

class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        example="Show my holdings this month",
        description="User query for reporting intent classification",
    )


class PredictResponse(BaseModel):
    module_best: str = Field(
        ...,
        example="holdings",
        description="Predicted reporting module",
    )
    date_best: str = Field(
        ...,
        example="Monthly",
        description="Predicted date period",
    )
    module_probs: Dict[str, float] = Field(
        ...,
        example={"holdings": 0.91, "capital_gains": 0.03},
        description="Probability distribution over module labels",
    )
    date_probs: Dict[str, float] = Field(
        ...,
        example={"Monthly": 0.88, "Yearly": 0.05},
        description="Probability distribution over date labels",
    )


# ============================
# Tokenizer
# ============================

class SimpleTokenizer:
    def __init__(self, vocab_path: str, config_path: str):
        logger.info("Loading tokenizer")
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
                start = 0
                while start < len(word):
                    end = len(word)
                    found = False
                    while start < end:
                        sub = word[start:end]
                        if start > 0:
                            sub = "##" + sub
                        if sub in self.vocab:
                            tokens.append(sub)
                            found = True
                            break
                        end -= 1
                    if not found:
                        tokens.append(self.unk_token)
                        start += 1
                    else:
                        start = end
        return tokens

    def encode(self, text: str):
        tokens = self.tokenize(text)

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


# ============================
# Model
# ============================

class TinyBERTDualClassifierONNX:
    def __init__(self, model_path, vocab_path, tokenizer_config, labels_path):
        logger.info("Loading ONNX model")
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.tokenizer = SimpleTokenizer(vocab_path, tokenizer_config)

        with open(labels_path) as f:
            labels = json.load(f)

        self.module_labels = labels["module_labels"]
        self.date_labels = labels["date_labels"]

        logger.info("Model loaded successfully")

    @staticmethod
    def softmax(x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def predict_all(self, text: str) -> Dict:
        logger.info(f"Inference request: {text}")

        input_ids, attention_mask = self.tokenizer.encode(text)

        module_logits, date_logits = self.session.run(
            None,
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )

        module_probs = self.softmax(module_logits[0])
        date_probs = self.softmax(date_logits[0])

        result = {
            "module_best": self.module_labels[int(np.argmax(module_probs))],
            "date_best": self.date_labels[int(np.argmax(date_probs))],
            "module_probs": dict(
                zip(self.module_labels, map(float, module_probs))
            ),
            "date_probs": dict(
                zip(self.date_labels, map(float, date_probs))
            ),
        }

        logger.info(
            f"Prediction → module={result['module_best']} | date={result['date_best']}"
        )

        return result


# ============================
# Graph Utils
# ============================

def plot_probs(title: str, probs: Dict[str, float]) -> str:
    labels = list(probs.keys())
    values = list(probs.values())

    plt.figure(figsize=(6, 4))
    plt.barh(labels, values)
    plt.xlim(0, 1)
    plt.title(title)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()

    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ============================
# FastAPI App
# ============================

app = FastAPI(
    title="Reporting Intent Classifier",
    version="1.0.0",
    description="TinyBERT dual classifier for reporting intent",
)

classifier = TinyBERTDualClassifierONNX(
    model_path="tinybert_dual_classifier_quantized.onnx",
    vocab_path="vocab.json",
    tokenizer_config="tokenizer_config.json",
    labels_path="labels.json",
)


# ============================
# Routes
# ============================

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
    <body style="font-family:Arial; max-width:700px; margin:40px auto">
      <h1>Reporting Intent Classifier</h1>
      <form method="post" action="/predict">
        <textarea name="text" style="width:100%;height:120px"></textarea><br/>
        <button style="margin-top:10px">Classify</button>
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
    <html>
    <body style="font-family:Arial; max-width:800px; margin:40px auto">
      <h2>Input</h2>
      <p>{text}</p>

      <h3>Prediction</h3>
      <p><b>Module:</b> {result['module_best']}</p>
      <p><b>Date:</b> {result['date_best']}</p>

      <h3>Module Probabilities</h3>
      <img src="data:image/png;base64,{module_img}" />

      <h3>Date Probabilities</h3>
      <img src="data:image/png;base64,{date_img}" />

      <br/><br/>
      <a href="/">← Try another</a>
    </body>
    </html>
    """


@app.post(
    "/api/predict",
    response_model=PredictResponse,
    summary="Predict reporting intent",
    tags=["Inference"],
)
async def api_predict(request: PredictRequest):
    return classifier.predict_all(request.text)


@app.get("/health")
async def health():
    return {"status": "ok"}
