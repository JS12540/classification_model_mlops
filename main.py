from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="JayShah07/reporting-multiclass",
    tokenizer="JayShah07/reporting-multiclass",
    return_all_scores=True
)

app = FastAPI(title="Reporting Intent Classifier")

DATE_LABELS = {
    "Daily", "Weekly", "Monthly", "Yearly",
    "Current Year", "Previous Year", "None_date"
}

REPORT_TYPE_LABELS = {
    "capital_gains",
    "investment_account_wise_returns",
    "scheme_wise_returns",
    "holdings",
    "portfolio_update",
    "None_module",
}


def split_intents(preds):
    """Split raw predictions into date + report-type best guesses."""
    # preds: list of {"label": ..., "score": ...}
    date_candidates = [p for p in preds if p["label"] in DATE_LABELS]
    report_candidates = [p for p in preds if p["label"] in REPORT_TYPE_LABELS]

    date_label = None
    report_label = None

    if date_candidates:
        date_candidates = sorted(date_candidates, key=lambda x: x["score"], reverse=True)
        date_label = date_candidates[0]

    if report_candidates:
        report_candidates = sorted(report_candidates, key=lambda x: x["score"], reverse=True)
        report_label = report_candidates[0]

    return date_label, report_label


@app.get("/", response_class=HTMLResponse)
async def home():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Reporting Intent Classifier</title>
        <meta charset="utf-8" />
        <style>
            body { font-family: Arial, sans-serif; max-width: 700px; margin: 40px auto; }
            h1 { color: #333; }
            textarea { width: 100%; height: 120px; padding: 8px; font-size: 14px; }
            button { margin-top: 12px; padding: 8px 16px; font-size: 14px; cursor: pointer; }
            .result { margin-top: 20px; padding: 10px; border-radius: 6px; background: #f5f5f5; }
            .label { font-weight: bold; }
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
    return html_content


@app.post("/predict", response_class=HTMLResponse)
async def predict(text: str = Form(...)):
    if not text.strip():
        result_html = "<p>Please enter some text.</p>"
    else:
        raw_preds = classifier(text)[0]  # list[dict]
        # Sort all scores for display
        all_sorted = sorted(raw_preds, key=lambda x: x["score"], reverse=True)

        date_label, report_label = split_intents(raw_preds)

        def format_label(label_dict):
            if label_dict is None:
                return "N/A"
            return f"{label_dict['label']} (conf: {label_dict['score']:.4f})"

        result_html = f"""
        <div class="result">
            <div><span class="label">Detected Report Type:</span> {format_label(report_label)}</div>
            <div><span class="label">Detected Date Period:</span> {format_label(date_label)}</div>
            <hr/>
            <div><span class="label">All scores:</span></div>
            <ul>
                {''.join([f"<li>{p['label']}: {p['score']:.4f}</li>" for p in all_sorted])}
            </ul>
        </div>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Reporting Intent Classifier - Result</title>
        <meta charset="utf-8" />
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 700px; margin: 40px auto; }}
            h1 {{ color: #333; }}
            textarea {{ width: 100%; height: 120px; padding: 8px; font-size: 14px; }}
            button {{ margin-top: 12px; padding: 8px 16px; font-size: 14px; cursor: pointer; }}
            .result {{ margin-top: 20px; padding: 10px; border-radius: 6px; background: #f5f5f5; }}
            .label {{ font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>Reporting Intent Classifier</h1>
        <form method="post" action="/predict">
            <label for="text">Enter a user query:</label><br/>
            <textarea id="text" name="text">{text}</textarea><br/>
            <button type="submit">Classify</button>
        </form>
        {result_html}
    </body>
    </html>
    """
    return html


@app.post("/api/predict")
async def api_predict(payload: dict):
    text = payload.get("text", "")
    raw_preds = classifier(text)[0]

    date_label, report_label = split_intents(raw_preds)
    all_sorted = sorted(raw_preds, key=lambda x: x["score"], reverse=True)

    return {
        "input": text,
        "report_type": report_label,
        "date_period": date_label,
        "all_scores": all_sorted,
    }
