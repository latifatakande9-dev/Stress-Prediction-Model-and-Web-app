import csv
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Stress.csv"
MAX_TEXT_LENGTH = 10_000
LABELS = {"0": "No stress", "1": "Stress"}

def train_model() -> Pipeline:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    texts = []
    labels = []
    with DATA_PATH.open(encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames or not {"text", "label"}.issubset(reader.fieldnames):
            raise ValueError("Stress.csv must contain 'text' and 'label' columns.")

        for row_number, row in enumerate(reader, start=2):
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip()
            if not text:
                continue
            if label not in LABELS:
                raise ValueError(f"Unexpected label {label!r} in CSV row {row_number}.")
            texts.append(text)
            labels.append(LABELS[label])

    if not texts or len(set(labels)) < 2:
        raise ValueError("Stress.csv must contain text examples for both labels.")

    classifier = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    classifier.fit(texts, labels)
    return classifier


model = train_model()
app = Flask(__name__)


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/predict")
def predict():
    is_json_request = request.is_json
    if is_json_request:
        payload = request.get_json(silent=True)
        text = payload.get("text") if isinstance(payload, dict) else None
    else:
        text = request.form.get("text")

    if not isinstance(text, str) or not text.strip():
        message = "Please enter some text to analyze."
        if is_json_request:
            return jsonify({"error": message}), 400
        return render_template("index.html", error=message), 400

    if len(text) > MAX_TEXT_LENGTH:
        message = f"Text must be {MAX_TEXT_LENGTH:,} characters or fewer."
        if is_json_request:
            return jsonify({"error": message}), 400
        return render_template("index.html", error=message), 400

    prediction = str(model.predict([text])[0])
    if is_json_request:
        return jsonify({"prediction": prediction})
    return render_template("index.html", prediction=prediction, text=text)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)