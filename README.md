# Stress Detector

A text-based stress detection project that trains a machine-learning classifier on
`Stress.csv` and exposes predictions through a Flask web interface. The repository
also includes Jupyter notebooks for exploratory analysis and model experiments.

> This application provides an automated estimate only. It is not a medical
> diagnosis or a substitute for professional support.

## Project files

- `app.py` - Flask web server and text-classification API
- `templates/index.html` - Web interface
- `Stress.csv` - Text and binary stress labels used to train the app
- `stress model.ipynb` - Stress-model analysis notebook
- `Stress_Detection_with_Machine_Learning.ipynb` - Extended analysis and model notebook
- `requirements.csv` - Human-readable dependency inventory
- `requirements.txt` - Dependencies for pip installation

## Requirements

- Python 3.10 or newer
- JupyterLab or Jupyter Notebook for opening the notebooks

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m nltk.downloader stopwords
```

The web app can be run with just the core dependencies.

## Run the web app

From the project directory:

```bash
python app.py
```

Open `http://127.0.0.1:5000` in a browser.

The server trains the TF-IDF and logistic regression pipeline when `app.py`
starts. It reads `Stress.csv` and requires both label values:

- `0` - No stress
- `1` - Stress

## API usage

### Analyze text

**Endpoint:** `POST /predict`

JSON request:

```json
{
  "text": "I am feeling overwhelmed and unable to relax."
}
```

Example response:

```json
{
  "prediction": "Stress"
}
```

The text is limited to 10,000 characters. Empty text returns HTTP 400 with an
`error` field.

## Run the notebooks

Start Jupyter from the project directory:

```bash
jupyter notebook
```

Then open either notebook. Run cells from top to bottom. The notebooks may use
additional optional libraries listed in `requirements.csv`.

## Dataset

`Stress.csv` should contain at least a `text` column and a `label` column. Labels
must be `0` or `1`. Blank text rows are ignored. The first header is not modified
by the web application.
