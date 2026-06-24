# Financial News Sentiment & Backtesting Framework

A research pipeline for performing sentiment analysis on financial news and tweets, followed by quantitative strategy backtesting using the derived sentiment signals.

---

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

---

## Overview

This project explores whether NLP-derived sentiment signals from financial news (Bloomberg) and social media (Trump tweets) can predict short-term equity market movements. It covers the full pipeline:

1. **Scraping** — collecting raw news and tweet data
2. **Preprocessing** — cleaning, tokenising, and tagging text
3. **Sentiment Analysis** — comparing rule-based (VADER), traditional ML (SVM, Naive Bayes), and transformer-based models (FinBERT, DistilBERT, Flair, RoBERTa)
4. **Topic Modelling** — LDA topic extraction segmented by sentiment class
5. **Backtesting** — using sentiment signals to drive SPY trading strategies

---

## Prerequisites

- **Python 3.10+** (tested on Python 3.12)
- **pip** (bundled with Python)
- A virtual environment manager (`venv` or `conda`) is strongly recommended
- **CUDA-capable GPU** (optional but significantly speeds up transformer inference in `Misc/Tagging.py`)

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/jaytai3336/Sentimental-Analysis.git
cd Sentimental-Analysis

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the spaCy language model (required by Misc/Tagging.py)
python -m spacy download en_core_web_sm

# 5. Download NLTK data (required by Misc/processing.py)
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger_eng'); nltk.download('maxent_ne_chunker_tab'); nltk.download('words'); nltk.download('wordnet'); nltk.download('stopwords')"
```

---

## Project Structure

```
Sentimental-Analysis/
├── Backtesting/
│   ├── V0.ipynb              # Initial strategy prototype
│   ├── V1.ipynb
│   ├── V2.ipynb
│   ├── V3.ipynb
│   └── V4.ipynb              # Most recent backtesting version
├── LDA/
│   ├── BoW Dict (Negative).ipynb
│   ├── BoW Dict (Neutral).ipynb
│   ├── BoW Dict (Positive).ipynb
│   └── Topic Categorization (Bloomberg).ipynb
├── Misc/
│   ├── Article Formatting.ipynb
│   ├── download datasets.ipynb
│   ├── processing.py         # Text preprocessing (tokenisation, lemmatisation, NER)
│   ├── Scraping.py           # Selenium-based web scraper
│   └── Tagging.py            # Zero-shot topic classification + embeddings
├── Sentiment Analysis/
│   ├── Bloomberg analysis.ipynb
│   ├── Comparisons.ipynb     # Cross-model benchmarking
│   ├── Neural Network.ipynb  # LSTM / ANN classifiers
│   ├── Python Libraries.ipynb
│   ├── Traditional ML models.ipynb
│   └── Trump tweet analysis.ipynb
├── data/
│   ├── News Articles/
│   │   ├── Bloomberg/        # Raw and DistilBERT-scored Bloomberg news CSVs
│   │   ├── Labelled/         # Grok and Kaggle labelled datasets
│   │   ├── processed/        # Cleaned, tagged, and model-scored datasets
│   │   └── raw/              # Raw scraped Trump tweets
│   ├── Results Charts Models/
│   │   ├── Backtesting/      # Backtesting plots and grouped performance CSVs
│   │   ├── LDA html charts/  # Interactive pyLDAvis HTML outputs
│   │   ├── Library Comparison Results/  # Per-model result CSVs
│   │   └── Models/           # Serialised Keras models, tokenisers, vectorisers
│   └── Stock Data/           # SPY intraday and daily OHLCV data (2006–2025)
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Usage

Run the pipeline steps in order. Each script and notebook contains inline comments describing what to edit for your own data sources.

### 1. Data Collection

Edit the target URL and CSS selectors in `Misc/Scraping.py`, then run:

```bash
python Misc/Scraping.py
```

> **Note:** `Scraping.py` requires Microsoft Edge and a matching `msedgedriver` binary. Set the `PATH` variable at the top of the file to the path of your driver executable.

### 2. Preprocessing

Edit the input CSV path at the top of `Misc/processing.py`, then run:

```bash
python Misc/processing.py
```

### 3. Topic Tagging & Embeddings

Edit the input CSV path in `Misc/Tagging.py`, then run:

```bash
python Misc/Tagging.py
```

> Transformer inference is GPU-accelerated automatically when CUDA is available.

### 4. Sentiment Analysis

Open and run the notebooks in `Sentiment Analysis/` in Jupyter:

```bash
jupyter notebook "Sentiment Analysis/Python Libraries.ipynb"
```

Key notebooks:
- `Python Libraries.ipynb` — compares VADER, TextBlob, FinBERT, DistilBERT, Flair, RoBERTa, BERTweet
- `Traditional ML models.ipynb` — SVM, Naive Bayes, KNN, Random Forest, GBDT
- `Neural Network.ipynb` — LSTM and ANN with custom TF-IDF features
- `Bloomberg analysis.ipynb` — full analysis on Bloomberg news data
- `Trump tweet analysis.ipynb` — tweet sentiment vs SPY returns

### 5. Topic Modelling (LDA)

Run notebooks in `LDA/` to generate interactive topic visualisations (saved to `data/Results Charts Models/LDA html charts/`).

### 6. Backtesting

Run notebooks in `Backtesting/` (V0 → V4) using the sentiment-tagged data. V4 is the most recent strategy.

---

## Results

Selected findings:

| Model | Accuracy |
|-------|----------|
| Optimised LSTM | 0.77 |
| FinBERT | — (see Comparisons.ipynb) |
| VADER | — (see Comparisons.ipynb) |

Backtesting plots and per-model result CSVs are stored under `data/Results Charts Models/`.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
