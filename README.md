# 🧠 Financial News Sentiment & Backtesting Framework

This repository contains a comprehensive pipeline for performing sentiment analysis on financial news and tweets, followed by quantitative strategy backtesting using the sentiment signals.

---

## 📁 Folder Structure Overview

```
└── 📁Sentimental-Analysis
    └── 📁Backtesting
        ├── V0.ipynb
        ├── V1.ipynb
        ├── V2.ipynb
        ├── V3.ipynb
        ├── V4.ipynb
    └── 📁data
        └── 📁News Articles
            └── 📁Bloomberg
                ├── BloombergNews.csv
                ├── BloombergNews1.csv
                ├── BloombergNews100.csv
                ├── distilbert_BloombergNews100.csv
            └── 📁Labelled
                ├── Grok_labelled.csv
                ├── Kraggle_labelled.csv
            └── 📁processed
                ├── Bloomberg_with_scores.csv
                ├── grok_processed.csv
                ├── trump_preprocessed_tagged_filtered.csv
                ├── trump_processed_tagged_scores.csv
                ├── trump_processed_tagged.csv
                ├── trump_processed.csv
                ├── trump_rough_working_excel_only.xlsx
            └── 📁raw
                ├── trump_scrapped_v1.csv
                ├── trump_scrapped_v2.csv
        └── 📁Results Charts Models
            └── 📁Backtesting
                └── 📁V1
                    ├── V1 grp_df.csv
                    ├── V1 Plot.png
                    ├── V1 Strat1 Plot.png
                    ├── V1 Strat2 Plot.png
                └── 📁V2
                    ├── V2 grp_df.csv
                    ├── V2 Strat1 Plot.png
                    ├── V2 Strat2 Plot.png
                └── 📁V3
                    └── 📁S1
                        ├── DistilBERT_plot.png
                        ├── FinBERT_plot.png
                        ├── Flair_plot.png
                        ├── GBDT_plot.png
                        ├── KNN_plot.png
                        ├── LR_plot.png
                        ├── NB_plot.png
                        ├── RF_plot.png
                        ├── SVC_plot.png
                        ├── Vader_plot.png
                    ├── plot.png
            └── 📁LDA html charts
                ├── LDA bloomberg.html
                ├── LDA negative sentiment.html
                ├── LDA neutral sentiment.html
                ├── LDA positive sentiment.html
            └── 📁Library Comparison Results
                ├── result_bertweet.csv
                ├── result_distilbert.csv
                ├── result_finbert.csv
                ├── result_flair.csv
                ├── result_roberta.csv
                ├── result_tb.csv
                ├── result_vader.csv
            └── 📁Models
                ├── ann_tfidf_custom_model.keras
                ├── custom_tokenizer.pkl
                ├── custom_vectorizer.pkl
                ├── lstm_custom_model.keras
                ├── optimised_lstm_label_dict.pkl
                ├── optimised_lstm_tokenizer.pkl
                ├── optimised_lstm_vocabulary.pkl
                ├── Optimized_LSTM (0.77 Acc).keras
            ├── trump_linear_regression_summary_3to1.csv
        └── 📁Stock Data
            ├── SPY 1min with sentiments.csv
            ├── SPY 1min.xlsx
            ├── SPY daily 08-11.csv
            ├── SPY daily 15-25 with technical indicators.csv
            ├── SPY_2006_2013_daily.csv
            ├── SPY_rough_working_excel_only.xlsx
    └── 📁LDA
        ├── BoW Dict (Negative).ipynb
        ├── BoW Dict (Neutral).ipynb
        ├── BoW Dict (Positive).ipynb
        ├── Topic Categorization (Bloomberg).ipynb
    └── 📁Misc
        ├── Article Formatting.ipynb
        ├── download datasets.ipynb
        ├── Processing.py
        ├── Scraping.py
        ├── Tagging.py
    └── 📁Sentiment Analysis
        ├── Bloomberg analysis.ipynb
        ├── Comparisons.ipynb
        ├── Neural Network.ipynb
        ├── Python Libraries.ipynb
        ├── Traditional ML models.ipynb
        ├── Trump tweet analysis.ipynb
    └── README.md
```


---

## 🔁 Backtesting

Contains iterative strategy development notebooks (`V0` to `V4`) that use sentiment signals derived from the processed data.

- **V0–V4.ipynb**: Backtesting strategies using different model outputs and sentiment weightings.

---

## 📊 data

All the datasets used across the pipeline.

### 📁 News Articles
Organized news data scraped from multiple sources.

- **Bloomberg/**: Raw and labeled Bloomberg news CSVs, including sentiment-scored versions using models like DistilBERT.
- **Labelled/**: Human-annotated or semi-automatically labeled news (e.g., Grok, Kraggle).
- **processed/**: Cleaned, tagged, and model-scored news datasets.
- **raw/**: Raw scraped Trump tweets and news headlines.

### 📁 Results Charts Models

Outputs from backtesting and model evaluation.

- **Backtesting/**: Plots and grouped performance metrics (Strat1, Strat2) across versions.
- **LDA html charts/**: Topic modeling visualizations from LDA.
- **Library Comparison Results/**: Performance comparisons of sentiment models like Flair, FinBERT, Vader, DistilBERT, etc.
- **Models/**: Serialized models (LSTM, ANN, TF-IDF), tokenizers, and vectorizers.
- `trump_linear_regression_summary_3to1.csv`: Summary statistics from regression on tweet sentiment vs stock return.

### 📁 Stock Data

SPY (S&P 500 ETF) data used for backtesting.

- Intraday and daily SPY data from 2006–2025.
- CSVs with technical indicators and joined sentiment signals.

---

## 🧠 LDA (Latent Dirichlet Allocation)

Topic modeling notebooks using Bag-of-Words dictionaries to analyze sentiment-based topic clusters:

- Topic models for **positive**, **neutral**, and **negative** sentiment.
- Categorization for Bloomberg data.

---

## 🧰 Misc

Utility notebooks and scripts used throughout the project:

- `Scraping.py`, `Tagging.py`: Scripts for scraping and tagging raw data.
- `Processing.py`: Centralized preprocessing logic.
- `download datasets.ipynb`: For downloading and organizing large-scale news datasets.

---

## 💬 Sentiment Analysis

Jupyter notebooks implementing sentiment analysis on financial text using various ML and NLP approaches:

- `Bloomberg analysis.ipynb`: Focused analysis on Bloomberg news.
- `Trump tweet analysis.ipynb`: Trump tweets sentiment vs market.
- `Traditional ML models.ipynb`: SVM, Naive Bayes, etc.
- `Neural Network.ipynb`: FFNN or LSTM-based classifiers.
- `Python Libraries.ipynb`, `Comparisons.ipynb`: Model benchmarking.

---

## 📌 How to Use

1. **Data Preparation**  
   Run scripts in `Misc/` to clean and preprocess raw data.

2. **Sentiment Tagging**  
   Use notebooks in `Sentiment Analysis/` to apply and compare models.

3. **Topic Modeling**  
   Use `LDA/` to explore thematic structure of financial text.

4. **Backtest Strategies**  
   Execute notebooks in `Backtesting/` using sentiment-tagged data.

---

## 🧪 Requirements

Install packages with:

```bash
pip install -r requirements.txt

Commonly used libraries: pandas, numpy, sklearn, matplotlib, seaborn, nltk, transformers, Flair, gensim, backtrader