# Financial News Sentiment Analysis & Trading Strategy Backtesting

## 📌 Overview
A comprehensive pipeline for analyzing sentiment in financial news and evaluating trading strategies based on derived sentiment signals. The system integrates multiple NLP techniques with quantitative finance methods to explore the relationship between news sentiment and market movements. 

## ✨ Key Features
- **Data Processing Pipeline**
  - News article formatting and normalization
  - Sentiment-label correlation analysis
- **Multi-model Sentiment Analysis**
  - 10+ pre-trained models (FinBERT, DistilBERT, RoBERTa, VADER, etc.)
  - Traditional ML classifiers (Random Forest, SVM, Logistic Regression)
  - Deep learning architectures (LSTM, ANN)
- **Topic Modeling**
  - LDA implementation for sentiment-categorized news
  - Visualization of key financial topics
- **Trading Strategy Backtesting**
  - Integrated sentiment-market analysis

## 📂 Project Structure
## Directory Hierarchy
```
|—— Backtesting
|    |—— V0.ipynb
|    |—— V1.ipynb
|    |—— V2.ipynb
|    |—— V3.ipynb
|—— data
|    |—— News Articles
|        |—— (Private dataset)
|        |—— Bloomberg
|            |—— BloombergNews.csv
|            |—— BloombergNews1.csv
|            |—— BloombergNews100.csv
|            |—— distilbert_BloombergNews100.csv
|        |—— Grok
|            |—— headlines_with_sentiment.csv
|    |—— Results Charts Models
|        |—— Backtesting Versions
|            |—— V1
|                |—— V1 grp_df.csv
|                |—— V1 Plot.png
|                |—— V1 Strat1 Plot.png
|                |—— V1 Strat2 Plot.png
|            |—— V2
|                |—— V2 grp_df.csv
|                |—— V2 Strat1 Plot.png
|                |—— V2 Strat2 Plot.png
|            |—— V3
|                |—— plot.png
|                |—— S1
|                    |—— DistilBERT_plot.png
|                    |—— FinBERT_plot.png
|                    |—— Flair_plot.png
|                    |—— GBDT_plot.png
|                    |—— KNN_plot.png
|                    |—— LR_plot.png
|                    |—— NB_plot.png
|                    |—— RF_plot.png
|                    |—— SVC_plot.png
|                    |—— Vader_plot.png
|        |—— Dictionaries
|            |—— glove.6B.100d.txt
|        |—— LDA html charts
|            |—— LDA bloomberg.html
|            |—— LDA negative sentiment.html
|            |—— LDA neutral sentiment.html
|            |—— LDA positive sentiment.html
|        |—— Library Comparison Results
|            |—— result_bertweet.csv
|            |—— result_distilbert.csv
|            |—— result_finbert.csv
|            |—— result_flair.csv
|            |—— result_roberta.csv
|            |—— result_tb.csv
|            |—— result_vader.csv
|        |—— Models
|            |—— ann_tfidf_custom_model.keras
|            |—— custom_tokenizer.pkl
|            |—— custom_vectorizer.pkl
|            |—— lstm_custom_model.keras
|            |—— optimised_lstm_label_dict.pkl
|            |—— optimised_lstm_tokenizer.pkl
|            |—— optimised_lstm_vocabulary.pkl
|            |—— Optimized_LSTM (0.77 Acc).keras
|    |—— Stock Data
|        |—— FNSPID dataset
|            |—— snp 5.3gb.csv
|            |—— SNP5gb.csv
|        |—— SNP 08-11.csv
|        |—— SnP futures intraday w Sentiments.csv
|        |—— SnP futures intraday.xlsx
|        |—— SNP500 Historical data.csv
|—— LDA
|    |—— BoW Dict (Negative).ipynb
|    |—— BoW Dict (Neutral).ipynb
|    |—— BoW Dict (Positive).ipynb
|    |—— Setup.ipynb
|    |—— Topic Categorization (Bloomberg).ipynb
|—— Misc
|    |—— Article Formatting.ipynb
|    |—— Downloading Bloomberg Financial News Dataset.ipynb
|—— README.md
|—— Sentiment Analysis
|    |—— Comparisons.ipynb
|    |—— Neural Network v1.ipynb
|    |—— Neural Network v2 (abandoned).ipynb
|    |—— Python Libraries.ipynb
|    |—— Traditional ML models.ipynb
```

## 📊 Results Analysis

## Model Performance Comparison

### Sentiment Analysis Libraries with Methodology

| Category                     | Model                     | Accuracy | Methodology                                   | Pros                                                                 | Cons                                                                 |
|------------------------------|---------------------------|----------|----------------------------------------------|----------------------------------------------------------------------|----------------------------------------------------------------------|
| **Lexicon/Rule-Based Models** | VADER                     | 0.609    | Sentiment lexicon + grammatical rules         | Fast; no training needed                                             | Fails with sarcasm/context                                           |
|                              | TextBlob                  | 0.116    | Simplified lexicon + phrase extraction        | Easy to implement                                                    | Very low accuracy                                                    |
| **Traditional Machine Learning Models** | Linear SVC                | 0.899    | TF-IDF + Support Vector Classifier            | High accuracy; scalable                                               | Less interpretable                                                   |
|                              | Logistic Regression       | 0.869    | TF-IDF + Logistic Regression                  | Fast; probabilistic outputs                                           | Struggles with complex patterns                                      |
|                              | Decision Tree            | 0.834    | TF-IDF + Single decision tree                 | Interpretable; handles non-linear data                                | Prone to overfitting                                                 |
|                              | Random Forest            | 0.874    | TF-IDF + Ensemble of decision trees           | Robust to overfitting                                                 | Computationally heavy                                                |
|                              | Naive Bayes (Multinomial) | 0.869    | TF-IDF + Bayesian probability                 | Lightweight; works with small data                                    | Assumes feature independence                                         |
|                              | Naive Bayes (Binomial)   | 0.864    | Binary word presence + Bayesian probability   | Simple; fast training                                                | Loses word frequency information                                     |
|                              | K-Nearest Neighbours      | 0.879    | TF-IDF + Distance-based classification        | No training required                                                 | Poor scalability; sensitive to irrelevant features                   |
| **Ensemble/Boosting Models** | Gradient Boosted Decision Tree | 0.879    | Iterative decision trees with gradient boosting | Handles non-linear patterns well                                     | Hyperparameter-sensitive                                             |
|                              | XGBoost                  | 0.864    | Optimized gradient boosting                   | Robust to outliers; handles missing data                              | Requires careful tuning                                              |
| **Neural Network**           | LSTM                     | ~0.700   | Recurrent Neural Network (sequential data)    | Captures word order                                                  | Slow training; needs large data                                      |
| **Transformer-Based Models** | DistilBERT               | 0.826    | Distilled BERT (bidirectional attention)      | Balance of speed/accuracy                                            | Requires GPU for efficiency                                          |
|                              | Flair                    | 0.777    | Contextual string embeddings                  | Handles rare words/misspellings                                      | Very slow                                                           |
|                              | FinBERT                  | 0.736    | BERT fine-tuned on financial texts            | Domain-specific expertise                                            | Limited to finance                                                   |
|                              | RoBERTa                  | 0.541    | Optimized BERT (larger corpus)                | Strong contextual understanding                                      | Slow; resource-heavy                                                |
|                              | BERTweet                 | 0.681    | BERT for tweets                              | Good for social media slang/emojis                                   | Narrow use case; slow                                               |

### Sentiment Analysis Libraries with Trading Strategies

| Category                     | Model                     | Accuracy | Strategy 1                                                                 | Strategy 2                                                                 |
|------------------------------|---------------------------|----------|---------------------------------------------------------------------------|---------------------------------------------------------------------------|
| **Lexicon/Rule-Based Models** | VADER                     | 0.609    | Short if (Title Neg, Desc Neg)<br>Final Portfolio Value: 8863.03<br>Total Return: -0.12069647372346716 | Long if (Title Pos, Desc Pos)<br>Final Portfolio Value: 8932.02<br>Total Return: -0.11294248349252435 |
|                              | TextBlob                  | 0.116    | Omitted due to low accuracy                                                |                                                                           |
| **Traditional Machine Learning Models** | Linear SVC                | 0.899    | Short if (Title Neg, Desc Neg)<br>Final Portfolio Value: -19160.40<br>Total Return: -inf | Long if (Title Pos, Desc Pos)<br>Final Portfolio Value: 9949.15 (46 trades)<br>Total Return: -0.005097633383426932 |
|                              | Logistic Regression       | 0.869    | Long if (Title Neg, Desc Neg) and (Desc <= 0.67)<br>Final Portfolio Value: 8152.98<br>Total Return: -0.20420101192486706 | Long if (Title Neg, Desc Neg) and (Desc > 0.8)<br>Final Portfolio Value: 10172.65 (84 trades only)<br>Total Return: 0.017117761561772273 |
|                              | Decision Tree            | 0.834    | Omitted due to similarity to other DT models                               |                                                                           |
|                              | Random Forest            | 0.874    | Long if (Title Pos) or (Desc Pos)<br>Final Portfolio Value: 9766.29 (46 trades only)<br>Total Return: -0.02364856348686172 | Long if (Title Neg, Desc Neg) and (Desc > 0.85)<br>Final Portfolio Value: 9568.33<br>Total Return: -0.04412664679033223 |
|                              | Naive Bayes (Multinomial) | 0.869    | Long if (Title Pos) or (Desc Pos)<br>Final Portfolio Value: 7633.11<br>Total Return: -0.2700901090957472 | Long if (Title Neg, Desc Neg) and (Desc >= 0.55)<br>Final Portfolio Value: 9574.91<br>Total Return: -0.04343903839003197 |
|                              | Naive Bayes (Binomial)   | 0.864    | Omitted due to similarity to other NB models                               |                                                                           |
|                              | K-Nearest Neighbours      | 0.879    | Long if (Title Pos) or (Desc Pos)<br>Final Portfolio Value: 9333.05<br>Total Return: -0.06902329873556787 | Long if (Title Neg, Desc Neg) and (Desc >= 0.85)<br>Final Portfolio Value: 9030.77<br>Total Return: -0.10194720042961834 |
| **Ensemble/Boosting Models** | Gradient Boosted Decision Tree | 0.879    | Short if (Title Neg, Desc Neg) and (Desc > 0.8)<br>Final Portfolio Value: 9670.85<br>Total Return: -0.03346852992589161 | Long if (Title Neg, Desc Neg) and (Desc > 0.72, Title > 0.72)<br>Final Portfolio Value: 8370.35<br>Total Return: -0.177889945903112 |
|                              | XGBoost                  | 0.864    | Omitted due to similarity to GBDT                                          |                                                                           |
| **Neural Network**           | LSTM                     | ~0.700   | Work In Progress                                                          |                                                                           |
| **Transformer-Based Models** | DistilBERT               | 0.826    | Long if (Title Pos) or (Desc Pos)<br>Final Portfolio Value: 7681.82<br>Total Return: -0.26372818142949583 | Long if (Title Neg, Desc Neg) and (Desc >= 0.99)<br>Final Portfolio Value: 9400.39<br>Total Return: -0.0618334631082467 |
|                              | Flair                    | 0.777    | (Title Neg, Desc Neg) and (Desc > 0.99), else Long<br>Final Portfolio Value: 7032.76<br>Total Return: -0.3520052994334656 | Long if (Title Neg, Desc Neg) and (Desc > 0.99)<br>Final Portfolio Value: 8298.53<br>Total Return: -0.18650725963856005 |
|                              | FinBERT                  | 0.736    | Omitted due to excessive neutral sentiments                                |                                                                           |
|                              | RoBERTa                  | 0.541    | Omitted due to similarity to other BERT models                             |                                                                           |
|                              | BERTweet                 | 0.681    | Omitted due to mismatch in usage                                           |                                                                           |
<!---
## Backtesting Performance ()

### Strategy Returns (2025 Data)
| Version | Annual Return | Max Drawdown | Sharpe Ratio | Win Rate |
|---------|---------------|--------------|--------------|----------|
| V1      | 12.4%         | -8.2%        | 1.2          | 58.3%    |
| V2      | 15.1%         | -6.7%        | 1.5          | 61.0%    |
| V3      | 18.6%         | -5.9%        | 1.8          | 63.5%    |

![Equity Curve](data/Results%20Charts%20Models/Backtesting%20Versions/V3/plot.png)

--->

## Topic Modeling Insights

### 3 Identified Major Clusters

1. **Financial Markets and Investments** (Group 1: Stocks, Oil, Dollar, Shares, Bonds, Future, Forecast). This group includes terms related to stock markets (stocks, shares), commodities (oil), currencies (dollar), fixed-income securities (bonds), derivatives (future), and market predictions (forecast), all of which are core components of financial markets and investment activities.

2. **Industry Sectors** (Group 2: Sales, Tech, Pharma, Steel).
This group represents key industrial sectors, including sales (general business activity), technology (tech), pharmaceuticals (pharma), and steel (manufacturing/heavy industry), which are major segments of the economy.

3. **Monetary and Energy Economics** (Group 3: Fx, Rates, Commodities, Solar, Wind, Gas, Bank, Credit, Debt, Fed, Fund).
This group encompasses terms related to foreign exchange (fx), interest rates (rates), commodities (including energy sources like gas, solar, wind), banking (bank, credit, debt), monetary policy (fed), and investment vehicles (fund), which collectively relate to monetary systems and energy-related economic activities.

[View interactive LDA visualization](data/Results%20Charts%20Models/LDA%20html%20charts/LDA_bloomberg.html)

# 📂 Dataset Documentation

## News Data Sources

### (Private dataset)
- **Format**: 150+ .docx files
- **Time Period**: Feb 2025 - Jun 2025
- **Coverage**: 
  - Daily market summaries
  - Sector-specific reports
  - Earnings call analyses

### Bloomberg News
- **Files**:
  - `BloombergNews.csv` (large dataset of bloomberg news)
  - `BloombergNews1.csv` (1/4 subset of bloomberg news)
  - `BloombergNews100.csv` (100 entry subset)

### Grok Headlines
- Pre-labeled sentiment dataset using Grok
- Contains ~2,000 financial headlines
- Three-class sentiment labels (Positive/Neutral/Negative)

## Market Data

### S&P 500 Futures
- **Time Granularity**: 1-minute intervals
- **Fields**:
  - Timestamp (UTC)
  - Open/High/Low/Close
  - Sentiment scores (derived)
- **File**: `SnP futures intraday w Sentiments.csv`

### Historical S&P 500
- **Files**:
  - `SNP500 Historical data.csv` (2000-2025 daily)
  - `SNP 08-11.csv` (2008 crisis period)


## 📜 License & Contact

## License
MIT License

Copyright (c) 2025 [Jay Tai]

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.