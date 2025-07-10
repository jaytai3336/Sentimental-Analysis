import pandas as pd
import re
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import re
import torch
from tqdm import tqdm
from transformers import pipeline, set_seed, AutoTokenizer, AutoModel
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import KMeans
import spacy
import numpy as np

df = pd.read_csv('./data/News Articles/raw/trump_social_results2.csv', index_col='datetime')[0:310]
df.head()

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger_eng")
nltk.download("maxent_ne_chunker_tab")
nltk.download("words")
nltk.download("wordnet")
nltk.download("stopwords")

stop_words = set(stopwords.words('english')).union({
    "said", "mr", "u", "s", "today", "report", "according"
})
lemmatizer = WordNetLemmatizer()

financial_phrases = [
    "beats expectations", "misses expectations", "strong guidance",
    "weak guidance", "downgraded rating", "upgraded rating",
    "raises outlook", "cuts outlook", "missed earnings", 
    "beat earnings", "profit warning", "record profits",
    "trading halt", "surprise loss", "share buyback", "stock split",
    "positive forecast", "negative forecast", "unexpected loss"
]

def extract_named_entities(text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    named_entities = []
    for subtree in chunked:
        if isinstance(subtree, Tree):
            entity = " ".join(token for token, pos in subtree.leaves())
            named_entities.append(entity.lower())
    return named_entities

def extract_financial_phrases(text):
    phrases_found = []
    for phrase in financial_phrases:
        # Allow variable whitespace, case-insensitive
        pattern = r"\b" + r"\s+".join(re.escape(word) for word in phrase.split()) + r"\b"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            phrases_found.append(match.group(0).lower())
    return phrases_found

def preprocess(text):
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.strip()
    text_lower = text.lower()

    named_entities = extract_named_entities(text)
    fin_phrases = extract_financial_phrases(text)

    all_preserved = set(named_entities + fin_phrases)

    tokens = word_tokenize(text_lower)
    other_tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token.isalpha() and token not in stop_words and token not in " ".join(all_preserved)
    ]

    return " ".join(fin_phrases + named_entities + other_tokens)


df['processed_text'] = df['text'].apply(preprocess)
df['processed_text'].head()

GROUPS = {
    "Financial Markets": [
        "stocks", "equities", "shares", "indices", "dow", "nasdaq", "s&p",
        "bonds", "treasuries", "yield", "spread", "options", "futures",
        "margin", "leverage", "short", "dividends", "etf", "volatility", 
        "vix", "hedge", "trading", "liquidity", "volume", "order book", 
        "forecast", "price action", "open interest", "technicals"
    ],
    "Corporate Sector": [
        "earnings", "revenue", "guidance", "layoffs", "acquisition", "merger", 
        "ipo", "startup", "valuation", "unicorn", "restructuring", "subsidiary",
        "brand", "expansion", "subsidiary", "joint venture", "conglomerate",
        "company", "profits", "CEO", "management", "shareholder", "stake"
    ],
    "Macro Finance": [
        "inflation", "interest", "rate hike", "fed", "ecb", "central bank", 
        "monetary", "liquidity", "policy", "credit", "debt", "balance sheet", 
        "yields", "bond buying", "qe", "qt", "macro", "gdp", "growth", "cpi", 
        "ppi", "unemployment", "deficit", "surplus", "sovereign", "treasury"
    ],
    "Energy & Commodities": [
        "oil", "brent", "wti", "crude", "natural gas", "coal", "uranium", 
        "gold", "silver", "copper", "commodity", "minerals", "barrel", "supply",
        "demand", "refinery", "pipeline", "opec", "inventory", "mining", 
        "extraction", "energy", "power", "grid", "electricity"
    ],
    "Geopolitics & Policy": [
        "war", "conflict", "military", "sanctions", "elections", "diplomacy", 
        "tariffs", "regime", "treaty", "un", "nato", "geopolitics", "alliance", 
        "border", "summit", "foreign policy", "cyberwar", "espionage", "blockade"
    ],
    "Tech & Innovation": [
        "ai", "machine learning", "deep learning", "chatbot", "quantum", 
        "robotics", "semiconductors", "chips", "hardware", "software", "cloud", 
        "infrastructure", "cybersecurity", "5g", "6g", "platform", "saas", 
        "startup", "innovation", "automation", "big data", "iot", "virtual reality"
    ],
    "Crypto & Digital Assets": [
        "crypto", "bitcoin", "ethereum", "nft", "token", "defi", "stablecoin", 
        "blockchain", "mining", "wallet", "exchange", "smart contract", 
        "airdrops", "gas fees", "halving", "web3", "metaverse", "yield farming"
    ],
    "Climate & Environment": [
        "climate", "global warming", "carbon", "emissions", "green", 
        "net zero", "biodiversity", "deforestation", "renewables", "solar", 
        "wind", "sustainability", "eco", "climate risk", "co2", "enviro", 
        "recycling", "pollution", "weather", "drought", "wildfire", "hurricane"
    ],
    "Health & Biotech": [
        "healthcare", "pharma", "vaccine", "covid", "pandemic", "epidemic", 
        "hospital", "insurance", "fda", "biotech", "drug", "therapy", 
        "clinical trials", "approval", "genomics", "mrna", "public health"
    ],
    "Consumer & Retail": [
        "retail", "e-commerce", "spending", "shopping", "consumer", "foot traffic", 
        "brand", "loyalty", "promotion", "fashion", "luxury", "discount", 
        "travel", "leisure", "supermarket", "apparel", "inventory", "mall", 
        "tourism", "holiday sales", "storefront", "lifestyle"
    ],
    "Society & Labor": [
        "labor", "employment", "wages", "strike", "union", "pension", 
        "migration", "education", "healthcare", "inequality", "crime", 
        "working class", "jobless", "benefits", "minimum wage", "social unrest"
    ],
    "Legal & Regulation": [
        "regulation", "compliance", "litigation", "lawsuit", "ban", 
        "fine", "sec", "doj", "fca", "privacy", "antitrust", "audit", 
        "whistleblower", "ethics", "governance", "oversight"
    ],
    "Real Estate & Housing": [
        "housing", "mortgage", "real estate", "rent", "home sales", 
        "construction", "property", "housing market", "zoning", "commercial real estate", 
        "land", "housing prices", "tenant", "eviction", "housing bubble"
    ],
    "Transport & Logistics": [
        "shipping", "freight", "logistics", "supply chain", "port", 
        "airline", "aviation", "rail", "infrastructure", "trucking", 
        "transportation", "container", "cargo", "disruption", "delivery"
    ],
    "Education & Knowledge": [
        "school", "university", "student", "curriculum", "exam", "scholarship", 
        "tuition", "degree", "research", "academic", "professor", 
        "online learning", "edtech", "education policy"
    ]
}

# Extended Tweet Classification Pipeline

set_seed(42)

# Load NLP models
classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33", device=0 if torch.cuda.is_available() else -1)
sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0 if torch.cuda.is_available() else -1)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
embed_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to('cuda' if torch.cuda.is_available() else 'cpu')
nlp = spacy.load("en_core_web_sm")

# Helper to extract contextual embeddings
def get_embedding(text):
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(embed_model.device)
    with torch.no_grad():
        model_output = embed_model(**encoded_input)
    return model_output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

# Multi-label classifier setup
def multi_label_classification(text, candidate_labels, threshold=0.5):
    result = classifier(text, candidate_labels=candidate_labels, multi_label=True)
    labels = [label for label, score in zip(result['labels'], result['scores']) if score >= threshold]
    return labels, result

# Data processing
results = []
embeddings = []
try:
    for sentence in tqdm(df['processed_text'], desc="Analyzing tweets"):
        try:
            # Topic classification
            topics, full_result = multi_label_classification(sentence, list(GROUPS.keys()), threshold=0.5)

            # Named entity recognition
            doc = nlp(sentence)
            named_entities = list(set(ent.text for ent in doc.ents))

            # Sentiment
            sent_result = sentiment(sentence)[0]

            # Keyword matching
            matched_keywords = []
            for label in topics:
                matched_keywords.extend([kw for kw in GROUPS[label] if re.search(rf"\\b{re.escape(kw)}\\b", sentence.lower())])

            # Embedding
            emb = get_embedding(sentence)
            embeddings.append(emb)

            results.append({
                "text": sentence,
                "topics": topics if topics else ["Uncertain"],
                "top_3_topics": ", ".join(full_result['labels'][:3]),
                "topic_confidences": [round(score, 3) for score in full_result['scores'][:3]],
                "matched_keywords": ", ".join(matched_keywords[:3]) if matched_keywords else "None",
                "sentiment": sent_result['label'].upper(),
                "sentiment_score": round(sent_result['score'], 4),
                "named_entities": named_entities
            })

        except Exception as e:
            print(f"Error: {sentence[:50]}... -> {str(e)}")
            continue
finally:
    if results:
        # Convert to DataFrame
        results_df = pd.DataFrame(results)

        # Embedding clustering
        X = np.stack(embeddings)
        kmeans = KMeans(n_clusters=6, random_state=42).fit(X)
        results_df['cluster'] = kmeans.labels_

        # Binarize multi-label topics
        mlb = MultiLabelBinarizer()
        topic_binarized = pd.DataFrame(mlb.fit_transform(results_df['topics']), columns=mlb.classes_)
        results_df = pd.concat([results_df, topic_binarized], axis=1)
        results_df_merged = df.merge(results_df, left_on='processed_text', right_on='text', how='left')

        # Save results
        results_df_merged.to_csv("trump_analysed1.csv", index=True)
        print("✅ Data saved successfully.")
    else:
        print("⚠️ No results to save.")