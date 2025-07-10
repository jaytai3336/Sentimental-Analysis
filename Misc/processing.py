import pandas as pd
import re
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

df = pd.read_csv('./data/News Articles/raw/trump_social_results2.csv')
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

df.to_csv('./data/News Articles/processed/trump_processed.csv')
