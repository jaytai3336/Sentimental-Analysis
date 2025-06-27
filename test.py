import warnings

from keras import Sequential

warnings.filterwarnings('ignore')

import pandas as pd
from gensim.models.phrases import Phrases, Phraser
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.utils import to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import tensorflow as tf
np.random.seed(999)
tf.random.set_seed(999)

# Load DataFrame (replace with your actual data, e.g., BloombergNews100.csv)
# Load the dataset
df = pd.read_csv('data/News Articles/Kraggle Datasets/Labeled-headlines (Kraggle).csv', encoding='ISO-8859-1', on_bad_lines='skip', header=None)
df = df.dropna()

# Define a function to preprocess the text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    tokens = [word for word in tokens if word.isalpha()]  # Remove non-alphabetic characters
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return tokens

# Apply the function to preprocess the 'data' column
df['processed_content'] = df[1].apply(preprocess_text)

df = pd.DataFrame({
    'processed_content': df['processed_content'],
    'sentiment': df[0]
})

df.head()

# Apply bigram/trigram preprocessing
bigram_model = Phrases(df['processed_content'], min_count=1, threshold=10)
trigram_model = Phrases(bigram_model[df['processed_content']], min_count=1, threshold=10)
bigram_phraser = Phraser(bigram_model)
trigram_phraser = Phraser(trigram_model)
df['processed_content_phrased'] = df['processed_content'].apply(
    lambda x: trigram_phraser[bigram_phraser[x]]
)

# Convert sentiment labels to numerical (if not already)
label_map = {'neutral': 1, 'none': 1, 'positive': 2, 'negative': 0}
if df['sentiment'].dtype == 'object':
    df['sentiment'] = df['sentiment'].map(label_map)

# Split data
X = df['processed_content_phrased']
y = df['sentiment']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts([' '.join(x) for x in X_train])
vocab_size = len(tokenizer.word_index) + 1

# Sequence padding
max_len = 100  # Adjust based on your data
X_train_seq = pad_sequences(tokenizer.texts_to_sequences([' '.join(x) for x in X_train]), maxlen=max_len)
X_val_seq = pad_sequences(tokenizer.texts_to_sequences([' '.join(x) for x in X_val]), maxlen=max_len)
X_test_seq = pad_sequences(tokenizer.texts_to_sequences([' '.join(x) for x in X_test]), maxlen=max_len)

# Load GloVe embeddings (download from https://nlp.stanford.edu/projects/glove/)
embedding_dim = 100
embeddings_index = {}
with open('data/Results Charts Models/Dictionaries/glove.6B.100d.txt', encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Create embedding matrix
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

def build_optimized_lstm():
    model = Sequential([
        Embedding(vocab_size, embedding_dim, weights=[embedding_matrix],
                 input_length=max_len, trainable=True),  # Now trainable!
        SpatialDropout1D(0.2),
        Bidirectional(LSTM(96, return_sequences=False,
                         kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))),
        Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        Dropout(0.3),
        Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Callbacks
optimized_callbacks = [
    EarlyStopping(patience=5, monitor='val_accuracy', mode='max', restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)
]

# Build and train
optimized_lstm = build_optimized_lstm()
optimized_lstm.summary()

history_optimized = optimized_lstm.fit(
    X_train_seq, y_train_onehot,
    validation_data=(X_val_seq, y_val_onehot),
    epochs=30,
    batch_size=32,  # Reduced from 64
    callbacks=optimized_callbacks,
    verbose=1
)