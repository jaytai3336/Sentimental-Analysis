import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from gensim.models.phrases import Phrases, Phraser
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import scipy.sparse

np.random.seed(999)
tf.random.set_seed(999)

# Load the dataset
df = pd.read_csv('data/News Articles/Kraggle Datasets/Labeled-headlines (Kraggle).csv', encoding='ISO-8859-1', on_bad_lines='skip', header=None)
df = df.dropna()

# Preprocess text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

df['processed_content'] = df[1].apply(preprocess_text)

df = pd.DataFrame({
    'processed_content': df['processed_content'],
    'sentiment': df[0]
})

# Apply bigram/trigram preprocessing
bigram_model = Phrases(df['processed_content'], min_count=1, threshold=10)
trigram_model = Phrases(bigram_model[df['processed_content']], min_count=1, threshold=10)
bigram_phraser = Phraser(bigram_model)
trigram_phraser = Phraser(trigram_model)
df['processed_content_phrased'] = df['processed_content'].apply(
    lambda x: trigram_phraser[bigram_phraser[x]]
)

# Convert sentiment labels
label_map = {'neutral': 1, 'none': 1, 'positive': 2, 'negative': 0}
if df['sentiment'].dtype == 'object':
    df['sentiment'] = df['sentiment'].map(label_map)

# Split data into 100 chunks
X = df['processed_content_phrased']
y = df['sentiment']
num_chunks = 100
chunk_size = max(1, len(X) // num_chunks)
chunks = [(X[i:i + chunk_size], y[i:i + chunk_size]) for i in range(0, len(X), chunk_size)]
print(f"Created {len(chunks)} chunks of approximate size {chunk_size}")

# TF-IDF Vectorization (fit once on all data)
positive_terms = [
    'profit', 'rose', 'increased', 'increase', 'agreement', 'signed', 'contract',
    'growth', 'grew', 'awarded', 'positive', 'operating profit', 'net profit',
    'profit rose', 'period increased', 'sales increased', 'signed agreement',
    'increased respectively', 'loss narrowed', 'pretax profit', 'profit period',
    'sales rose', 'profit net', 'grew percent', 'awarded contract', 'sales increase',
    'profit increased', 'cost savings', 'expected increase', 'profit totalled',
    'operating profit rose', 'net profit rose', 'period increased revenue',
    'increased respectively compared', 'net sales increased', 'sales period increased',
    'operating profit net', 'profit net sales', 'today net profit', 'sales rose net',
    'operating profit increased', 'net sales rose', 'net sales increase',
    'said net profit', 'operating profit totalled', 'operating profit period',
    'grew percent million'
]

negative_terms = [
    'loss', 'decreased', 'fell', 'lower', 'cut', 'dropped', 'negative',
    'declined', 'warning', 'laid', 'decreased eur', 'operating loss',
    'net loss', 'profit fell', 'sales decreased', 'profit decreased',
    'profit warning', 'loss totalled', 'pct lower', 'period decreased',
    'pretax loss', 'today slipped', 'sales fell', 'slipped net',
    'personnel reductions', 'fell percent', 'loss totaled', 'decreased net',
    'laid off', 'operating profit fell', 'profit period decreased',
    'operating loss totalled', 'period decreased net', 'net sales decreased',
    'said today slipped', 'decreased net sales', 'net profit decreased',
    'slipped net loss', 'issued profit warning', 'net profit fell',
    'operating profit decreased', 'today slipped net'
]

vocabulary = positive_terms + negative_terms
vocabulary = [term.replace('_', ' ') for term in vocabulary]

# Fit TF-IDF on all data to ensure consistent feature dimensions
vectorizer = TfidfVectorizer(
    vocabulary=vocabulary,
    ngram_range=(1, 3),
    lowercase=False
)
X_all_tfidf = vectorizer.fit_transform([' '.join(x) for x in X])

# Define the neural network
model = Sequential([
    Dense(128, input_dim=len(vocabulary), activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

model.summary()

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Function to get misclassified samples
def get_misclassified_indices(X, y_true, model):
    try:
        y_pred = model.predict(X, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_true, axis=1)
        misclassified = np.where(y_pred_classes != y_true_classes)[0]
        return misclassified
    except Exception as e:
        print(f"Error in get_misclassified_indices: {e}")
        return []

# Simulate new data by reserving 20% of chunks
new_data_chunk_count = int(0.2 * len(chunks))
new_data_chunks = chunks[:new_data_chunk_count]
training_chunks = chunks[new_data_chunk_count:]
print(f"Reserved {new_data_chunk_count} chunks for new data simulation")
print(f"Using {len(training_chunks)} chunks for initial training")

# Initialize history for tracking
history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
misclassified_X = None
misclassified_y = None
new_data_idx = 0

# Train on chunks
for i, (X_chunk, y_chunk) in enumerate(training_chunks):
    print(f"\nTraining on chunk {i+1}/{len(training_chunks)}")
    
    # Perform train/validation/test split on this chunk
    if len(X_chunk) < 3:  # Skip chunks that are too small
        print(f"Skipping chunk {i+1} due to insufficient samples ({len(X_chunk)})")
        continue
    
    X_train, X_temp, y_train, y_temp = train_test_split(X_chunk, y_chunk, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)
    
    # Apply TF-IDF transformation using the pre-fitted vectorizer
    X_train_tfidf = vectorizer.transform([' '.join(x) for x in X_train])
    X_val_tfidf = vectorizer.transform([' '.join(x) for x in X_val])
    X_test_tfidf = vectorizer.transform([' '.join(x) for x in X_test])
    
    # One-hot encode labels
    y_train_onehot = to_categorical(y_train.values, num_classes=3)
    y_val_onehot = to_categorical(y_val.values, num_classes=3)
    y_test_onehot = to_categorical(y_test.values, num_classes=3)
    
    print(f"Chunk {i+1} shapes: X_train_tfidf={X_train_tfidf.shape}, y_train_onehot={y_train_onehot.shape}, "
          f"X_val_tfidf={X_val_tfidf.shape}, y_val_onehot={y_val_onehot.shape}")
    
    # Add misclassified samples from previous chunk
    if misclassified_X is not None and misclassified_X.shape[0] > 0:
        try:
            X_train_tfidf = scipy.sparse.vstack([X_train_tfidf, misclassified_X])
            y_train_onehot = np.vstack([y_train_onehot, misclassified_y])
            print(f"Added {misclassified_X.shape[0]} misclassified samples from previous chunk")
        except Exception as e:
            print(f"Error combining misclassified samples: {e}")
            misclassified_X = None
            misclassified_y = None
    
    # Add new data (one new chunk at a time)
    if new_data_idx < len(new_data_chunks):
        X_new, y_new = new_data_chunks[new_data_idx]
        X_new_tfidf = vectorizer.transform([' '.join(x) for x in X_new])
        y_new_onehot = to_categorical(y_new.values, num_classes=3)
        try:
            X_train_tfidf = scipy.sparse.vstack([X_train_tfidf, X_new_tfidf])
            y_train_onehot = np.vstack([y_train_onehot, y_new_onehot])
            print(f"Added new data chunk {new_data_idx + 1}/{len(new_data_chunks)} with {X_new_tfidf.shape[0]} samples")
            new_data_idx += 1
        except Exception as e:
            print(f"Error adding new data chunk: {e}")
    
    # Train on this chunk
    try:
        chunk_history = model.fit(
            X_train_tfidf.toarray(),
            y_train_onehot,
            validation_data=(X_val_tfidf.toarray(), y_val_onehot),
            epochs=5,
            batch_size=32,
            callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)],
            verbose=1
        )
        
        # Update history
        history['loss'].extend(chunk_history.history['loss'])
        history['accuracy'].extend(chunk_history.history['accuracy'])
        history['val_loss'].extend(chunk_history.history['val_loss'])
        history['val_accuracy'].extend(chunk_history.history['val_accuracy'])
    except Exception as e:
        print(f"Error during training chunk {i+1}: {e}")
        continue
    
    # Get misclassified samples from this chunk
    misclassified_indices = get_misclassified_indices(X_train_tfidf, y_train_onehot, model)
    if len(misclassified_indices) > 0:
        misclassified_X = X_train_tfidf[misclassified_indices]
        misclassified_y = y_train_onehot[misclassified_indices]
        print(f"Misclassified samples in chunk {i+1}: {len(misclassified_indices)}")
    else:
        misclassified_X = None
        misclassified_y = None
        print(f"No misclassified samples in chunk {i+1}")
    
    # Evaluate on test set for this chunk
    try:
        test_loss, test_accuracy = model.evaluate(X_test_tfidf.toarray(), y_test_onehot, verbose=0)
        print(f"Chunk {i+1} Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    except Exception as e:
        print(f"Error evaluating test set for chunk {i+1}: {e}")

# Final evaluation on the last chunk's test set
print(f"\nFinal Test Loss: {test_loss:.4f}, Final Test Accuracy: {test_accuracy:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Make predictions and print classification report for the last chunk's test set
try:
    y_pred = model.predict(X_test_tfidf.toarray())
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test_onehot, axis=1)
    print(classification_report(y_true_classes, y_pred_classes, target_names=['Negative', 'Neutral', 'Positive']))
except Exception as e:
    print(f"Error generating classification report: {e}")