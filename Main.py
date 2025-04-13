#!/usr/bin/env python3
import os
import re
import string
import joblib
import pandas as pd
import numpy as np

# --- Scikit-learn Utilities ---
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- Traditional & Boosting Libraries ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

<<<<<<< HEAD
# --- Deep Learning Libraries ---
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
=======
# Load the dataset
data = pd.read_csv('Data/Dataset/data1.csv')
>>>>>>> 118f055b82567e698f5892e1ca8eb0a0e393c5f1

# --- NLP Libraries ---
# NLTK for tokenization and stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# spaCy for lemmatization
import spacy
nlp = spacy.load('en_core_web_sm')

# gensim for Word2Vec experiments (optional)
import gensim
from gensim.models import Word2Vec

# transformers for BERT embeddings
from transformers import BertTokenizer, TFBertModel

# ---------------------------------------------
# Helper Functions for Preprocessing
# ---------------------------------------------
def preprocess_text(text):
    """
    Preprocess the input text by:
      - Converting to lowercase
      - Removing bracketed expressions, URLs, HTML tags, punctuation, and newline characters
      - Tokenizing using NLTK and removing stopwords and non-alphabetic tokens
      - Lemmatizing tokens using spaCy
    Returns the cleaned text.
    """
    # Basic cleaning
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', ' ', text)
    
    # Tokenize using NLTK
    tokens = word_tokenize(text)
    filtered_tokens = [
        token for token in tokens
        if token not in stopwords.words('english') and token.isalpha()
    ]
    
    # Lemmatize using spaCy
    doc = nlp(" ".join(filtered_tokens))
    lemmas = [token.lemma_ for token in doc]
    return " ".join(lemmas)

# ---------------------------------------------
# (Optional) Train a Word2Vec Model using gensim
# ---------------------------------------------
def train_word2vec(texts):
    """
    Trains a Word2Vec model on preprocessed texts.
    texts: Iterable of cleaned text strings.
    Returns: Trained gensim Word2Vec model.
    """
    tokenized_texts = [text.split() for text in texts]
    model_w2v = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)
    return model_w2v

def get_average_w2v(text, w2v_model):
    """
    Computes the average Word2Vec embedding for the given text.
    """
    tokens = text.split()
    vectors = [w2v_model.wv[token] for token in tokens if token in w2v_model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(w2v_model.vector_size)

# ---------------------------------------------
# BERT-based Embedding Extraction for Deep Learning
# ---------------------------------------------
# Load pre-trained BERT tokenizer and model (using distilBERT for speed)
tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = TFBertModel.from_pretrained('distilbert-base-uncased')

def get_bert_embeddings(texts, max_length=128):
    """
    Extract BERT embeddings for texts.
    texts: pandas Series or list of strings.
    Returns: Numpy array of [CLS] token embeddings.
    """
    inputs = tokenizer(texts.tolist(), return_tensors='tf', padding=True, truncation=True, max_length=max_length)
    outputs = bert_model(inputs)
    # Use the [CLS] token's embedding (first token) for each text.
    cls_embeddings = outputs.last_hidden_state[:, 0, :]
    return cls_embeddings.numpy()

# ---------------------------------------------
# Load Dataset
# ---------------------------------------------
# Update the path to reflect the project structure
data_path = os.path.join("Data", "Dataset", "data.csv")
data = pd.read_csv(data_path)

# Assure that 'text' and 'label' columns exist
data['text_clean'] = data['text'].apply(preprocess_text)

# ---------------------------------------------
# Define Features and Labels
# ---------------------------------------------
X = data['text_clean']
y = data['label']

# Split the dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------
# Pipeline 1: TF-IDF + RandomForest Classifier
# ---------------------------------------------
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train_tfidf, y_train)

y_pred_rf = model_rf.predict(X_test_tfidf)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

<<<<<<< HEAD
# Save the RandomForest model and TF-IDF vectorizer
joblib.dump(model_rf, os.path.join("Models", "random_forest.pkl"))
joblib.dump(vectorizer, os.path.join("Models", "tfidf_vectorizer.pkl"))
=======
# Save the trained model and vectorizer
joblib.dump(model, 'Models/model.pkl')
>>>>>>> 118f055b82567e698f5892e1ca8eb0a0e393c5f1

# ---------------------------------------------
# Pipeline 2: TF-IDF + XGBoost Classifier
# ---------------------------------------------
model_xgb = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model_xgb.fit(X_train_tfidf, y_train)

y_pred_xgb = model_xgb.predict(X_test_tfidf)
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("XGBoost Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))
print("XGBoost Classification Report:\n", classification_report(y_test, y_pred_xgb))

# Save the XGBoost model
joblib.dump(model_xgb, os.path.join("Models", "xgboost_model.pkl"))

# ---------------------------------------------
# Pipeline 3: Deep Learning using BERT Embeddings
# ---------------------------------------------
# Extract BERT embeddings for the training and testing data
X_train_bert = get_bert_embeddings(X_train)
X_test_bert = get_bert_embeddings(X_test)

# Define a simple neural network for binary classification
input_dim = X_train_bert.shape[1]  # Typically 768 for distilBERT
model_deep = Sequential([
    Dense(128, activation='relu', input_shape=(input_dim,)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # Change to softmax if doing multi-class classification
])
model_deep.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the deep learning model
model_deep.fit(X_train_bert, y_train, epochs=3, batch_size=16, validation_split=0.1)

# Evaluate the deep learning model
loss, accuracy = model_deep.evaluate(X_test_bert, y_test)
print("Deep Learning Model Accuracy:", accuracy)

# Save the deep learning model in .h5 format
model_deep.save(os.path.join("Models", "deep_model.h5"))

# ---------------------------------------------
# (Optional) Word2Vec Experiment (not used in main pipelines)
# ---------------------------------------------
# Uncomment the following lines to train and test a Word2Vec model on your data.
# w2v_model = train_word2vec(data['text_clean'])
# sample_text = X_test.iloc[0]
# avg_embedding = get_average_w2v(sample_text, w2v_model)
# print("Average Word2Vec embedding for sample text:", avg_embedding)

if __name__ == '__main__':
    print("Training complete. All models have been saved in the 'Models' folder.")
