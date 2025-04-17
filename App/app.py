from flask import Flask, render_template, request
import joblib
import re
import string
import pandas as pd
import os
import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, TFBertModel

# Initialize Flask app
app = Flask(__name__)

# Load the spaCy model for lemmatization
nlp = spacy.load('en_core_web_sm')

# Load the necessary models and vectorizers
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
model_main = os.path.join(parent_dir, "Models/model.pkl")
model_path_rf = os.path.join(parent_dir, "Models/random_forest.pkl")
model_path_xgb = os.path.join(parent_dir, "Models/xgboost_model.pkl")
vectorizer_path = os.path.join(parent_dir, "Models/tfidf_vectorizer.pkl")

# Load the models
model_main = joblib.load(model_main)
model_rf = joblib.load(model_path_rf)
model_xgb = joblib.load(model_path_xgb)
vectorizer = joblib.load(vectorizer_path)

# Load BERT tokenizer and model for deep learning
tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = TFBertModel.from_pretrained('distilbert-base-uncased')

# Preprocessing function
def wordpre(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)  # remove special characters
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    
    # Tokenize and Lemmatize
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token.isalpha()]
    doc = nlp(" ".join(tokens))
    lemmatized = [token.lemma_ for token in doc]
    return " ".join(lemmatized)

# Prediction route
@app.route('/', methods=['POST'])
def pre():
    if request.method == 'POST':
        txt = request.form['txt']
        txt = wordpre(txt)  # Apply preprocessing
        
        # Transform the text using the TF-IDF vectorizer
        txt_tfidf = vectorizer.transform([txt])
        
        # Prediction using Random Forest model
        result_rf = model_rf.predict(txt_tfidf)[0]
        # Prediction using XGBoost model
        result_xgb = model_xgb.predict(txt_tfidf)[0]
        
        # For BERT, use the tokenizer to process the text
        inputs = tokenizer([txt], return_tensors='tf', padding=True, truncation=True, max_length=128)
        outputs = bert_model(inputs)
        bert_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token embeddings
        result_bert = bert_embeddings.numpy()

        # Assuming a binary classification
        if result_rf == 1:
            result_rf = "Fake"
        else:
            result_rf = "Real"
        
        if result_xgb == 1:
            result_xgb = "Fake"
        else:
            result_xgb = "Real"
        
        return render_template("index.html", result_rf=result_rf, result_xgb=result_xgb)

# Main route
@app.route('/')
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
