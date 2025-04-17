from flask import Flask, render_template, request
import joblib
import re, string, os
import pandas as pd

# — NEW IMPORTS —
import nltk
from nltk.tokenize import word_tokenize
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import DistilBertTokenizer, TFDistilBertModel

# Download models/data if needed (you can also run these once in a setup script)
nltk.download('punkt')
# python -m spacy download en_core_web_sm

app = Flask(__name__)

# load spaCy
nlp = spacy.load('en_core_web_sm')

# paths...
parent_dir      = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
model_main_path = os.path.join(parent_dir, "Models/model.pkl")
vectorizer_path = os.path.join(parent_dir, "Models/tfidf_vectorizer.pkl")
model_rf_path   = os.path.join(parent_dir, "Models/random_forest.pkl")
model_xgb_path  = os.path.join(parent_dir, "Models/xgboost_model.pkl")

# load your pickles
model_main = joblib.load(model_main_path)
vectorizer = joblib.load(vectorizer_path)
model_rf    = joblib.load(model_rf_path)
model_xgb   = joblib.load(model_xgb_path)

# load distilbert
tokenizer  = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = TFDistilBertModel .from_pretrained('distilbert-base-uncased')

def wordpre(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)

    # tokenize & lemmatize
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]
    doc    = nlp(' '.join(tokens))
    return ' '.join(tok.lemma_ for tok in doc)

@app.route('/', methods=['GET','POST'])
def index():
    result_rf  = result_xgb = None
    if request.method == 'POST':
        raw    = request.form['txt']
        clean  = wordpre(raw)
        tfidf  = vectorizer.transform([clean])
        rf_pred = model_rf .predict(tfidf)[0]
        xgb_pred= model_xgb.predict(tfidf)[0]

        # turn 0/1 into labels
        result_rf  = 'Fake' if rf_pred  == 1 else 'Real'
        result_xgb = 'Fake' if xgb_pred == 1 else 'Real'

        # (optional) BERT embeddings
        inputs   = tokenizer([clean], return_tensors='tf', padding=True,
                             truncation=True, max_length=128)
        outputs  = bert_model(inputs)
        embeddings = outputs.last_hidden_state[:,0,:].numpy()
        # …do something with embeddings if needed…

    return render_template("index.html",
                           result_rf=result_rf,
                           result_xgb=result_xgb,
                           original_text=request.form.get('txt',''))

if __name__ == '__main__':
    app.run(debug=True)
