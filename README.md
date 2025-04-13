```markdown
# Fake News Detection Project

This project demonstrates a fake news detection pipeline using a variety of NLP libraries and machine learning frameworks. It is built using Python and combines multiple approaches ranging from classic machine learning pipelines to deep learning with BERT embeddings.

## Overview

The project features the following key elements:

- **Data Handling & Preprocessing:**  
  Uses `pandas` and `numpy` to handle datasets, along with `re` and Python’s built-in libraries for text cleaning.

- **Advanced NLP Processing:**  
  Integrates libraries such as:
  - **nltk:** For tokenization, stopword removal, and other text processing tasks.
  - **spaCy:** For efficient and modern lemmatization.
  - **gensim:** For training and experimenting with Word2Vec embeddings (optional).
  - **transformers:** For obtaining BERT (using a distilled model for speed) based embeddings.

- **Machine Learning Pipelines:**  
  Implements multiple pipelines for fake news classification:
  - **Traditional Pipeline:** TF-IDF vectorization with a RandomForest classifier.
  - **Boosting Approach:** TF-IDF with XGBoost.
  - **Deep Learning Approach:** BERT embeddings fed into a TensorFlow/Keras neural network.

- **Deployment:**  
  A Flask web application is provided to allow interactive prediction through a simple HTML interface.

## Project Structure

fake_news_detection/
├── App/
│   └── templates/
│       └── index.html       # Front-end HTML template for the Flask app
├── Data/
│   └── Dataset/
│       └── data.csv         # CSV dataset containing news texts and labels
├── Models/
│   ├── random_forest.pkl    # Saved RandomForest model
│   ├── tfidf_vectorizer.pkl # Saved TF-IDF vectorizer
│   ├── xgboost_model.pkl    # Saved XGBoost model
│   └── deep_model.h5        # Saved deep learning model (TensorFlow/Keras)
├── notebooks/
│   └── Fakenewsdetection.ipynb  # Jupyter Notebook for experiments and visualizations
├── Main.py                  # Main training script that builds and evaluates pipelines
├── app.py                   # Flask app to serve predictions using pre-trained models
├── Notes.mk                 # Project notes and informal documentation
├── README.md                # This file
├── requirements.txt         # Detailed list of required Python libraries and versions
└── requirements-install.txt # Alternative/simplified installation requirements

## File Descriptions

- **Main.py:**  
  This script handles the core functionality of the project:
  - Loads and preprocesses the dataset (`Data/Dataset/data.csv`).
  - Applies enhanced text processing using nltk and spaCy.
  - Optionally trains a Word2Vec model with gensim.
  - Builds three pipelines:
    1. **TF-IDF + RandomForest:** Classic machine learning.
    2. **TF-IDF + XGBoost:** Gradient boosting approach.
    3. **BERT Embeddings + Deep Learning:** Uses Hugging Face transformers to extract embeddings which feed into a TensorFlow neural network.
  - Evaluates each model and saves them under the **Models/** directory.

- **app.py:**  
  Implements a Flask web application to interactively predict fake news:
  - Loads the pre-trained model and vectorizer from **Models/**.
  - Provides routes to submit news text for prediction.
  - Preprocesses the input text and displays whether it is classified as fake or real through the HTML form (served via **index.html**).

- **index.html:**  
  The front-end HTML template located in **App/templates/**. It:
  - Contains a form for users to enter text.
  - Displays the entered text along with the prediction result after processing.

- **Fakenewsdetection.ipynb:**  
  A Jupyter Notebook that supports exploration, experimentation, and visualization:
  - Used for testing different preprocessing and modeling approaches.
  - Helps in fine-tuning the models and understanding the dataset.

- **Notes.mk:**  
  A plain text file for keeping track of project ideas, tasks, and reminders.

- **README.md:**  
  This file serves as a guide and overview of the project structure, setup instructions, and file functions.

- **requirements.txt:**  
  Lists all project dependencies with specific versions to ensure compatibility across environments.

- **requirements-install.txt:**  
  Provides an alternative or simplified list of installation packages for quick setup.

## Setup and Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/fake-news-detection.git
   cd fake-news-detection
   ```

2. **Create and Activate a Virtual Environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate   # For Unix/Linux
   venv\Scripts\activate      # For Windows
   ```

3. **Install Dependencies:**

   Use either the detailed list or the simplified requirements:
   
   ```bash
   pip install -r requirements.txt
   # or
   pip install -r requirements-install.txt
   ```

4. **Download spaCy Model:**

   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage

- **Training the Models:**

  Run the main training script to preprocess data, train models, and evaluate performance:

  ```bash
  python Main.py
  ```

- **Launching the Web App:**

  Start the Flask application to serve predictions:

  ```bash
  python app.py
  ```

  Open your browser and navigate to `http://127.0.0.1:5000/` to interact with the app.

## Additional Information

- The project supports multiple pipelines for experimentation. You can modify the Main.py script to tune hyperparameters or add new model pipelines.
- The Word2Vec functions in Main.py provide optional experiments if you wish to incorporate more complex embedding-based features.
- For further exploration and visualization, work with the Jupyter Notebook provided in the **notebooks/** folder.

---


=======
App still on development
>>>>>>> 118f055b82567e698f5892e1ca8eb0a0e393c5f1
