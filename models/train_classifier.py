import sys
import re
import nltk
import pandas as pd
import pickle
import sqlite3
from sqlalchemy import create_engine
import warnings
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize,pos_tag
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
warnings.filterwarnings('ignore')
nltk.download(['punkt', 'wordnet', 'stopwords', 'omw-1.4', 'averaged_perceptron_tagger']) 



def load_data(database_filepath):
    """
    Load the data from the SQLite database file and return X, Y and category names.
    
    Args:
    database_filepath: Path to the SQLite database file.
    
    Returns:
    X: features (messages)
    Y: target data (categories)
    category_names: names of the categories

    """
    
    # Connect to the SQLite database
    conn = sqlite3.connect(database_filepath)

    # Read the data from the 'diasterdata' table into a DataFrame
    df = pd.read_sql_query("SELECT * from disasterdata", con=conn)
    
    # Extract the 'message' column as the feature data
    X = df['message']
    
    # Extract the category columns as the target data
    Y = df.iloc[:, 4:]
    
    # Get the names of the category columns
    category_names = Y.columns

    # Close the database connection
    conn.close()

    return X, Y, category_names


def tokenize(text):
    """
    Tokenize the text by replacing URLs, removing non-alphanumeric characters, and converting to lowercase
    
    Args:
    text: The input text to be tokenized
    
    Returns:
    list: A list of clean tokens

    """
    
    # define URL regex pattern
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Replace URLs with a placeholder
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')

    # Remove all non-alphanumeric characters and convert to lowercase
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Initialize the lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Lemmatize and clean tokens
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = pos_tag(tokenize(sentence))
            if not pos_tags:  # Check if the list is empty
                continue
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False


    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def build_model():
    """
    Build a pipeline using a combination of text processing, feature extraction, and classification
    pipeline is optimized using GridSearchCV
    
    Returns:
    gridsearch: A GridSearchCV object with the optimized machine learning pipeline

    """
    
    # define the pipeline with text processing, feature extraction, and classification steps
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('starting_verb', StartingVerbExtractor())
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # define the hyperparameters to optimize the pipeline
    parameters = {
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__min_samples_split': [2, 4]
    }

    # sse GridSearchCV to find the optimal hyperparameters for the pipeline
    gridsearch = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, cv=5, verbose=2)

    return gridsearch


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the performance of model on the test dataset and print the classification report
    
    Args:
    model: Trained model for classification
    X_test: feature data for testing
    Y_test: target data for testing
    category_names: names of the categories
    """
    
    # Predict the categories using the trained model on the test dataset
    y_pred = model.predict(X_test)
    
    # Print the classification report to evaluate the model's performance
    print(classification_report(Y_test.values, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """
    Save the trained model as a pickle file
    
    Args:
    model: trained model to be saved
    model_filepath: path where the model will be saved
    """
    
    # Open the specified file in write-binary mode
    with open(model_filepath, 'wb') as f:
        # Dump the model into the file using pickle
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()