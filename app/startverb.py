from sklearn.base import BaseEstimator,TransformerMixin
from nltk import sent_tokenize,pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def tokenize(text):
    """
    Tokenize the text by replacing URLs, removing non-alphanumeric characters, and converting to lowercase.
    
    Args:
    text: The input text to be tokenized.
    
    Returns:
    list: A list of clean tokens.
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
    """
    Custom transformer that extracts the starting verb of a sentence if there is one.
    """

    def starting_verb(self, text):
        """
        Determine if the first word of any sentence in the input text is a verb.
        
        Args:
        text: The input text to be analyzed.
        
        Returns:
        bool: True if any sentence in the input text starts with a verb, otherwise False.
        """
        
        # Tokenize the text into sentences
        sentence_list = sent_tokenize(text)
        
        # Iterate through each sentence
        for sentence in sentence_list:
            # Tokenize and POS-tag the sentence
            pos_tags = pos_tag(tokenize(sentence))
            
            # Check if the list is empty and continue with the next sentence if so
            if not pos_tags:
                continue
            
            # Get the first word and its POS-tag
            first_word, first_tag = pos_tags[0]
            
            # Check if the first word is a verb or 'RT' (retweet)
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        
        # If no sentence starts with a verb, return False
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Apply the starting_verb method to each element in the input data.
        
        Args:
        X: input data to be transformed.
        
        Returns:
        DataFrame: dataFrame containing the results of applying starting_verb to each element in X.
        """
        
        # Apply the starting_verb method to each element in X
        X_tagged = pd.Series(X).apply(self.starting_verb)
        
        # Convert the series to a DataFrame and return
        return pd.DataFrame(X_tagged)
