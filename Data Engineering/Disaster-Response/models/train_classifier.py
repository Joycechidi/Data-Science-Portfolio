import sys

import pandas as pd
from sqlalchemy import create_engine

import pickle

import re
import numpy as np
import pandas as pd

import nltk
# nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn import metrics

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    
    """
    Reading the SQLite data file function
    
    Arguments:
        database_filepath: SQLite database file path 
    Return:
       1) X - dataframe for the feature X
       2) y - dataframe for the label y
       3) category_names  - names of the different categories
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('Response', engine)
    
    X = df['message']
    #print(df)
    y = df.iloc[:,4:]
    category_names = y.columns

    #display(y.head())
    return X, y, category_names
    #pass


def tokenize(text):
    """
    Tokenization and Lemmatization function
        Argument:
               text: English texts from the user input messages                 
        Return:
               clean_tokens: Cleaned tokenized and lemmatized texts         
    """
    
    url_regex ='http[s]?://(?:[a-zA-Z]|[$-_@.&+]|[!*\(\),](?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens    
    
    
    #pass
    
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    
    """
    Class for extracting starting verbs used in building the Machine Learning pipelines
    """
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)    


def build_model():
    """
    Model Building Function
    Argument:
          None
     Return: 
           cv - Result of the GridSearchCV model
    """
    
    pipeline = Pipeline([
        ('text', TfidfVectorizer(tokenizer=tokenize)), 
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42, verbose=10))),
    ])

 
   #parameters for the GridSearchCV 
    make_scorer(f1_score, average='micro')

    parameters = { 'text__max_df': (0.75, 1.0),
             #  'vect__stop_words': ('english', None),
                'clf__estimator__n_estimators': [10, 20],
                'clf__estimator__min_samples_split': [2, 5]
              }
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=1)
    #print(pipeline.get_params())
          
          
    return cv 
    #pass


def evaluate_model(model, X_test, y_test, category_names):
    
    """
    Model validation/evaluation function
    Argument:
             model - model created in the build_model function
             X_test - percentage of the dataset to be used for testing the model
             Y_test - category_names
    
    """
    
    improved_y_pred = model.predict(X_test)

    pred_dict = {}

    for pred, label, col in zip(improved_y_pred.transpose(), y_test.values.transpose(), y_test.columns):
        print(col)
        print(classification_report(label, pred))
        pred_dict[col] = classification_report(label, pred)

# This part will be passed to avoiding outputing a lot of information for this project.
# The output will rather be logged into a text file.
    
    pass


def save_model(model, model_filepath):
    """
    Save the model for reuse
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
    

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