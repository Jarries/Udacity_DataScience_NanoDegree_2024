import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, precision_recall_fscore_support, f1_score, recall_score, precision_score
import pickle

def load_data(database_filepath):
    """
    Load data from the SQLite database and split into features and target variables.

    Args:
    database_filepath (str): Filepath for the SQLite database.

    Returns:
    X (pd.Series): Series containing the messages.
    Y (pd.DataFrame): DataFrame containing the categories.
    categories (list): List of category names.
    """
    engine = create_engine("sqlite:///{}".format(database_filepath))
    df = pd.read_sql_table('disaster_messages', engine)
    X = df['message']
    Y = df[df.columns[4:]]
    categories = Y.columns.tolist()
    return X, Y, categories

def tokenize(text):
    """
    Tokenizes the input text by performing the following steps:
    1. Converts text to lowercase.
    2. Removes non-alphanumeric characters.
    3. Tokenizes the text into words.
    4. Removes English stopwords.
    5. Lemmatizes the tokens.

    Args:
        text (str): The input text to be tokenized.

    Returns:
        list: A list of processed tokens.
    """
    # Remove non-alphanumeric characters and convert to lowercase
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stopwords_to_remove = set(stopwords.words('english'))
    
    # Initialize the lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Lemmatize and remove stopwords
    final_tokens = [lemmatizer.lemmatize(token).strip() for token in tokens if token not in stopwords_to_remove]
    
    return final_tokens


def build_model():
    """
    Build a machine learning model pipeline with GridSearchCV for hyperparameter tuning.
    This function creates a pipeline that includes:
    - CountVectorizer: Converts text data into a matrix of token counts.
    - TfidfTransformer: Transforms the count matrix to a normalized tf-idf representation.
    - MultiOutputClassifier: A wrapper for multi-output classification using RandomForestClassifier.
    The pipeline is then wrapped in a GridSearchCV object to perform hyperparameter tuning.
    
    Returns:
        GridSearchCV: A GridSearchCV object with the specified pipeline and parameters.
    """
    # Create a pipeline with a CountVectorizer, TfidfTransformer, and MultiOutputClassifier
    randomforestpipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),  # Convert text to token counts
        ('tfidf', TfidfTransformer()),  # Transform counts to tf-idf representation
        ('classifier', MultiOutputClassifier(RandomForestClassifier(), n_jobs=1))  # Multi-output classification
    ])

    # Define hyperparameters for GridSearchCV
    parameters = {
        'classifier__estimator__n_estimators': [10, 30, 50, 100],  # Number of trees in the forest
        'classifier__estimator__random_state': [42]  # Random state for reproducibility
    }

    # Wrap the pipeline in a GridSearchCV object
    randomForestCv = GridSearchCV(randomforestpipeline, param_grid=parameters)

    return randomForestCv

def evaluate_model(model, X_test, y_test, category_names, verbose=1):
    """
    Evaluate the performance of a multi-output classification model.
    This function predicts the labels for the test set and calculates the precision, recall,
    and F1 score for each category. It also prints the classification report for each category
    if verbose is set to 1.
    Args:
        model: The trained model to be evaluated.
        X_test: The test features.
        y_test: The true labels for the test set.
        category_names: List of category names corresponding to the labels.
        verbose (int, optional): If set to 1, prints the classification report for each category. Default is 1.

    Returns:
        None
    """
    # Predict the labels for the test set
    y_pred = model.predict(X_test)
    
    # Initialize lists to store overall metrics
    overall_f1 = []
    overall_recall = []
    overall_precision = []
    
    # Iterate over each category to calculate and print metrics
    for idx, column in enumerate(category_names):
        if verbose == 1:
            print("Classification Report for column: " + column + "\n")
            print(classification_report(y_test.values[:, idx], y_pred[:, idx]))
        
        # Calculate and store precision, recall, and F1 score for the current category
        overall_precision.append(precision_score(y_test.values[:, idx], y_pred[:, idx], average='weighted'))
        overall_f1.append(f1_score(y_test.values[:, idx], y_pred[:, idx], average='weighted'))
        overall_recall.append(recall_score(y_test.values[:, idx], y_pred[:, idx], average='weighted'))
    
    # Print overall metrics if verbose is set to 1
    if verbose == 1:
        print('Overall precision: ', np.mean(overall_precision))
        print('Overall F1 score: ', np.mean(overall_f1))
        print('Overall recall: ', np.mean(overall_recall))

def save_model(model, model_filepath):
    """
    Save the best estimator of a trained model to a file.
    This function extracts the best estimator from a GridSearchCV object and saves it to the specified file path using pickle.
    Args:
        model: The trained GridSearchCV object containing the best estimator.
        model_filepath: The file path where the model should be saved.
    Returns:
        None
    """
    # Extract the best estimator from the GridSearchCV object
    best_model = model.best_estimator_
    
    # Save the best model to the specified file path using pickle
    with open(model_filepath, 'wb') as pickle_file:
        pickle.dump(best_model, pickle_file)


def main():
    """
    Main function to train and save a machine learning model for disaster response messages.

    This function performs the following steps:
    1. Loads data from a SQLite database.
    2. Splits the data into training and test sets.
    3. Builds a machine learning model pipeline.
    4. Trains the model on the training data.
    5. Evaluates the model on the test data.
    6. Saves the trained model to a pickle file.

    The function expects two command-line arguments:
    1. The filepath of the SQLite database containing the disaster messages.
    2. The filepath where the trained model should be saved.

    Example:
        python train_classifier.py ../data/DisasterResponse.db classifier.pkl
    """
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
        evaluate_model(model, X_test, Y_test, category_names, verbose = 1)

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