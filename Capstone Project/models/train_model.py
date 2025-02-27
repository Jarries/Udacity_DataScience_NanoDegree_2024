import pickle
import sys
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split, GridSearchCV
import time

random_state = 42


def load_and_split_data(data_filepath):
    """
    Load data from a CSV file and split it into features and target variable.

    This function reads data from the specified CSV file, splits the DataFrame into features (X)
    and the target variable (y), and returns them.

    Args:
        data_filepath (str): The file path to the CSV file containing the data.

    Returns:
        tuple: A tuple containing:
            - X (pd.DataFrame): The DataFrame of features.
            - y (pd.Series): The Series of the target variable.
    """
    df = pd.read_csv(data_filepath)
    X = df.drop(['successful_offer'], axis=1)
    y = df['successful_offer'].astype(int)
    return X, y

def build_model(X_train, y_train,random_state=random_state, cross_validation=3):
    """
    Builds and returns the best RandomForestClassifier model using GridSearchCV.

    Parameters:
    random_state (int): Seed used by the random number generator for reproducibility.
    cross_validation (int): Number of folds for cross-validation.

    Returns:
    RandomForestClassifier: The best RandomForestClassifier model found by GridSearchCV.
    
    The function performs the following steps:
    1. Initializes a RandomForestClassifier with the specified random_state and default n_estimators.
    2. Defines a parameter grid for hyperparameter tuning.
    3. Uses GridSearchCV to find the best hyperparameters based on cross-validation.
    4. Returns the best estimator found by GridSearchCV.
    """
    randomforrest = RandomForestClassifier(random_state=random_state, n_estimators=100)
    
    parameters = {
        'n_estimators': [100, 300],  # Number of trees in the forest
        'criterion': ['gini'],  # Function to measure the quality of a split
        'max_depth': [10, 20],  # Maximum depth of the tree
        'min_samples_split': [2, 5],  # Minimum number of samples required to split an internal node
        'min_samples_leaf': [1, 2],  # Minimum number of samples required to be at a leaf node
        'min_weight_fraction_leaf': [0.0],  # Minimum weighted fraction of the sum total of weights required to be at a leaf node
        'max_features': ['log2', 'sqrt'],  # Number of features to consider when looking for the best split
        'max_leaf_nodes': [None],  # Unlimited number of leaf nodes
        'min_impurity_decrease': [0.0],  # Minimum decrease in impurity required for a split
        'bootstrap': [True],  # Whether bootstrap samples are used when building trees
        'oob_score': [False],  # Whether to use out-of-bag samples to estimate the generalization accuracy
        'n_jobs': [None],  # Number of jobs to run in parallel
        'random_state': [random_state],  # Random state for reproducibility
        'verbose': [0],  # Controls the verbosity when fitting and predicting
        'warm_start': [False],  # Reuse the solution of the previous call to fit and add more estimators to the ensemble
        'class_weight': [None],  # Weights associated with classes
        'ccp_alpha': [0.0],  # Complexity parameter used for Minimal Cost-Complexity Pruning
        'max_samples': [None]  # Number of samples to draw from X to train each base estimator
    }
    grid_search_model = GridSearchCV(randomforrest, parameters, cv=cross_validation)
    grid_search_model.fit(X_train, y_train)

    best_model = grid_search_model.best_estimator_
    return best_model

def train_and_test_model(model, X_train, X_test, y_train, y_test, print_time=True):
    """
    Train and test a machine learning model.

    Parameters:
    model: The machine learning model to be trained and tested.
    X_train (DataFrame or array-like): The training feature data.
    X_test (DataFrame or array-like): The testing feature data.
    y_train (Series or array-like): The training target data.
    y_test (Series or array-like): The testing target data.
    print_time (bool, optional): Whether to print the training time. Defaults to True.

    Returns:
    model: The trained machine learning model.

    This function trains the provided model on the training data and evaluates it on both the training and testing data.
    It displays the accuracy, recall, and precision metrics for both the training and testing datasets in a table format.
    """

    # Record the start time
    start = time.time()
    
    # Train the model
    model = model.fit(X_train, y_train)
    
    # Record the end time
    end = time.time()
    
    # Optionally print the training time
    if print_time:
        print(f'The model took {end - start:.2f} seconds to train.')
    
    # Make predictions on the training and testing data
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    # Calculate accuracy, recall, and precision metrics
    metrics = {
        'Metric': ['Accuracy', 'Recall', 'Precision'],
        'Train': [
            accuracy_score(y_train, train_predictions),
            recall_score(y_train, train_predictions),
            precision_score(y_train, train_predictions)
        ],
        'Test': [
            accuracy_score(y_test, test_predictions),
            recall_score(y_test, test_predictions),
            precision_score(y_test, test_predictions)
        ]
    }
    
    # Create a DataFrame to display the metrics
    metrics_df = pd.DataFrame(metrics)
    
    # Display the metrics in a table format
    print(metrics_df)
    
    return model

def save_model(model, model_filepath):
    """
    Save the model as a pickle file.

    This function saves the given model to the specified file path using the pickle module.
    The model is serialized and stored in binary format.

    Args:
        model (object): The model to be saved.
        model_filepath (str): The file path where the model will be saved.

    Returns:
        None
    """
    with open(model_filepath, 'wb') as pickle_model:
        pickle.dump(model, pickle_model)




def main():
    """
    Main function to load data, build, train, and save a machine learning model.

    This function performs the following steps:
    1. Loads the data from the specified filepath.
    2. Applies oversampling to balance the dataset.
    3. Splits the data into training and testing sets.
    4. Builds the model using GridSearchCV for hyperparameter tuning.
    5. Trains and evaluates the model.
    6. Saves the trained model to the specified filepath.

    Parameters:
    None

    Command Line Arguments:
    data_filepath (str): The filepath of the data to train the model with.
    model_filepath (str): The filepath where the trained model will be saved.

    Usage:
    python train_model.py <data_filepath> <model_filepath>

    Example:
    python train_model.py ../data/cleaned_data.csv random_forest_classifier.pkl

    Notes:
    - Ensure that the data file exists at the specified filepath.
    - The function expects exactly two command line arguments: data_filepath and model_filepath.
    - If the correct number of arguments is not provided, an error message is displayed with usage instructions.
    """
    if len(sys.argv) == 3:
        data_filepath, model_filepath = sys.argv[1:]
        print("Loading the data...")
        X, y = load_and_split_data(data_filepath)
        
        sampler = RandomOverSampler()
        X_oversampled, y_oversampled = sampler.fit_resample(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X_oversampled, y_oversampled, test_size=0.20, random_state=random_state)
        
        print("Build the model... This might take some time... Take a break, go for a walk or call a friend.. Or you can just stare at the screen if you want to! ;)")
        model = build_model(random_state=random_state, cross_validation=3)
        
        print("Training the model...")
        train_and_test_model(model, X_train, X_test, y_train, y_test, print_time=False)
        
        print(f"Saving the model... Model: {model_filepath}")
        save_model(model, model_filepath)
        
        print("The model has been saved successfully...")
        
    else:
        print('Please provide the filepath of the data to train the model with '
              'as the first argument and the filepath of where you want to save '
              'the model to as the second argument. \n\n'
              'Example: python train_model.py ../data/cleaned_data.csv random_forest_classifier.pkl')
        
if __name__ == '__main__':
    main()