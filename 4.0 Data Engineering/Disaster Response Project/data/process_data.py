import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load and merge messages and categories datasets.

    Args:
    messages_filepath (str): Filepath for the messages CSV file.
    categories_filepath (str): Filepath for the categories CSV file.

    Returns:
    df: Merged DataFrame containing messages and categories data.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on = 'id')
    return df


def clean_data(df):
    """
    Clean and transform the merged DataFrame by splitting categories into separate columns,
    converting values to numeric, and removing duplicates.

    Args:
    df (pd.DataFrame): Merged DataFrame containing messages and categories data.

    Returns:
    df (pd.DataFrame): Cleaned DataFrame with categories split into separate columns and duplicates removed.
    """
    # Split the categories column into separate columns
    categories_split = pd.Series(df['categories']).str.split(pat=";", n=36, expand=True)
    
    # Extract column names for categories
    first_row = categories_split.iloc[0].tolist()
    category_colnames = list(map(lambda x: x[:-2], first_row))
    categories_split.columns = category_colnames

    # Replace 'related-2' with 'related-1' to ensure binary values
    categories_split['related'] = categories_split['related'].apply(lambda val: 'related-1' if val == 'related-2' else val)
    
    # Convert category values to numeric
    for column in categories_split:
        categories_split[column] = categories_split[column].astype(str).str.split('-').str[-1]
        categories_split[column] = pd.to_numeric(categories_split[column])

    # Drop the original categories column from the DataFrame
    df = df.drop("categories", axis=1)
    
    # Concatenate the original DataFrame with the new categories DataFrame
    df = pd.concat([df, categories_split], axis=1, sort=False)
    
    # Remove duplicate rows
    df = df.drop_duplicates()

    # Ensure no duplicates remain
    assert (df.duplicated().sum()) == 0

    return df

def save_data(df, database_filename):
    """
    Save the cleaned DataFrame to an SQLite database.

    Args:
    df (pd.DataFrame): Cleaned DataFrame containing messages and categories data.
    database_filename (str): Filename for the SQLite database.

    Returns:
    None
    """
    database_path = "sqlite:///{}".format(database_filename)
    engine = create_engine("sqlite:///{}".format(database_filename))
    df.to_sql('messages', engine, index=False)

def main():
    """
    Main function to load, clean, and save data.

    This function performs the following steps:
    1. Loads data from the specified file paths for messages and categories.
    2. Cleans the loaded data by splitting categories into separate columns and removing duplicates.
    3. Saves the cleaned data to an SQLite database.

    The function expects three command-line arguments:
    1. Filepath of the messages dataset (CSV file).
    2. Filepath of the categories dataset (CSV file).
    3. Filepath of the SQLite database to save the cleaned data.

    If the required arguments are not provided, it prints usage instructions.

    Example:
        python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()