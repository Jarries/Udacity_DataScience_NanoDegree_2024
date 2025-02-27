import pandas as pd
import numpy as np
import sys

CHANNELS = ['email', 'mobile', 'social', 'web']

def load_data(portfolio_path, profile_path, transcript_path):
    """
    Load data from JSON files and return as DataFrames.

    This function reads three JSON files: 'portfolio.json', 'profile.json', and 'transcript.json'.
    Each file is read into a pandas DataFrame with the specified orientation and line format.

    Returns:
        tuple: A tuple containing three pandas DataFrames:
            - portfolio (DataFrame): Data from 'portfolio.json'.
            - profile (DataFrame): Data from 'profile.json'.
            - transcript (DataFrame): Data from 'transcript.json'.
    """
    portfolio = pd.read_json(portfolio_path, orient='records', lines=True)
    profile = pd.read_json(profile_path, orient='records', lines=True)
    transcript = pd.read_json(transcript_path, orient='records', lines=True)
    
    return portfolio, profile, transcript



def transform_channel_values(channels):
    """
    Transform a list of channel names into a list of binary values indicating the presence of each channel.

    This function takes a list of channel names and returns a list of binary values (0 or 1) for each
    channel in the predefined list of all possible channels ('email', 'mobile', 'social', 'web').
    A value of 1 indicates the presence of the channel in the input list, while 0 indicates its absence.

    Args:
        channels (list): A list of channel names (strings).

    Returns:
        list: A list of binary values indicating the presence (1) or absence (0) of each channel.
    """
    channels_booleans = []
    channel_all_options = CHANNELS
    for channel in channel_all_options:
        if channel in channels:
            channels_booleans.append(1)
        else:
            channels_booleans.append(0)
    return channels_booleans


def portfolio_engineering(dataframe):
    """
    Perform data preprocessing and feature engineering on the input DataFrame for portfolio analysis.

    This function processes the input DataFrame by renaming columns, transforming channel values into binary
    indicators, converting duration from days to hours, and creating dummy variables for the offer type.

    Args:
        dataframe (pd.DataFrame): The input DataFrame containing portfolio data.

    Returns:
        pd.DataFrame: The processed DataFrame with engineered features.
    """
    # Rename 'id' column to 'offer_id' for clarity
    dataframe.rename(columns={'id': 'offer_id'}, inplace=True)
    
    # Create a copy of the DataFrame to avoid modifying the original
    dataframe = dataframe.copy()
    
    # Define all possible channel options
    channel_all_options = ['email', 'mobile', 'social', 'web']
    
    # Transform 'channels' column into binary indicators for each channel
    dataframe[channel_all_options] = dataframe['channels'].apply(transform_channel_values).apply(pd.Series)
    
    # Drop the original 'channels' column
    dataframe = dataframe.drop('channels', axis=1)
    
    # Convert 'duration' from days to hours
    dataframe['duration_hours'] = dataframe['duration'].apply(lambda val: val * 24)
    
    # Drop the original 'duration' column
    dataframe = dataframe.drop('duration', axis=1)
    
    # Create dummy variables for 'offer_type'
    dataframe = pd.get_dummies(dataframe, columns=['offer_type'], dtype=int)
    
    return dataframe

def profile_engineering(dataframe):
    """
    Perform data preprocessing and feature engineering on the input DataFrame for profile analysis.

    This function processes the input DataFrame by converting membership dates to datetime format,
    extracting the year and weekday of membership, and creating dummy variables for gender and year.

    Args:
        dataframe (pd.DataFrame): The input DataFrame containing profile data.

    Returns:
        pd.DataFrame: The processed DataFrame with engineered features.
    """
    # Convert 'became_member_on' to datetime format
    dataframe['became_member_on'] = pd.to_datetime(dataframe['became_member_on'], format='%Y%m%d')
    
    # Extract the year of membership and convert to integer
    dataframe['year'] = dataframe['became_member_on'].dt.year.astype("int")
    
    # Extract the weekday of membership
    dataframe['weekday_membership'] = dataframe['became_member_on'].dt.weekday
    
    # Create dummy variables for 'gender' and 'year', and drop the original 'became_member_on' column
    dataframe = pd.get_dummies(dataframe, prefix=['gender', 'became_member_on'], columns=['gender', 'year'], dtype=int).drop('became_member_on', axis=1)
    
    return dataframe


def transcript_engineering(transcript, profile):
    """
    Perform data engineering on the transcript DataFrame.

    This function processes the input transcript DataFrame by filtering out records where the person
    is not present in the profile DataFrame. It also extracts the 'offer_id' from the 'value' column,
    handling cases where the key might be 'offer_id' or 'offer id'.

    Args:
        transcript (pd.DataFrame): The input DataFrame containing transcript data.
        profile (pd.DataFrame): The input DataFrame containing profile data.

    Returns:
        pd.DataFrame: The processed transcript DataFrame with the 'offer_id' extracted.
    """
    # Filter transcript to include only records where 'person' is in the profile 'id'
    transcript = transcript[transcript['person'].isin(profile['id'])]
    
    # Extract 'offer_id' from the 'value' column, handling different key names
    transcript['offer_id'] = transcript['value'].apply(lambda val: val.get('offer_id', val.get('offer id', np.nan)))
    
    return transcript


def get_complete_dataframe(portfolio_after_engineer, profile_after_engineered, transcript_after_engineered):
    """
    Get a complete DataFrame by merging the engineered portfolio, profile, and transcript DataFrames.

    This function merges the transcript DataFrame with the profile DataFrame on the 'person' and 'id' columns,
    respectively, and then merges the result with the portfolio DataFrame on the 'offer_id' column. The final
    DataFrame is sorted by 'id' and 'time' and the index is reset.

    Args:
        portfolio_after_engineer (pd.DataFrame): The engineered portfolio DataFrame.
        profile_after_engineered (pd.DataFrame): The engineered profile DataFrame.
        transcript_after_engineered (pd.DataFrame): The engineered transcript DataFrame.

    Returns:
        pd.DataFrame: The complete merged DataFrame.
    """
    # Merge transcript with profile on 'person' and 'id', and drop the 'person' column
    merged_transcript = transcript_after_engineered.merge(
            profile_after_engineered,
            left_on='person',
            right_on='id',
            how='left'
        ).drop('person', axis=1)

    # Merge the result with portfolio on 'offer_id', sort by 'id' and 'time', and reset the index
    df = merged_transcript.merge(
            portfolio_after_engineer,
            on='offer_id',
            how='left'
        ).sort_values(by=['id', 'time']).reset_index(drop=True)

    return df


def is_offer_successfull(customer_df, full_df):
    """
    Checks if offers are successful for a given customer.

    Parameters:
    customer_df (DataFrame): A DataFrame containing customer offer data with columns such as 'event', 'time', 'duration_hours', and 'offer_type_informational'.

    Returns:
    dict: A dictionary where keys are row indices and values indicate whether the offer was successful (1) or not (0) for each corresponding row.

    The function iterates through each row in the customer DataFrame to determine if an offer was successful. An offer is considered successful if:
    - The 'offer received' event is followed by an 'offer viewed' event.
    - For informational offers, a 'transaction' event occurs before the deadline.
    - For other offers, either an 'offer completed' or 'transaction' event occurs before the deadline.

    The function updates a dictionary (`customer_successful_map`) with the success status for each row and its subsequent rows involved in the offer process.
    """
    completed_with_success = False

    customer_successful_map = {}

    for idx, row in customer_df.iterrows():
        successful = 0
        if row.event == 'offer received':
            deadline = row.time + row.duration_hours
            next_row = full_df.loc[idx + 1]
            if next_row.event == 'offer viewed':
                next_next_row = full_df.loc[idx + 2]
                if next_next_row.time <= row.time + deadline:
                    if row.offer_type_informational == 1:
                        if next_next_row.event == 'transaction':
                            successful = 1
                    else:
                        if next_next_row.event == 'offer completed' or next_next_row.event == 'transaction':
                            successful = 1
                            completed_with_success = True

        if idx not in customer_successful_map:
            customer_successful_map[idx] = successful
            customer_successful_map[idx + 1] = successful
            customer_successful_map[idx + 2] = successful
            if completed_with_success:
                customer_successful_map[idx + 3] = successful
    return customer_successful_map


def fill_successful_offers(dataframe):
    """
    Fill the offer IDs and tag them as successful or not.

    This function processes the input DataFrame by iterating through each customer, determining if their offers
    were successful, and updating the DataFrame with this information. The success of each offer is determined
    using the `is_offer_successful` function, and the results are stored in the 'successful_offer' column.

    Args:
        dataframe (pd.DataFrame): The input DataFrame containing offer data.

    Returns:
        pd.DataFrame: The DataFrame with the 'successful_offer' column filled.
    """
    # Initialize a dictionary to map successful offers
    successful_map = {}
    
    # Create a new column 'successful_offer' and initialize with NaN
    dataframe['successful_offer'] = np.nan
    
    # Group the DataFrame by 'id' and sort by 'time'
    grouped_users = dataframe.sort_values('time').groupby('id')
    
    # Iterate through each customer and determine if their offers were successful
    for customer, customer_df in grouped_users:
        customer_map = is_offer_successfull(customer_df, dataframe)
        successful_map.update(customer_map)
    
    # Fill the 'successful_offer' column with the results from the successful_map
    dataframe['successful_offer'] = dataframe['successful_offer'].fillna(successful_map)
    
    return dataframe


def get_analysis_df(df, portfolio):
    """
    Prepare a DataFrame for analysis by cleaning and merging with a portfolio DataFrame.

    Parameters:
    df (DataFrame): The original DataFrame containing offer data.
    portfolio (DataFrame): The portfolio DataFrame containing offer details.

    Returns:
    DataFrame: A cleaned and merged DataFrame ready for analysis.

    The function performs the following steps:
    1. Drops the 'value', 'time', and 'event' columns from the original DataFrame.
    2. Removes rows with any missing values.
    3. Drops duplicate rows, keeping the first occurrence.
    4. Merges the cleaned DataFrame with the portfolio DataFrame on the 'offer_id' column using an inner join.
    5. Drops the 'difficulty_x' and 'reward_x' columns from the merged DataFrame.
    6. Renames the 'difficulty_y' and 'reward_y' columns to 'difficulty' and 'reward', respectively.

    This results in a DataFrame that is cleaned, merged, and ready for further analysis.
    """
    df_analysis = df.drop(['value', 'time', 'event'], axis=1) \
        .dropna(axis=0) \
        .drop_duplicates(keep='first') \
        .merge(portfolio, on='offer_id', how='inner') \
        .drop(['difficulty_x', 'reward_x'], axis=1) \
        .rename(columns={'difficulty_y': 'difficulty', 'reward_y': 'reward'})
    return df_analysis


def save_df_to_csv(df, filepath):
    """
    Save a given DataFrame to a CSV file.

    Parameters:
    df (DataFrame): The DataFrame to be saved.
    filepath (str): The path (including the filename) where the CSV file will be saved.

    This function saves the provided DataFrame to a CSV file at the specified filepath. The index of the DataFrame is not included in the CSV file.
    """
    df.to_csv(path_or_buf=filepath, index=False)


def engineer_final_df(df):
    """
    Engineer the DataFrame to be ready for machine learning.

    This function processes the input DataFrame by dropping unnecessary columns, handling missing values,
    and removing duplicates. It also includes assertions to ensure there are no missing values and that
    the 'successful_offer' column contains only binary values (0 and 1).

    Args:
        df (pd.DataFrame): The input DataFrame to be engineered.

    Returns:
        pd.DataFrame: The processed DataFrame ready for machine learning.
    """
    # Drop unnecessary columns
    df = df.drop(['id', 'offer_id', 'value', 'time', 'event'], axis=1)
    
    # Drop rows with any missing values
    df = df.dropna(axis=0)
    
    # Remove duplicate rows, keeping the first occurrence
    df = df.drop_duplicates(keep='first')
    
    # Assert that there are no missing values in the DataFrame
    assert df.isnull().sum().sum() == 0
    
    # Assert that the 'successful_offer' column contains only binary values (0 and 1)
    assert list(df['successful_offer'].value_counts().to_dict().keys()) == [0, 1]
    
    return df


def main():
    """
    Main function to load, engineer DataFrames, and save the cleaned DataFrames in CSV format.

    This function performs the following steps:
    1. Loads the data from JSON files.
    2. Engineers the portfolio, profile, and transcript DataFrames.
    3. Merges the engineered DataFrames into a complete DataFrame.
    4. Fills the 'successful_offer' column in the complete DataFrame.
    5. Prepares a DataFrame for analysis and ensures it has the correct set of values.
    6. Engineers the final DataFrame for machine learning.
    7. Saves the cleaned and engineered DataFrames to CSV files.

    The function also includes assertions to verify the integrity of the DataFrames.

    Returns:
        None
    """
    if len(sys.argv) == 4:
        portfolio_path, profile_path, transcript_path = sys.argv[1:]
        print("Loading data...")
        portfolio, profile, transcript = load_data(portfolio_path, profile_path, transcript_path)
        portfolio_tmp = portfolio.copy()

        print("Engineering DataFrames...")
        engineered_portfolio = portfolio_engineering(portfolio)
        engineered_profile = profile_engineering(profile)
        engineered_transcript = transcript_engineering(transcript, profile)
        
        complete_df = get_complete_dataframe(
            portfolio_after_engineer=engineered_portfolio,
            profile_after_engineered=engineered_profile,
            transcript_after_engineered=engineered_transcript
        )
        success_df = fill_successful_offers(dataframe=complete_df)

        portfolio_tmp = portfolio_tmp.rename(columns={'id': 'offer_id'})

        df_analysis = get_analysis_df(df=success_df, portfolio=portfolio_tmp)

        # # Test that the DataFrame for analysis has the correct set of values for offer IDs and user IDs
        assert not (set(df_analysis['id']) - set(transcript['person']))
        assert not (set(df_analysis['offer_id']) - set(portfolio_tmp['offer_id']))
        df_for_machine_learning = engineer_final_df(df=success_df)

        print("Saving DataFrames to CSV...")
        save_df_to_csv(df_for_machine_learning, 'cleaned_data.csv')
        save_df_to_csv(df_analysis, 'data_for_analysis.csv')
        save_df_to_csv(engineered_portfolio, 'engineered_portfolio.csv')
        save_df_to_csv(engineered_profile, 'engineered_profile.csv')

        print("DataFrames successfully saved!")
    else:
        print("""Please provide the filepaths for the portfolio dataset, the profle dataset, the transcript dataset.\n\n
              Example: python data_processing.py portfolio.json profile.json transcript.json""")


if __name__ == '__main__':
    main()