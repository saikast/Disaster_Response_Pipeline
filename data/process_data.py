import sys
import pandas as pd
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')


def load_data(messages_filepath, categories_filepath):
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = messages.merge(categories, on=('id'))

    return df


def clean_data(df):
    """
    Cleans the dataframe by splitting the 'categories' column into separate columns, 
    dropping unnecessary columns and values, and removing duplicates.

    Args:
        df: dataframe to be cleaned.

    Returns:
        df: Cleaned dataframe with updated columns and no duplicates.
    """
    # Split categories column and create a new dataframe
    categories_split = df['categories'].str.split(';', expand=True)
    column_names = categories_split.iloc[0].str.split('-', expand=True)[0]
    categories_split.columns = column_names

    # replace the values in dataframe with only the last numbers and change the type to int
    categories_clean = categories_split.apply(lambda x: x.str.split('-').str[-1].astype(int))

    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new `new_columns_updated` dataframe
    df = pd.concat([df, categories_clean], axis=1)

    # replace all the values of 2 with 1
    df['related'] = df['related'].apply(lambda x : 1 if x == 2 else x)

    # as chile_alone contains only 0's, hence dropped
    df.drop('child_alone', axis=1, inplace=True)

    # dropping the duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    con = create_engine('sqlite:///' + database_filename, echo=False)  
    df.to_sql('Disaster_data', con, if_exists='replace', index=False)  


def main():
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