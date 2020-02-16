import sys
# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

"""
To run this script use the command:
'python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db'

"""

def load_data(messages_filepath, categories_filepath):
    """
    Data Loading Function
    
    Arguments:
        messages_filepath -> path to disaster_messages.csv file
        categories_filepath -> path to disaster_categories.csv file
    Return:
        df - Merged messages and ccategories data in the form of pandas dataframe
    """
        
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    #merge the datasets
    df = pd.merge(messages, categories, on='id', how='left')
    
    return df



def clean_data(df):
    """
    Data Cleaning Function
    
    Arguments:
        df - original data
       
    Return:
        df - Clean dataframe with dummy variables and no duplicate values
    """
    categories = df.categories.str.split(';', expand = True)
    row = categories.loc[0]
    category_colnames = row.apply(lambda x: x[:-2]).values.tolist()
    categories.columns = category_colnames
    categories.related.loc[categories.related == 'related-2'] = 'related-1'
    
    ##Convert the categories to dummy variables
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        
    # drop the original categories column from `df` and # concatenate the original dataframe with the new `categories` dataframe    
    df.drop('categories', axis = 1, inplace = True)
    df = pd.concat([df, categories], axis = 1)
    #check for duplicates and remove them
    df.drop_duplicates(subset = 'id', inplace = True)
    return df

def save_data(df, database_filepath):
    """
    Function to save the dataframe to sql in the form of sqlite database
    
    Argument:
             df - data in the form of dataframe
             database_filepath - file path to the sqlite database
    """         
             
    
    engine = create_engine('sqlite:///' + database_filepath)
    df.to_sql('Response', engine, index=False, if_exists='replace')
     


def main():
    
    """
    Data Extraction, Transformation and Loading function
    
    This function creates the ETL pipeline which include:
        1) Extract data from two csv files (messages and categories files)
        2) Merge the two file
        3) Convert the categories to dummy variables
        4) Search for and drop duplicate values
        5) Load data to SQLite database
        
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