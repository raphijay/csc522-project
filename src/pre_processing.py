import pandas as pd

def preprocess():
    # Read in the forex dataset
    forex       = pd.read_csv('forex.csv')

    # Read in the USD/INR dataset
    usd_inr     = pd.read_csv('USDINR.csv')

    # Concatenate the two datasets vertically
    forex_data  = pd.concat([forex, usd_inr], axis = 0)

    # Drop any rows with missing values
    forex_data.dropna(inplace = True)

    # Drop any duplicate rows
    forex_data.drop_duplicates(inplace = True)

    # Reset the index
    forex_data.reset_index(drop = True, inplace = True)

    # Save the merged dataset to a new CSV file
    preprocessed_df = forex_data.to_csv('forex_dataset.csv', index = False)

    return preprocessed_df
