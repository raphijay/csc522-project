import pandas as pd

##
# Takes Pandas DataFrame dataFrame as input
# Drops rows by calling dropna() on cleanded_df dataframe. dropna() can remove rows or columns. By
# default it removes rows.
# @return returns a cleaned DataFrame with any rows that contain null or missing values removed.
##
def remove_rows_with_null_values(dataFrame: pd.DataFrame):
    cleaned_df = dataFrame.dropna()
    return cleaned_df
