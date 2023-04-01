import pandas as pd

def remove_rows_with_null_values(dataFrame: pd.DataFrame):
    cleaned_df = dataFrame.dropna()
    return cleaned_df
