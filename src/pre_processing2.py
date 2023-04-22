import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def epoch_days(date):

    return (datetime.datetime.strptime(date, "%Y-%m-%d") - datetime.datetime(1970, 1, 1)).days


def pre_p_forex(data):

    newdata = data[data.slug == 'USD/INR'].copy()
    # Encoding all dates into an integer based on days since epoch (1/1/1970)

    newdata['date'] = newdata['date'].apply(epoch_days)

    # drop certain columns here
    newdata.drop(columns=['slug', 'currency'], inplace=True)

    return newdata

def pre_p_usdinrx(data):

        # Encoding all dates into an integer based on days since epoch (1/1/1970)

    newdata = data.copy()
    newdata['Date'] = newdata['Date'].apply(epoch_days)
    # drop certain columns here
    newdata.drop(columns=['Volume', 'Adj Close'], inplace=True)

    return newdata


def merge(forex, usdinrx):

    usdinrx = usdinrx.rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close"})

    merged_data = pd.concat([forex, usdinrx], axis=0, ignore_index=True)

    # Drop any duplicate rows based on date
    merged_data.drop_duplicates(subset = "date", inplace=True)

    return merged_data


def split(data):
    # Split data into x and y
    X = data[['date', 'open', 'high', 'low']]
    y = data.close

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=25)

    return X_train, X_test, y_train, y_test


def normalize(training_data, test_data):

    # spliting data required, this function does not split!

    ss = StandardScaler().fit(training_data)

    return ss.transform(training_data), ss.transform(test_data)


