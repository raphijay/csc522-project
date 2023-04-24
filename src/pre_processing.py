import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

##
# Takes a date string in the format "YYYY-MM-DD" as input and returns the number of days between that
# date and January 1, 1970, which is the epoch date in Unix time.
##
def epoch_days(date):
    return (datetime.datetime.strptime(date, "%Y-%m-%d") - datetime.datetime(1970, 1, 1)).days

##
# Drops columns & rows with missing values in Forex dataset
#
# inplace = True parameter in a dataframe method modifies the dataframe itself instead of returning a
# copy of the modified dataframe.
##
def pre_p_forex(data):
    if (not hasattr(data, 'slug')):
        return None

    # Makes a copy of the resulting dataframe.
    newdata = data[data.slug == 'USD/INR'].copy()

    # Drop any rows with missing values
    newdata.dropna(inplace = True)

    # Drop certain columns here
    newdata.drop(columns = ['slug', 'currency'], inplace = True)

    # Encoding all dates into an integer based on days since epoch (1/1/1970)
    newdata['date'] = newdata['date'].apply(epoch_days)

    return newdata

##
# Drops columns & rows with missing values in USD/INR dataset
#
# inplace = True parameter in a dataframe method modifies the dataframe itself instead of returning a
# copy of the modified dataframe.
##
def pre_p_usdinr(data):

    # Makes a copy of the resulting dataframe.
    newdata = data.copy()

    # Drop any rows with missing values
    newdata.dropna(inplace = True)

    # Drop certain columns here
    newdata.drop(columns=['Volume', 'Adj Close'], inplace = True)

    # Encoding all dates into an integer based on days since epoch (1/1/1970)
    newdata['Date'] = newdata['Date'].apply(epoch_days)

    return newdata

##
# Concatenates two dataframes (forex and usdinrx) vertically (i.e., side-by-side) along the row
# axis (axis = 0). The resulting merged dataframe has columns from both dataframes. The
# ignore_index = True parameter is used to ignore the original row indexes of the input dataframes and
# generate a new index based on the merged dataframe. If ignore_index is not set to True, the row index
# of the merged dataframe will have duplicated values.
#
# For example, suppose forex has columns 'date', 'open', 'high', and 'low', and usdinrx has columns
# 'date' and 'close'. After merging the dataframes using pd.concat(), the resulting merged_data
# dataframe will have columns 'date', 'open', 'high', 'low', and 'close' in that order. The number of
# rows in merged_data will be the same as the number of rows in forex and usdinrx.
##
def merge(forex, usdinrx):

    merged_data = pd.concat([forex, usdinrx], axis = 0, ignore_index = True)

    # Drop any duplicate rows based on date
    merged_data.drop_duplicates(subset = "date")

    return merged_data

##
# Splits it into training and testing sets.
#
# Function separates input data into X and y. X consists of features used to predict target variable y.
# In this case, features are date, open price, high price, and low price. y is target variable, which
# is closing price of forex for given date.
#
# Next, function uses train_test_split() function from scikit-learn to randomly split the data
# into training and testing sets. test_size parameter is set to 0.25, which means that 25% of the
# data will be used for testing, and remaining 75% will be used for training. random_state parameter is
# set to 25 to ensure that results are reproducible.
#
# Finally, function returns four variables: X_train, X_test, y_train, and y_test. X_train and
# y_train contain training data for features and target variable, respectively. X_test and y_test
# contain testing data for features and target variable, respectively. These variables can then be used
# to train and evaluate a machine learning model for forex prediction.
##
def split(data):
    # Split data into X and y
    X = data[['date', 'open', 'high', 'low']]
    y = data.close

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 522)

    return X_train, X_test, y_train, y_test

##
# Takes in training and testing data and returns their normalized versions using StandardScaler.
# StandardScaler is a function from scikit-learn library which scales the data to have zero mean and
# unit variance. It ensures that all features contribute equally to the model and prevents some
# features from dominating others.
#
# The fit method of StandardScaler is used to calculate the mean and variance of the training data, and # the transform method is used to apply the same scaling to both the training and testing data.
##
def normalize(training_data, test_data):

    # Spliting data required, this function does not split!
    ss = StandardScaler().fit(training_data)

    return ss.transform(training_data), ss.transform(test_data)