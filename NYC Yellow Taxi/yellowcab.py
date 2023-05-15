import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn import metrics
def import_data(file_name) -> pd.DataFrame: 
    df = pd.read_csv (file_name)  #The data in the file is read into a DataFrame
    df = df[df['total_amount'] > 0] #keep only values which are greater than zero under the total_amount column 
    return df 

def add_tip_time_features(df) -> pd.DataFrame: 
    """
    The function computes 3 new columns:
    percent_tip: which is 100*tip_amount/(total_amount-tip_amount)
    duration: the time the trip took in seconds.
    dayofweek: the day of the week that the trip started, represented as 0 for Monday, 1 for Tuesday, ... 6 for Sunday.
    The original DataFrame with these additional three columns is returned."""
    return df
def impute_numeric_cols(df, x_num_cols) -> pd.DataFrame: 
    #x_num_cols: a list of numerical columns in df.
    #Missing data in the columns x_num_cols are replaced with the median of the column. Returns a DataFrame containing only the imputed numerical columns from input df.
    df = df[x_num_cols].fillna(df[x_num_cols].median())
    return df

def transform_numeric_cols(df_num, degree_num=2) -> np.ndarray: 
    """
    An object of sklearn.preprocessing.PolynomialFeatures is instantiated with inputs degree=degree_num. 
    The model is than fitted (using fit_transform) to the numeric columns, df_num_imputed, and the resulting array is returned.
    """
    transformer = PolynomialFeatures(degree=degree_num  , include_bias=False )
    X = transformer.fit_transform(df_num)
    return X
     
def fit_linear_regression(x_train, y_train):
    #Fits a linear model to x_train and y_train and returns the intercept, model coefficients, and model object.
    linear_model = LinearRegression()
    linear_model.fit(x_train, y_train) #the model has been trained now 
    return (linear_model.intercept_,linear_model.coef_,linear_model)
def predict_using_trained_model(mod, x, y): 
    """
    mod: a trained model for the data.
    x: an array or DataFrame of numeric columns with no null values.
    y: an array or DataFrame of numeric columns with no null values.
    Computes and returns the mean squared error and r2 score between the values predicted by the model (mod on x) and the actual values (y). Note that sklearn.metrics contains two functions that may be of use: mean_squared_error and r2_score."""
    predicted  = mod.predict(x) #using the trained model to predict the x value
    return metrics.mean_squared_error (y,predicted ), metrics.r2_score(y,predicted ) #now comparing to the actual value (y), to see how close the prediction was 