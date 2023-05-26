## main ##
import numpy as np

def calculate_MASE(training_series, testing_series, forecast_series) -> float:
    """
    Computes the MEAN-ABSOLUTE SCALED ERROR forcast error for univariate time series forecast.
    
    See "Another look at measures of forecast accuracy", Rob J Hyndman
    
    parameters:
        training_series: the series used to train the model, 1d numpy array
        testing_series: the test series to predict, 1d numpy array or float
        prediction_series: the prediction of testing_series, 1d numpy array (same size as testing_series) or float
        absolute: "squares" to use sum of squares and root the result, "absolute" to use absolute values.
    
    """
    print("Needs to be tested.")
    
    n = training_series.shape[0]
    d = np.abs(np.diff(training_series)).sum()/(n-1)
    
    errors = np.abs(testing_series - forecast_series)
    return errors.mean()/d


def calculate_smape(actual, forecast) -> float:
    # Convert actual and forecast  to numpy
    # array data type if not already
    if not all([isinstance(actual, np.ndarray), 
                isinstance(forecast, np.ndarray)]):
         actual, forecast  = np.array(actual), np.array(forecast)
  
    return round(\
        np.mean(\
        np.abs(forecast - actual) / ((np.abs(forecast) + np.abs(actual))/2))*100, \
            2)

def mae(y_hat, y_true):
    return np.mean(np.abs(y_hat-y_true))
