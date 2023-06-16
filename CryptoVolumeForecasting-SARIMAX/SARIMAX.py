import pandas as pd
import numpy as np
from pmdarima import auto_arima
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

if __name__=='__main__':

    data_input    = pd.read_csv("\\dataset_complete.csv", index_col=0)
    data_ts       = data_input.iloc[:,0]
    data_ts.index = pd.to_datetime(data_ts.index)
    
    # prepare Fourier terms
    exog          = pd.DataFrame({'date': np.arange(37,len(data_ts)+37)})
    exog.index    = data_ts.index
    exog          = exog % 42 + 1
    exog['sin']   = np.sin(2 * np.pi * exog["date"] / 42)
    exog['cos']   = np.cos(2 * np.pi * exog["date"] / 42)
    exog['sin_2'] = np.sin(4 * np.pi * exog["date"] / 42)
    exog['cos_2'] = np.cos(4 * np.pi * exog["date"] / 42)
    exog          = exog.drop(columns=['date'])

    
    arima_exog_model = auto_arima(y=data_ts.iloc[:420], exogenous=exog.iloc[:420,:], seasonal=True, m=6)# Forecast

    data = np.array([])
    for i in range(3046,len(data_ts),1):
        arima_exog_model      = auto_arima(y=data_ts.iloc[i-420:i], exogenous=exog.iloc[i-420:i,:], seasonal=True, m=6)# Forecast
        y_arima_exog_forecast = arima_exog_model.predict(n_periods=1, exogenous=exog.iloc[i:i+1,:])
        data                  = np.append(data, y_arima_exog_forecast)

    data_sarima = pd.Series(data, index=data_ts.index)
    data_res    = data_ts-data_sarima
    
    data_SARIMAX = pd.Series(data, index=data_ts.iloc[-1000:].index)
    data_res     = data_ts.iloc[-1000:] - data_SARIMAX

    mse = mean_squared_error(data_SARIMAX, data_ts.iloc[-1000:])    
    mae = mean_absolute_error(data_SARIMAX, data_ts.iloc[-1000:])
    
    plt.plot(data_sarima)
    plt.plot(data_ts)
    plt.plot(data_res)
    
    data_sarima_output = pd.DataFrame(data=[data_SARIMAX, data_res], index=["SARIMA","SARIMA_residuals"]).transpose()
    data_sarima_output.to_csv(r"\SARIMA_output.csv", index=True)