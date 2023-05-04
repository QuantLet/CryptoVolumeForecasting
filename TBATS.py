import pandas as pd
import numpy as np
from tbats import TBATS
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


if __name__=='__main__':

    data_input    = pd.read_csv("\\dataset_complete.csv", index_col=0)
    data_ts       = data_input.iloc[:,0]
    data_ts.index = pd.to_datetime(data_ts.index)

    
    estimator = TBATS(seasonal_periods=(6, 42),         
        use_arma_errors  = True,  # shall try models with and without ARMA
        use_box_cox      = False, # will not use Box-Cox
        use_trend        = True,  # will try models with trend and without it
        use_damped_trend = True,  # will try models with daming and without it
        show_warnings    = False) # will not be showing any warnings for chosen model)


    data = np.array([])
    for i in range(3046,len(data_ts),1):
        model      = estimator.fit(data_ts.iloc[i-420:i])
        y_forecast = model.forecast(steps=1)
        data       = np.append(data, y_forecast)

    data_tbat = pd.Series(data, index=data_ts.index)
    data_res  = data_ts-data_tbat

    data_TBATS = pd.Series(data, index=data_ts.iloc[-1000:].index)
    data_res   = data_ts.iloc[-1000:] - data_TBATS
    mse        = mean_squared_error(data_TBATS, data_ts.iloc[-1000:])    
    mae        = mean_absolute_error(data_TBATS, data_ts.iloc[-1000:])

    plt.plot(data_res)
    plt.plot(data_tbat)
    plt.plot(data_ts)

    data_tbat_output = pd.DataFrame(data=[data_TBATS, data_res], index=["TBATS","TBATS_residuals"]).transpose()
    data_tbat_output.to_csv(r"C:\Users\Ilyas Agakishiev\Documents\GitHub\Bitwala_Volume_Forecasting\TBATS_output.csv", index=True)