[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="1100" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **CryptoVolumeForecasting-Validation_test_plot** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

```yaml

Name of QuantLet: CryptoVolumeForecasting-Validation_test_plot

Published in: Cryptocurrency liquidity forecasting

Description: This Quantlet visualizes the market liquidity prediction for various methods and different time periods.

Keywords: cryptocurrency, Bitcoin, visualization, LSTM, TBATS, SARIMAX

Author: Ilyas Agakishiev

Submitted: Friday, 14 of July 2023 by Ilyas Agakishiev

```

![Picture1](Test.png)

![Picture2](Validation_1.png)

![Picture3](Validation_2.png)

![Picture4](Validation_3.png)

![Picture5](Validation_4.png)

### PYTHON Code
```python

import pandas as pd
import matplotlib.pyplot as plt

df       = pd.read_csv(r"\Results_comparison.csv", index_col=0)
df.index = pd.to_datetime(df.index, dayfirst=True)
start    = 800 # Change this value for different periods
stop     = start+200

fig = plt.figure()
ax  = fig.add_subplot(111)

ax.set_ylabel("Net Bitcoin sold to Bitwala")

plt.plot(df["SARIMA"].iloc[start:stop], linewidth=1, color="tomato")
plt.plot(df["TBATS"].iloc[start:stop], linewidth=1, color="steelblue")
plt.plot(df["LSTM"].iloc[start:stop], linewidth=1, color="green")
plt.plot(df["TS"].iloc[start:stop], linewidth=1, color="black")
plt.show()

```

automatically created on 2023-07-22