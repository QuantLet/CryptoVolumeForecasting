[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="1100" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **CryptoVolumeForecasting-Feature_engineering** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

```yaml

Name of QuantLet: CryptoVolumeForecasting-Feature_engineering

Published in: Cryptocurrency liquidity forecasting

Description: This Quantlet combines the data from various sources and generates new features to create one full dataset that can be used for LSTM.

Keywords: cryptocurrency, Bitcoin, feature engineering

Author: Ilyas Agakishiev

Submitted: Friday, 14 of July 2023 by Ilyas Agakishiev
```

### PYTHON Code
```python

import numpy as np
import pandas as pd
import holidays
import matplotlib.pyplot as plt
import ta
#import seaborn as sns

# Add your path here
path="" 

orders = pd.read_csv(path+"orders_4H.csv")
orders = orders[634:4680]

#sns.boxplot(y=orders["BTC_sell"])
#sns.histplot(x=orders["BTC_sell"])
#sns.histplot(x=orders["trade_count_sell"], binwidth=10, binrange=(0,300))

orders_extra = pd.read_csv(path+"orders_extra.csv")
orders_extra = orders_extra.dropna()
orders_extra["quoted_at"] = pd.to_datetime(orders_extra["timestamp"]*1000000000, utc=True)
orders_extra = orders_extra.iloc[:,2:]
orders_extra = orders_extra.drop(["percentage_new_traders_volume","last_activity"], axis=1)

quotes = pd.read_csv(path+"quotes_4H.csv")
plt.plot(quotes["mid"])

fgi = pd.read_csv(path+"FGIndex.csv")
vol = pd.read_csv(path+"BTCEUR_vol.csv")
euribor = pd.read_csv(path+"EURIBOR.csv",sep=";",decimal=",", skiprows=6, header=None, usecols=[0,1,2,3,4,5,6])
euribor = euribor.rename({0:"date",1:"EONIA",2:"1_month",3:"1_week",4:"12_months",5:"3_months",6:"6_months"}, axis=1)
euribor.iloc[:,1:] = euribor.iloc[:,1:].shift(6)
eur_usd = pd.read_csv(path+"eur_usd.csv")
eur_usd = eur_usd.rename({"0":"date"}, axis=1)


orders["quoted_at"] = pd.to_datetime(orders["quoted_at"])
orders=orders.merge(orders_extra, how="left",on="quoted_at")
quotes["quoted_at"] = pd.to_datetime(quotes["quoted_at"])
fgi["Date"] = pd.to_datetime(fgi["Date"])
fgi["Date"] = pd.DatetimeIndex(fgi["Date"]).normalize()
df=orders.merge(quotes, on="quoted_at", how="inner")
df["quoted_at"] = pd.DatetimeIndex(df["quoted_at"]).tz_localize(None)
df = df.merge(fgi,left_on="quoted_at",right_on="Date", how="left")
df = df.fillna(method="ffill")
vol["date"] = pd.to_datetime(vol["date"])
df = df.merge(vol[["date", "volume_eur"]],left_on="quoted_at",right_on="date", how="left")
df = df.fillna(method="ffill")

euribor["date"] = pd.to_datetime(euribor["date"])
df = df.merge(euribor,left_on="quoted_at",right_on="date", how="left")

eur_usd["date"] = pd.to_datetime(eur_usd["date"])
df = df.merge(eur_usd,left_on="quoted_at",right_on="date", how="left")

df = df.fillna(method="ffill")

df[["volume_eur","FGI","FGI_bin"]] = df[["volume_eur","FGI","FGI_bin"]].shift(6)

df       = ta.add_all_ta_features(df, open="open", high="max", low="min", close="mid", volume="volume_eur", fillna=True)
df.index = df["quoted_at"]
df["spread_ratio"]   = df["spread"]/df["mid"]
df["volume_MA_buy"]  = -df["BTC_buy"].rolling(window=42).mean()
df["volume_MA_sell"] = df["BTC_sell"].rolling(window=42).mean()
df["average_trade_volume_buy"]  = -df["BTC_buy"]/(df["trade_count_buy"]+0.00000001)
df["average_trade_volume_sell"] = df["BTC_sell"]/(df["trade_count_sell"]+0.00000001)
df["BTC_diff"] = -df["BTC_buy"]-df["BTC_sell"] 

df.to_csv(path+"dataset_partial.csv", index=False)

df["BTC_buy"]  = np.log((-df["BTC_buy"] + 0.01) * 100)
df["BTC_sell"] = np.log((df["BTC_sell"] + 0.01) * 100)

df["price_MA_day"]   = df["mid"].rolling(window=6).mean()
df["price_MA_week"]  = df["mid"].rolling(window=42).mean()
df["price_EWMA"]     = df["mid"].ewm(com=0.9).mean()
df["return_MA_day"]  = df["price_MA_day"].pct_change()
df["return_MA_week"] = df["price_MA_week"].pct_change()
df["return_EWMA"]    = df["price_EWMA"].pct_change()
df["return"]         = df["mid"].pct_change()
df = df.dropna()
df["weekday"] = df["quoted_at"].dt.weekday
df["weekday"] = df["weekday"].apply(lambda x: x>=5)
df["hour"]    = df["quoted_at"].dt.hour.astype(str)
df["holiday"] = df["quoted_at"].apply(lambda x: x in holidays.DE())
df["holiday"] = df["holiday"] | df["weekday"]
df["volume_MA_diff"] = df["volume_MA_buy"]-df["volume_MA_sell"] 

FGI_bin       = pd.get_dummies(df["FGI_bin"])
df            = pd.concat([df,FGI_bin], axis=1)
hours         = pd.get_dummies(df["hour"])
df            = pd.concat([df,hours], axis=1)
df["holiday"] = df["holiday"].astype(int)

df = df.drop(columns=["Date", "date_x", "date_y", "date", "quoted_at", "balance_EUR", "balance_BTC", "FGI", "weekday", "hour", "FGI_bin", "Neutral", "0"])
df = pd.concat([df.iloc[:,2:], df.iloc[:,:2]], axis=1)
df = pd.concat([df["BTC_diff"],df.drop("BTC_diff", axis=1)], axis=1)
df["BTCUSD"] = df["mid"] * df["EURUSD"]

df.to_csv(path+"dataset_complete.csv", index=True)

```

automatically created on 2023-07-22