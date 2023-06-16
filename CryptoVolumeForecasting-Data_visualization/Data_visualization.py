import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

df = pd.read_csv("\\dataset_partial.csv", index_col=0)

huber = HuberRegressor(epsilon=1.0).fit(df["BTC_buy"].values.reshape(-1,1), df["BTC_sell"])

df = df.loc[df["BTC_buy"]!=0]
df = df.loc[df["BTC_sell"]!=0]
plt.scatter(df["BTC_buy"],df["BTC_sell"],s=4)

p = np.poly1d([huber.coef_[0],huber.intercept_])
plt.plot(df["BTC_buy"],p(df["BTC_buy"]),"r--")
plt.xlabel("Bitcoin sold to Bitwala")
plt.ylabel("Bitcoin sold by Bitwala")
plt.show()

#########################################

df["hour"] = df["hour"].astype(int)
df["BTC_buy"]=-df["BTC_buy"]
sns.boxplot("hour","BTC_diff", data=df.sort_values("hour"), color="steelblue").set(ylabel="Net Bitcoin sold to Bitwala")
sns.boxplot("hour","BTC_buy", data=df.sort_values("hour"), color="steelblue").set(ylabel="Bitcoin sold by Bitwala")
sns.boxplot("hour","BTC_sells", data=df.sort_values("hour"), color="steelblue").set(ylabel="Bitcoin sold to Bitwala")

#########################################

df["BTC_diff_log"] = np.log(-df["BTC_diff"]+df["BTC_diff"].max()+1)
sns.distplot(df["BTC_buy"])
