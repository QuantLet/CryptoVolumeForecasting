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
