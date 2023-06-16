import pandas as pd
import requests
import io

data = requests.get("https://api.alternative.me/fng/?limit=700&format=csv").text

data = io.StringIO(data)
df   = pd.read_csv(data, skiprows=4, skipfooter=5, header=None)
df   = df.rename({0:"Date", 1:"FGI", 2:"FGI_bin"}, axis=1)
df.to_csv("\\FGIndex.csv", index=False)