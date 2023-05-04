import pandas as pd
import numpy as np
from datetime import datetime
from os import walk
import quandl

# Aggregate quote data. We use 4H for 4 hours, but other aggregations are possible
def quotes_compressor(period="1D"):
    mypath    = r"\\data\\quotes\\" # Set your path here
    file_list = []
    for (dirpath, dirnames, filenames) in walk(mypath):
        file_list.extend(filenames)
        break
    df_full = pd.DataFrame()
    for file in file_list:
        df              = pd.read_csv(mypath+file)
        df["quoted_at"] = pd.to_datetime(df["quoted_at"])
        df.index        = df["quoted_at"]
        df["mid"]       = (df["ask"]+df["bid"])/2
        df["spread"]    = df["ask"]-df["bid"]
        try:
            df = df.drop(df[df["mid"]==0].index, axis=0)
        except:
            pass
        df         = df.drop(["quoted_at","bid", "ask"], axis=1)
        df1        = df.resample(period, convention="end").last()
        st         = df["mid"].resample("1T").last()
        df1["std"] = st.resample(period).std()
        df1["min"] = df["mid"].resample(period).min()
        df1["max"] = df["mid"].resample(period).max()        
        df_full    = pd.concat([df_full,df1], axis=0)
    df_full = df_full.sort_index()
    return df_full

# Aggregate order volume data, generate some featuress
def order_volume_compressor(period="1D"):
    df                     = pd.read_csv(r"\data\orders_2019_2020.csv") # Set your path here
    df["input_amount"]     = df["input_amount"]*(-1)
    df["balance_EUR"]      = 0
    df["balance_BTC"]      = 0
    df["trade_count_buy"]  = 0
    df["trade_count_sell"] = 0    
    df.loc[df["input_currency"]=="EUR","balance_EUR"]  = df.loc[df["input_currency"]=="EUR","input_amount"]
    df.loc[df["input_currency"]=="BTC","balance_BTC"]  = df.loc[df["input_currency"]=="BTC","input_amount"]
    df.loc[df["output_currency"]=="EUR","balance_EUR"] = df.loc[df["output_currency"]=="EUR","output_amount"]
    df.loc[df["output_currency"]=="BTC","balance_BTC"] = df.loc[df["output_currency"]=="BTC","output_amount"]
    df["quoted_at"] = pd.to_datetime(df["quoted_at"])
    df["BTC_buy"]   = df["balance_BTC"].copy()
    df["BTC_sell"]  = df["balance_BTC"].copy()
    df.loc[df["balance_BTC"]>=0,"BTC_buy"]      = 0
    df.loc[df["balance_BTC"]<0,"BTC_sell"]      = 0    
    df.loc[df["BTC_buy"]<0,"trade_count_buy"]   = 1
    df.loc[df["BTC_sell"]>0,"trade_count_sell"] = 1    
    df.index = df["quoted_at"]
    df       = df.drop(["quoted_at","input_amount","output_amount","input_currency","output_currency"], axis=1)
    df1      = df.resample(period, convention="end").sum()
    return df1

# Aggregate order ID data, generate some features
def orders_id_features(period="1D"):
    df = pd.read_csv(r"\data\orders_new.csv") # Set your path here
    df["quotedAt"]  = pd.to_datetime(df["quotedAt"])
    df["timestamp"] = df["quotedAt"].astype(int)/1000000000
    df_userinfo     = pd.DataFrame(columns=("owner","first_appearance","last_activity","inputBTC","inputEUR","outputBTC", "outputEUR", "number_of_transactions", "avg_volume_per_transaction"))
    df_features     = pd.DataFrame(index=(np.arange(4500)),columns=("timestamp", "traders_count", "last_activity", "new_traders_count", "percentage_new_traders", "inputBTC_new_traders", "outputBTC_new_traders", "percentage_new_traders_volume","average_age", "weighted_sell_buy_ratio_of_traders"))
    df_features     = df_features.fillna(0)
    df_userinfo     = df_userinfo.fillna(0)    
    curr_time_frame = datetime(2019,3,26,9,0,0).timestamp()
    time_step       = datetime(2019,3,26,8,0,0).timestamp() - datetime(2019,3,26,4,0,0).timestamp()
    feature_index   = 0
    for i in range(len(df)):
        if df.loc[i, "timestamp"] >= curr_time_frame + time_step:
            df_features.loc[feature_index,"timestamp"] = curr_time_frame
            feature_index += 1
            curr_time_frame+=time_step
        if i == 0:
            df_userinfo = df_userinfo.append(df.loc[i,["owner","inputBTC","inputEUR","outputBTC", "outputEUR"]])
            df_userinfo["first_appearance"].iloc[-1]           = df["timestamp"].iloc[i]
            df_userinfo["last_activity"].iloc[-1]              = df["timestamp"].iloc[i]
            df_userinfo["number_of_transactions"].iloc[-1]     = 1
            df_features.loc[feature_index,"traders_count"]     += 1
            df_features.loc[feature_index,"new_traders_count"] += 1
            df_features.loc[feature_index,"inputBTC_new_traders"] += df.loc[i,"inputBTC"]
            df_features.loc[feature_index,"outputBTC_new_traders"] += df.loc[i,"outputBTC"]
            df_features.loc[feature_index,"average_age"] +=0 
            df_features.loc[feature_index,"last_activity"] += df.loc[i,"timestamp"] 
            df_features.loc[feature_index,"weighted_sell_buy_ratio_of_traders"] += (df.loc[i,"inputBTC"]-df.loc[i,"outputBTC"]) 

        elif df_userinfo["owner"].iloc[-1] < df["owner"][i]:
            df_userinfo = df_userinfo.append(df.loc[i,["owner","inputBTC","inputEUR","outputBTC", "outputEUR"]])
            df_userinfo["first_appearance"].iloc[-1] = df["timestamp"].iloc[i]
            df_userinfo["last_activity"].iloc[-1] = df["timestamp"].iloc[i]
            df_userinfo["number_of_transactions"].iloc[-1] = 1
            df_features.loc[feature_index,"traders_count"] += 1
            df_features.loc[feature_index,"new_traders_count"] += 1
            df_features.loc[feature_index,"inputBTC_new_traders"] += df.loc[i,"inputBTC"]
            df_features.loc[feature_index,"outputBTC_new_traders"] += df.loc[i,"outputBTC"]
            df_features.loc[feature_index,"average_age"] += 0 #Divide by number of traders later
            df_features.loc[feature_index,"last_activity"] += df.loc[i,"timestamp"] #Divide by number of traders later
            df_features.loc[feature_index,"weighted_sell_buy_ratio_of_traders"] += (df.loc[i,"inputBTC"]-df.loc[i,"outputBTC"]) #Divide by volume later
        else:
            df_userinfo.loc[df_userinfo["owner"]==df["owner"][i],["inputBTC","inputEUR","outputBTC", "outputEUR"]]=df_userinfo.loc[df_userinfo["owner"] == df["owner"][i],["inputBTC","inputEUR","outputBTC", "outputEUR"]]+df.loc[i,["inputBTC","inputEUR","outputBTC", "outputEUR"]]
            df_userinfo.loc[df_userinfo["owner"]==df["owner"][i],"number_of_transactions"] += 1
            df_userinfo["last_activity"].iloc[-1] = df["timestamp"].iloc[i]
            df_features.loc[feature_index,"traders_count"] += 1
            df_features.loc[feature_index,"average_age"] += -df_userinfo.loc[df_userinfo["owner"]==df.loc[i,"owner"],"first_appearance"].iloc[0]+df.loc[i,"timestamp"] #Divide by number of traders later
            df_features.loc[feature_index,"last_activity"] += df_userinfo.loc[df_userinfo["owner"]==df.loc[i,"owner"],"last_activity"].iloc[0] #Divide by number of traders later
            df_features.loc[feature_index,"weighted_sell_buy_ratio_of_traders"] += df_userinfo.loc[df_userinfo["owner"]==df["owner"][i],"inputBTC"].iloc[0]-df_userinfo.loc[df_userinfo["owner"]==df["owner"][i],"outputBTC"].iloc[0] #Divide by volume later
    df_features["average_age"]   /= df_features["traders_count"]
    df_features["last_activity"] /= df_features["traders_count"]

    df_userinfo["avg_volume_per_transaction"]  = (df_userinfo["inputBTC"]+df_userinfo["outputBTC"])/df_userinfo["number_of_transactions"]
    df_features.loc[feature_index,"timestamp"] = curr_time_frame
    df_features["percentage_new_traders"]      = df_features["new_traders_count"] / df_features["traders_count"]
    return df_features, df_userinfo

# Aggregate EUR/USD currency data
def eur_usd_aggregator(period="1D"):
    mypath = "\\data\\EUR_USD\\" # Set your path here
    file_list = []
    for (dirpath, dirnames, filenames) in walk(mypath):
        file_list.extend(filenames)
        break
    df_full=pd.DataFrame()
    for file in file_list:
        df       = pd.read_csv(mypath+file, sep=";", header=None, usecols=[0,4], indsex_col=0)
        df       = df.rename({4:"EURUSD"}, axis=1)
        df.index = pd.to_datetime(df.index)
        df_full  = pd.concat([df_full, df], axis=0)
    df_full = df_full.resample(period, convention="end").last()
    df_full = df_full.fillna(method="ffill")
    return df_full

# Load EURIBOR data from Quandl
def euribor_load():
    get_list=["BOF/QS_D_IEUEONIA","BOF/QS_D_IEUTIO1M","BOF/QS_D_IEUTIO3M","BOF/QS_D_IEUTIO6M","BOF/QS_D_IEUTIO1A"]
    quandl.ApiConfig.api_key = '' # Enter API key here
    df=pd.DataFrame()
    for i in get_list:
        df1 = quandl.get(i,start_date='2019-01-01', end_date='2021-01-31')
        df1 = df1.rename({"Value":i[-2:]}, axis=1)
        df  = df.merge(df1, how="outer", left_index=True, right_index=True)       
    return df

# Run code and save    
orders  = order_volume_compressor("4H")
quotes  = quotes_compressor("4H")
eur_usd = eur_usd_aggregator("4H")
orders_extra, userbase_info = orders_id_features("4H")
euribor = euribor_load()

# Change path if desired
orders.to_csv("orders_4H.csv")
quotes.to_csv("quotes_4H.csv")
eur_usd.to_csv("eur_usd.csv")
orders_extra.to_csv("orders_extra.csv")
userbase_info.to_csv("userbase_info.csv")
euribor.to_csv("EURIBOR.csv", index=True)