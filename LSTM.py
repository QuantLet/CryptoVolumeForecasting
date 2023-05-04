import pandas as pd
import numpy as np
import pywt
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing
import random


# Low pass filter
def lowpassfilter(signal, layers = 3, wavelet="sym15"):

    coeff = pywt.wavedec(signal, wavelet, mode="constant")

    for i in range(0,layers):
        coeff[i].fill(0)
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="constant" )
    return reconstructed_signal

# Transform data into LSTM input format
def create_dataset(dataset, y_column=0, look_back=1):
	dataX, dataY = [], []
	for i in range(dataset.shape[0]-look_back-1):
		a = dataset[i:(i+look_back), :]
		dataX.append(a)
		dataY.append(dataset[i + look_back, y_column])
	return np.array(dataX), np.array(dataY)

# Read and prepare data
data     = pd.read_csv("C:\\Users\\Ilyas Agakishiev\\Documents\\GitHub\\Bitwala_Volume_Forecasting\\dataset_complete.csv", index_col=0)
col      = data.columns
ind      = data.index
scaler   = preprocessing.MinMaxScaler()
scaler_y = preprocessing.StandardScaler()

split   = int(round(0.85*data.shape[0]))
data1_1 = pd.DataFrame(scaler.fit_transform(data.iloc[:split,1:].values), columns=col[1:], index=ind[:split])
data1_2 = pd.DataFrame(scaler.transform(data.iloc[split:,1:].values), columns=col[1:], index=ind[split:])
data1   = pd.concat([data1_1, data1_2], axis=0)
data0_1 = pd.DataFrame(scaler_y.fit_transform(data.iloc[:split,0].values.reshape(-1,1)), columns=[col[0]], index=ind[:split])
data0_2 = pd.DataFrame(scaler_y.transform(data.iloc[split:,0].values.reshape(-1,1)), columns=[col[0]], index=ind[split:])
data0   = pd.concat([data0_1, data0_2], axis=0)

data          = pd.concat([data0,data1], axis=1)
data3         = pd.DataFrame(data.iloc[:,3:].apply(lowpassfilter,axis=0, args=(3,)).iloc[:-1,:].values, index=ind, columns=col[3:])
data          = pd.concat([data.iloc[:,:3],data3], axis=1)
lstm_input    = 200
dataX, dataY2 = create_dataset(data.values,0,lstm_input)



dataY  = data.iloc[lstm_input:,0]
dataY  = dataY.iloc[:-1].values
trainX = dataX[:split,:,:]
trainY = dataY[:split]
testX  = dataX[len(trainY):,:,:]
testY  = dataY[len(trainY):]

random.seed(1)

dropout_rate = 0.5

# Define LSTM model

model_lstm = Sequential()
model_lstm.add(LSTM(dataX.shape[2], return_sequences=True, activation = 'relu', dropout = dropout_rate, recurrent_dropout = dropout_rate))
model_lstm.add(LSTM(dataX.shape[2], activation = 'relu', dropout = dropout_rate, recurrent_dropout = dropout_rate))
model_lstm.add(Dense(dataX.shape[2], activation = 'relu'))
model_lstm.add(Dropout(dropout_rate))
model_lstm.add(Dense(dataX.shape[2], activation = 'relu'))
model_lstm.add(Dropout(dropout_rate))
model_lstm.add(Dense(dataX.shape[2], activation = 'relu'))
model_lstm.add(Dropout(dropout_rate))
model_lstm.add(Dense(1))

model_lstm.compile(
    loss='mean_squared_error',
    optimizer=Adam(clipvalue=1.0)
)

with tf.device("cpu:0"):
    model_lstm.fit(trainX, trainY, epochs=150)

# Make predictions
trainPredict = model_lstm.predict(trainX)
testPredict  = model_lstm.predict(testX)

# Invert predictions
trainPredict = scaler_y.inverse_transform(trainPredict)
trainY       = scaler_y.inverse_transform([trainY])
testPredict  = scaler_y.inverse_transform(testPredict)
testY        = scaler_y.inverse_transform([testY])

# Calculate mean squared error
trainScore = (mean_absolute_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f MAE' % (trainScore))
testScore  = (mean_absolute_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f MAE' % (testScore))
testScore  = (mean_absolute_error(testY[0,:-1], testY[0,1:]))
print('Reference Score: %.2f MAE' % (testScore))

trainScore = (mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f MSE' % (trainScore))
testScore  = (mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f MSE' % (testScore))
testScore  = (mean_squared_error(testY[0,:-1], testY[0,1:]))
print('Reference Score: %.2f MSE' % (testScore))

testY  = testY.reshape(-1,1)
trainY = trainY.reshape(-1,1)
plt.plot(testY)
plt.plot(testPredict)

