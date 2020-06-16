import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

goog = pd.read_csv('C:\COSC 4381 - AI in Python\TensorFlow project\stock_dfs\GOOG.csv', date_parser=True)
aapl = pd.read_csv('C:\COSC 4381 - AI in Python\TensorFlow project\stock_dfs\AAPL.csv', date_parser=True)
amzn = pd.read_csv('C:\COSC 4381 - AI in Python\TensorFlow project\stock_dfs\AMZN.csv', date_parser=True)
fb = pd.read_csv('C:\COSC 4381 - AI in Python\TensorFlow project\stock_dfs\FB.csv', date_parser=True)
nflx = pd.read_csv('C:\COSC 4381 - AI in Python\TensorFlow project\stock_dfs\\NFLX.csv', date_parser=True)

# print(nflx.tail())

companies = [goog, aapl, amzn, fb, nflx]

# combined_dataset = combined_dataset.append(aapl)

# y = goog.loc[:, ['Close']]
# y = y.append(aapl.loc[:, ['Close']])
# y = y.append(amzn.loc[:, ['Close']])
# y = y.append(fb.loc[:, ['Close']])
# y = y.append(nflx.loc[:, ['Close']])


# print(goog)
# print(y)

# Calculating moving average based on the Adjusted Close value
close_px = goog['Adj Close']
mavg = close_px.rolling(window=100).mean()

# Setting up the matplotlib
mpl.rc('figure', figsize=(8, 7))
# mpl.__version__
style.use('ggplot')

# Plotting the moving average
close_px.plot(label='GOOG')
mavg.plot(label='mavg')
plt.legend()
# plt.show()

# Plotting the return rate per sample
rets = close_px / close_px.shift(1) - 1
rets.plot(label='return')
# plt.show()

data_train = goog[goog['Date'] < '2019-01-01'].copy()
# print(data_train)

data_test = goog[goog['Date'] >= '2019-01-01'].copy()
# print(data_test)

training_data = data_train.drop(['Date'], axis=1)
print(training_data)

scaler = MinMaxScaler()
training_data = scaler.fit_transform(training_data)
print(training_data)

x_train = []
y_train = []

for i in range(30, training_data.shape[0]):
    x_train.append(training_data[i-30:i])
    y_train.append(training_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Sequential based model
regression = Sequential()

# input shape for the first layer of LSTM model
print(x_train.shape[1], 6)

regression.add(LSTM(units=60, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 6)))
regression.add(Dropout(0.1))    # drop 20% of random neurons

regression.add(LSTM(units=90, activation='relu', return_sequences=True))
regression.add(Dropout(0.1))

regression.add(LSTM(units=90, activation='relu', return_sequences=True))
regression.add(Dropout(0.2))

regression.add(LSTM(units=120, activation='relu'))
regression.add(Dropout(0.2))

regression.add(Dense(units=1))    # output layer for predicting the Open values of the dataset

regression.summary()    # summary of the model

regression.compile(optimizer='adam', loss='mean_squared_error')

regression.fit(x_train, y_train, epochs=10, batch_size=32)

past_30_days = data_train.tail(30)

df = past_30_days.append(data_test, ignore_index=True)
df = df.drop(['Date'], axis=1)

# print(df.head())

inputs = scaler.transform(df)

# print(inputs)

x_test = []
y_test = []

for i in range(30, inputs.shape[0]):
    x_test.append(inputs[i-30:i])
    y_test.append(inputs[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# print(x_test.shape, y_test.shape)

y_pred = regression.predict(x_test)

print(scaler.scale_)

scale = 1/8.17521128e-04
print("Inverse scale:", scale)

y_pred = y_pred * scale
y_test = y_test * scale

plt.figure(figsize=(14, 5))
plt.plot(y_test, color='red', label='Real Google Stock Price')
plt.plot(y_pred, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

