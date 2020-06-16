import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

data = pd.read_csv('C:\COSC 4381 - AI in Python\TensorFlow project\stock_dfs\GOOG.csv', date_parser=True)

print(data.tail())

'''
data_training = data[data['Date'] < '2019-01-01'].copy()
# print(data_training)

data_test = data[data['Date'] >= '2019-01-01'].copy()
# print(data_test)

training_data = data_training.drop(['Date', 'Adj Close'], axis=1)

# print(training_data.head())

scaler = MinMaxScaler()
training_data = scaler.fit_transform(training_data)
# print(training_data)

x_train = []
y_train = []

# Read the first 60 days of data

for i in range(60, training_data.shape[0]):
    x_train.append(training_data[i-60:i])
    y_train.append(training_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# print(x_train.shape, y_train.shape)

# #### BUILDING LSTM
#
regression = Sequential()

# print(x_train.shape[1], 5)

regression.add(LSTM(units=60, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 5)))
regression.add(Dropout(0.2))

regression.add(LSTM(units=60, activation='relu', return_sequences=True))
regression.add(Dropout(0.2))

regression.add(LSTM(units=80, activation='relu', return_sequences=True))
regression.add(Dropout(0.2))

regression.add(LSTM(units=120, activation='relu'))
regression.add(Dropout(0.2))

regression.add(Dense(units=1))

regression.summary()

regression.compile(optimizer='adam', loss='mean_squared_error')

regression.fit(x_train, y_train, epochs=50, batch_size=32)

#### PREPARE TEST DATASET

# data_test.head()

# data_training.tail(60)

past_60_days = data_training.tail(60)

df = past_60_days.append(data_test, ignore_index=True)
df = df.drop(['Date', 'Adj Close'], axis=1)

# print(df.head())

inputs = scaler.transform(df)

# print(inputs)

x_test = []
y_test = []

for i in range(60, inputs.shape[0]):
    x_test.append(inputs[i-60:i])
    y_test.append(inputs[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# print(x_test.shape, y_test.shape)

y_pred = regression.predict(x_test)

# print(y_pred, y_test)

# print(scaler.scale_)

scale = 1/8.17521128e-04
# print(scale)

y_pred = y_pred * scale
y_test = y_test * scale

#### VISUALIZATION

# Visualizing the results
plt.figure(figsize=(14, 5))
plt.plot(y_test, color='red', label='Real Google Stock Price')
plt.plot(y_pred, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
'''

