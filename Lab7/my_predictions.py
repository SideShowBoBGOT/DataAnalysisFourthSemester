# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from datetime import datetime


def get_predictions(df_input: pd.DataFrame, column_train: str,
                train_period: int, nodes_number: int=50,
                dense: int=25) -> np.array:
    df = df_input.loc[:, column_train]
    dataset = df.values.reshape(-1, 1)
    training_data_len = dataset.shape[0]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    train_data = scaled_data[0:training_data_len, :]
    x_train, y_train = [], []
    for i in range(train_period, len(train_data)):
        x_train.append(train_data[i-train_period:i, 0])
        y_train.append(train_data[i, 0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    model = Sequential()
    model.add(LSTM(nodes_number, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(nodes_number, return_sequences=False))
    model.add(Dense(dense))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    
    test_data = scaled_data[training_data_len - train_period:, :]
    x_predict = np.array([test_data[-train_period:, 0]])
    x_predict = np.reshape(x_predict, (x_predict.shape[0], x_predict.shape[1], 1))

    preds = []
    for i in range(train_period):
        predictions = model.predict(x_predict)
        x_predict = np.roll(x_predict, -1)
        preds.append(predictions[0][0])
        np.append(x_predict[0], predictions[0])

    predictions = np.array(preds).reshape(-1, 1)
    predictions: np._ArrayFloat_co = scaler.inverse_transform(predictions)
    return predictions.flatten()


# path = 'data/seattleWeather_1948-2017 copy.csv'
# df = pd.read_csv(path, index_col=['DATE'], parse_dates=['DATE'])
# predictions_tmax = get_predictions(df, 'TMAX', 30, 50, 25)
