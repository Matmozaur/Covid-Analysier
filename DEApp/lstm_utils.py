import pandas as pd
from datetime import timedelta, datetime
import pickle
from datetime import date
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import numpy as np
from DEApp.data_loader import COVID_DATA


def prepare_one_dim_LSTM(x_train, n_future, lstm_units=200, dense_neurons=None,
                         dense_activation='relu', optimizer='adam',
                         loss='mean_absolute_error'):
    if dense_neurons is None:
        dense_neurons = [70]
    regressor = Sequential()
    regressor.add(Bidirectional(LSTM(units=lstm_units,
                                     input_shape=(x_train.shape[1], 1))))
    for d_n in dense_neurons:
        regressor.add(Dense(d_n, activation=dense_activation))

    regressor.add(Dense(units=n_future, activation='linear'))
    regressor.compile(
        optimizer=optimizer,
        loss=loss)
    return regressor


def prepare_lstm_pred_new_cases():
    today = datetime.now()
    time_prior = today - timedelta(days=15)
    new_cases_pl = COVID_DATA[COVID_DATA['location'] == 'Poland']['new_cases']
    new_cases_pl = new_cases_pl[new_cases_pl.index >= time_prior]
    with open('data/new_cases_scaler.pickle', 'rb') as fp:
        new_cases_scaler = pickle.load(fp)
    with open('data/new_cases_seasonal.pickle', 'rb') as fp:
        new_cases_pl_seasonal = pickle.load(fp)
    day = date.weekday(new_cases_pl.index[0])
    new_cases_pl_seasonal2 = np.concatenate((new_cases_pl_seasonal.values[day:],
                                             new_cases_pl_seasonal.values[:day]))
    new_cases_pl_seasonal2 = np.concatenate((new_cases_pl_seasonal2,
                                             new_cases_pl_seasonal2))
    new_cases_x = new_cases_pl.values[-14:] / new_cases_pl_seasonal2
    new_cases_x = new_cases_scaler.transform(new_cases_x.reshape(-1, 1))
    new_cases_x = new_cases_x.reshape(-1)
    new_cases_x = new_cases_x.reshape(1, 14, 1)
    regressor = prepare_one_dim_LSTM(new_cases_x, 7)
    regressor.build(input_shape=(1, 14, 1))
    regressor.load_weights('data/new_cases_lstm.h5')
    pred = regressor.predict(new_cases_x)
    pred = new_cases_scaler.inverse_transform(pred)
    day = date.weekday(new_cases_pl.index[-1])
    day = (day + 1) % 7
    new_cases_pl_seasonal = np.concatenate((new_cases_pl_seasonal.values[day:],
                                            new_cases_pl_seasonal.values[:day]))
    pred *= new_cases_pl_seasonal
    idx = [new_cases_pl.index[-1] + timedelta(days=i) for i in range(1, 8)]
    pred = pd.Series(pred.reshape(-1), index=idx)
    return new_cases_pl, pred


def prepare_lstm_pred_new_deaths():
    today = datetime.now()
    time_prior = today - timedelta(days=15)
    new_deaths_pl = COVID_DATA[COVID_DATA['location'] == 'Poland']['new_deaths']
    new_deaths_pl = new_deaths_pl[new_deaths_pl.index >= time_prior]
    with open('data/new_deaths_scaler.pickle', 'rb') as fp:
        new_deaths_scaler = pickle.load(fp)
    with open('data/new_deaths_seasonal.pickle', 'rb') as fp:
        new_deaths_pl_seasonal = pickle.load(fp)
    day = date.weekday(new_deaths_pl.index[0])
    new_deaths_pl_seasonal2 = np.concatenate((new_deaths_pl_seasonal.values[day:],
                                             new_deaths_pl_seasonal.values[:day]))
    new_deaths_pl_seasonal2 = np.concatenate((new_deaths_pl_seasonal2,
                                             new_deaths_pl_seasonal2))
    new_deaths_x = new_deaths_pl.values[-14:] / new_deaths_pl_seasonal2
    new_deaths_x = new_deaths_scaler.transform(new_deaths_x.reshape(-1, 1))
    new_deaths_x = new_deaths_x.reshape(-1)
    new_deaths_x = new_deaths_x.reshape(1, 14, 1)
    regressor = prepare_one_dim_LSTM(new_deaths_x, 7, lstm_units=100)
    regressor.build(input_shape=(1, 14, 1))
    regressor.load_weights('data/new_deaths_lstm.h5')
    pred = regressor.predict(new_deaths_x)
    pred = new_deaths_scaler.inverse_transform(pred)
    day = date.weekday(new_deaths_pl.index[-1])
    day = (day + 1) % 7
    new_deaths_pl_seasonal = np.concatenate((new_deaths_pl_seasonal.values[day:],
                                            new_deaths_pl_seasonal.values[:day]))
    pred *= new_deaths_pl_seasonal
    idx = [new_deaths_pl.index[-1] + timedelta(days=i) for i in range(1, 8)]
    pred = pd.Series(pred.reshape(-1), index=idx)
    return new_deaths_pl, pred
