from datetime import datetime
import numpy as np
from DEApp.data_loader import COVID_DATA
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import datetime


def new_cases_xgboost():
    new_cases_pl = COVID_DATA[COVID_DATA['location'] == 'Poland']
    new_cases_pl['shift1'] = new_cases_pl['new_cases'].shift(1)
    new_cases_pl['shift2'] = new_cases_pl['shift1'].shift(1)
    new_cases_pl['shift3'] = new_cases_pl['shift2'].shift(1)
    new_cases_pl['shift4'] = new_cases_pl['shift3'].shift(1)
    new_cases_pl['shift5'] = new_cases_pl['shift4'].shift(1)
    new_cases_pl['shift6'] = new_cases_pl['shift5'].shift(1)
    new_cases_pl['shift7'] = new_cases_pl['shift6'].shift(1)

    X = new_cases_pl[['shift1', 'shift2', 'shift3', 'shift4', 'shift5', 'shift6', 'shift7']]
    Y = new_cases_pl['new_cases']
    test_size = 0.20
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=7)
    model = XGBClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    x_pred_on_train = model.predict(X_train)
    predictions = [round(value) for value in y_pred]

    accuracy_train = [np.abs(x_pred_on_train[i] - y_train[i]) / y_train[i] for i in range(0, len(predictions))]
    accuracy_train = [v for v in accuracy_train if not np.isnan(v) and not np.isinf(v)]
    error_train = np.mean(accuracy_train) * 100

    accuracy_test = [np.abs(predictions[i] - y_test[i])/y_test[i] for i in range(0, len(predictions))]
    accuracy_test =  [v for v in accuracy_test if not np.isnan(v) and not np.isinf(v)]
    error_test = np.mean(accuracy_test)*100

    # x_to_prediction = X_test.iloc[[-1]]
    new_row = {'shift1': new_cases_pl["new_cases"][-1],
               'shift2': new_cases_pl['new_cases'][-2],
               'shift3': new_cases_pl['new_cases'][-3],
               'shift4': new_cases_pl['new_cases'][-4],
               'shift5': new_cases_pl['new_cases'][-5],
               'shift6': new_cases_pl['new_cases'][-6],
               'shift7': new_cases_pl['new_cases'][-7]}
    x_to_prediction = pd.DataFrame(new_row, index=[0])
    y_to_prediction = [model.predict(x_to_prediction)[0]]
    pred_days = [Y.index[-1] + datetime.timedelta(days=1)]
    for i in range(1, 7):
        new_row = {'shift1': y_to_prediction[i-1],
                   'shift2': x_to_prediction['shift1'][i-1],
                   'shift3': x_to_prediction['shift2'][i-1],
                   'shift4': x_to_prediction['shift3'][i-1],
                   'shift5': x_to_prediction['shift4'][i-1],
                   'shift6': x_to_prediction['shift5'][i-1],
                   'shift7': x_to_prediction['shift6'][i-1]}
        pred_days.append(pred_days[i-1] + datetime.timedelta(days=1))
        x_to_prediction = x_to_prediction.append(new_row, ignore_index=True)
        y_to_prediction.append(model.predict(x_to_prediction.iloc[[i]])[0])
    pred_df = pd.DataFrame({"prediction": y_to_prediction}, index=pred_days)
    return pred_df, error_train, error_test


def new_deaths_xgboost():
    new_cases_pl = COVID_DATA[COVID_DATA['location'] == 'Poland']
    new_cases_pl['new_deaths'] = new_cases_pl['new_deaths'].fillna(0)
    new_cases_pl['shift1'] = new_cases_pl['new_deaths'].shift(1)
    new_cases_pl['shift2'] = new_cases_pl['shift1'].shift(1)
    new_cases_pl['shift3'] = new_cases_pl['shift2'].shift(1)
    new_cases_pl['shift4'] = new_cases_pl['shift3'].shift(1)
    new_cases_pl['shift5'] = new_cases_pl['shift4'].shift(1)
    new_cases_pl['shift6'] = new_cases_pl['shift5'].shift(1)
    new_cases_pl['shift7'] = new_cases_pl['shift6'].shift(1)

    X = new_cases_pl[['shift1', 'shift2', 'shift3', 'shift4', 'shift5', 'shift6', 'shift7']]
    Y = new_cases_pl['new_deaths']
    test_size = 0.20
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=7)
    model = XGBClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    x_pred_on_train = model.predict(X_train)
    predictions = [round(value) for value in y_pred]

    accuracy_train = [np.abs(x_pred_on_train[i] - y_train[i]) / y_train[i] for i in range(0, len(predictions))]
    accuracy_train = [v for v in accuracy_train if not np.isnan(v) and not np.isinf(v)]
    error_train = np.mean(accuracy_train) * 100

    accuracy_test = [np.abs(predictions[i] - y_test[i])/y_test[i] for i in range(0, len(predictions))]
    accuracy_test =  [v for v in accuracy_test if not np.isnan(v) and not np.isinf(v)]
    error_test = np.mean(accuracy_test)*100

    # x_to_prediction = X_test.iloc[[-1]]
    new_row = {'shift1': new_cases_pl["new_deaths"][-1],
               'shift2': new_cases_pl['new_deaths'][-2],
               'shift3': new_cases_pl['new_deaths'][-3],
               'shift4': new_cases_pl['new_deaths'][-4],
               'shift5': new_cases_pl['new_deaths'][-5],
               'shift6': new_cases_pl['new_deaths'][-6],
               'shift7': new_cases_pl['new_deaths'][-7]}
    x_to_prediction = pd.DataFrame(new_row, index=[0])
    y_to_prediction = [model.predict(x_to_prediction)[0]]
    pred_days = [Y.index[-1] + datetime.timedelta(days=1)]
    for i in range(1, 7):
        new_row = {'shift1': y_to_prediction[i-1],
                   'shift2': x_to_prediction['shift1'][i-1],
                   'shift3': x_to_prediction['shift2'][i-1],
                   'shift4': x_to_prediction['shift3'][i-1],
                   'shift5': x_to_prediction['shift4'][i-1],
                   'shift6': x_to_prediction['shift5'][i-1],
                   'shift7': x_to_prediction['shift6'][i-1]}
        pred_days.append(pred_days[i-1] + datetime.timedelta(days=1))
        x_to_prediction = x_to_prediction.append(new_row, ignore_index=True)
        y_to_prediction.append(model.predict(x_to_prediction.iloc[[i]])[0])
    pred_df = pd.DataFrame({"prediction": y_to_prediction}, index=pred_days)
    return pred_df, error_train, error_test
