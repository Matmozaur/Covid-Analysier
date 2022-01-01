import pickle
from datetime import datetime
import pandas as pd
import datetime


def new_cases_xgboost(covid_data):
    new_cases_pl = covid_data[covid_data['location'] == 'Poland']
    new_cases_pl = new_cases_pl[['new_cases']]
    new_cases_pl['shift1'] = new_cases_pl['new_cases'].shift(1)
    new_cases_pl['shift2'] = new_cases_pl['shift1'].shift(1)
    new_cases_pl['shift3'] = new_cases_pl['shift2'].shift(1)
    new_cases_pl['shift4'] = new_cases_pl['shift3'].shift(1)
    new_cases_pl['shift5'] = new_cases_pl['shift4'].shift(1)
    new_cases_pl['shift6'] = new_cases_pl['shift5'].shift(1)
    new_cases_pl['shift7'] = new_cases_pl['shift6'].shift(1)
    new_cases_pl = new_cases_pl.dropna()

    X = new_cases_pl[['shift1', 'shift2', 'shift3', 'shift4', 'shift5', 'shift6', 'shift7']]
    Y = new_cases_pl['new_cases']
    with open('data/xgb_cases.pickle', 'rb') as handle:
        model = pickle.load(handle)

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
    return pred_df


def new_deaths_xgboost(covid_data):
    new_cases_pl = covid_data[covid_data['location'] == 'Poland']
    new_cases_pl = new_cases_pl[['new_deaths']]
    new_cases_pl['new_deaths'] = new_cases_pl['new_deaths'].fillna(0)
    new_cases_pl['shift1'] = new_cases_pl['new_deaths'].shift(1)
    new_cases_pl['shift2'] = new_cases_pl['shift1'].shift(1)
    new_cases_pl['shift3'] = new_cases_pl['shift2'].shift(1)
    new_cases_pl['shift4'] = new_cases_pl['shift3'].shift(1)
    new_cases_pl['shift5'] = new_cases_pl['shift4'].shift(1)
    new_cases_pl['shift6'] = new_cases_pl['shift5'].shift(1)
    new_cases_pl['shift7'] = new_cases_pl['shift6'].shift(1)
    new_cases_pl = new_cases_pl.dropna()

    X = new_cases_pl[['shift1', 'shift2', 'shift3', 'shift4', 'shift5', 'shift6', 'shift7']]
    Y = new_cases_pl['new_deaths']
    with open('data/xgb_deaths.pickle', 'rb') as handle:
        model2 = pickle.load(handle)

    # x_to_prediction = X_test.iloc[[-1]]
    new_row = {'shift1': new_cases_pl["new_deaths"][-1],
               'shift2': new_cases_pl['new_deaths'][-2],
               'shift3': new_cases_pl['new_deaths'][-3],
               'shift4': new_cases_pl['new_deaths'][-4],
               'shift5': new_cases_pl['new_deaths'][-5],
               'shift6': new_cases_pl['new_deaths'][-6],
               'shift7': new_cases_pl['new_deaths'][-7]}
    x_to_prediction = pd.DataFrame(new_row, index=[0])
    y_to_prediction = [model2.predict(x_to_prediction)[0]]
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
        y_to_prediction.append(model2.predict(x_to_prediction.iloc[[i]])[0])
    pred_df = pd.DataFrame({"prediction": y_to_prediction}, index=pred_days)
    return pred_df
