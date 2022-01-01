import requests
import pandas as pd
from datetime import timedelta, datetime

from DEApp.data_loader import CURRENCY_CODES

URL = 'http://api.nbp.pl/api/exchangerates/rates/'


def download_data(start_date, end_date, currency, table='a'):
    url = URL + table + '/' + currency + '/' + start_date + '/' + end_date
    params = {'format': 'JSON'}
    r = requests.get(url=url, params=params)
    data = r.json()
    data = [[d['effectiveDate'], d['mid']] for d in data['rates']]
    data = pd.DataFrame(data, columns=['time', 'price'])
    data.set_index('time', inplace=True)
    data.index = pd.to_datetime(data.index)
    return data


def get_currencies_from_code(currency1, currency2, start_date, end_date):
    if currency1 == 'PLN':
        data1 = download_data(start_date, end_date, 'USD')
        data1['price'] = 1
    else:
        data1 = download_data(start_date, end_date, currency1)
    if currency2 == 'PLN':
        data2 = download_data(start_date, end_date, 'USD')
        data2['price'] = 1
    else:
        data2 = download_data(start_date, end_date, currency2)
    result = data1['price']/data2['price']
    return result


def get_currencies(currency_codes, country1, country2, start_date, end_date):
    currency1 = currency_codes[country1]
    currency2 = currency_codes[country2]
    return get_currencies_from_code(currency1, currency2, start_date, end_date)
