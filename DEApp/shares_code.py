import yfinance as yf

from DEApp.data_loader import SHARES


def find_ticker_list(df_names, str_name):
    return df_names[df_names["Name"].str.contains(str_name)]["Ticker"].tolist()


def is_ticker_possible(df_names, str_name):
    ticker_list = df_names[df_names["Name"].str.contains(str_name)]["Ticker"].tolist()
    if len(ticker_list) > 0:
        return True
    else:
        return False


def check_date(value, min_val, max_val):
    value = str(value)
    if value.isdecimal():
        value = int(value)
        if min_val <= value <= max_val:
            return True
        else:
            return False
    else:
        return False


def return_valid_date(value):
    value = str(value)
    if len(value) == 1:
        value = '0' + value
        return value
    else:
        return value


def get_stock_data(shares, long_name, year, month, day):
    if is_ticker_possible(shares, long_name):
        name = find_ticker_list(shares, long_name)[0]
        if check_date(year, 2000, 2021) and check_date(month, 1, 12) and check_date(day, 1, 31):
            year, month, day = str(year), return_valid_date(month), return_valid_date(day)
            string_data = "{0}-{1}-{2}".format(str(year), str(month), str(day))
            df = yf.download(name, start=string_data)
            return df