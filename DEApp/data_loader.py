import pandas as pd
from dask.distributed import Client
import dask.dataframe

COLUMNS_COVID_USABLE = ['total_cases', 'new_cases', 'total_deaths', 'iso_code',
                        'new_deaths', 'total_cases_per_million', 'new_cases_per_million', 'total_deaths_per_million',
                        'new_deaths_per_million', 'weekly_hosp_admissions', 'weekly_hosp_admissions_per_million',
                        'new_tests', 'total_tests', 'total_tests_per_thousand', 'new_tests_per_thousand',
                        'positive_rate',
                        'tests_per_case', 'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated',
                        'new_vaccinations', 'total_vaccinations_per_hundred', 'people_vaccinated_per_hundred',
                        'people_fully_vaccinated_per_hundred']
COLUMNS_COVID = COLUMNS_COVID_USABLE + ['location']
TIMES_COVID = ['week', 'month', '3 months', '6 months', 'all']
COUNTRIES = []
GROUPS = ['World', 'Europe', 'Asia', 'North America', 'European Union', 'South America', 'Africa']
COVID_DATA = None
CURRENCY_CODES = None
SHARES = None
SHARES_NAMES = ["Apple", "Microsoft"]
CLIENT = None


# Shares
def read_shares():
    df_names = pd.read_excel('data/yahoo_data.xlsx')
    global SHARES
    SHARES = df_names.dropna()


# Covid
def download_whole_covid_data():
    data = pd.read_csv('https://covid.ourworldindata.org/data/owid-covid-data.csv', index_col='date', parse_dates=True)
    data = data[COLUMNS_COVID]
    global COUNTRIES
    COUNTRIES = sorted(list(set(data['location'])))
    global COVID_DATA
    COVID_DATA = dask.dataframe.from_pandas(data, 10)
    # COVID_DATA = data


# Currency
def load_currency_code():
    currs = pd.read_excel('data/currency_code.xls', skiprows=3)
    currs['ENTITY'] = currs['ENTITY'].apply(lambda x: x.title())
    currs = currs[['ENTITY', 'Alphabetic Code']]
    currs.set_index('ENTITY', inplace=True)
    global CURRENCY_CODES
    CURRENCY_CODES = currs['Alphabetic Code']


def download_whole_data():
    client = Client()
    client.cluster.scale(10)
    global CLIENT
    CLIENT = client
    download_whole_covid_data()
    load_currency_code()
    read_shares()
