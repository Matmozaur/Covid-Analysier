import dask
from datetime import timedelta, datetime
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta

from DEApp.currency_code import get_currencies
from DEApp.data_loader import GROUPS
from DEApp.moving_average import new_cases_xgboost, new_deaths_xgboost
from DEApp.shares_code import get_stock_data

dict_with_measures_to_advanced_analysis = \
    {1: ["new_vaccinations", "new_cases", "New vaccinations vs new cases"],
     2: ["new_deaths", "new_cases", "New deaths vs new cases"],
     3: ["new_tests", "new_cases", "New tests vs new cases"],
     4: ["new_vaccinations", "new_deaths", "New vaccinations vs new deaths"],
     5: ["people_fully_vaccinated_per_hundred", "new_cases_per_million", "Fully vaccinated vs new cases"]
     # 6: []
     }


def prepare_covid_data(data, time):
    today = datetime.now()
    time_prior = today - timedelta(days=1)
    if time == 'week':
        time_prior = today - timedelta(weeks=1)
    elif time == 'month':
        time_prior = today - relativedelta(months=1)
    elif time == '3 months':
        time_prior = today - relativedelta(months=3)
    elif time == '6 months':
        time_prior = today - relativedelta(months=6)
    elif time == 'all':
        time_prior = data.index[0]
    data = data[data.index >= time_prior]
    return data, time_prior


@dask.delayed
def draw_covid1(data, country, measure, time):
    """
    Prepare basic plot.
    :param country: str country name
    :param measure: str measurement to be plotted
    :param time: str time range for plot
    """
    data, time_prior = prepare_covid_data(data, time)
    data = data[data['location'] == country]
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.set_xlim([time_prior, datetime.now()])
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    ax.plot(data[measure], 'r.-', markersize=25, linewidth=3, label=measure + ' in ' + country)
    ax.plot(data[measure].rolling(window='3d').mean(), 'y-', linewidth=8, label='running average (3 days)')
    ax.plot(data[measure].rolling(window='7d').mean(), 'g-', linewidth=8, label='running average (week)')
    ax.grid()
    plt.xticks(rotation=30, )
    plt.title(measure + " in " + country, fontdict={'fontsize': 40, 'color': "white"})
    plt.legend()
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_fontsize(30)
    ax.yaxis.label.set_fontsize(30)
    ax.tick_params(colors='white')
    fig.savefig('static/images/covid1.png', dpi=300, bbox_inches='tight', transparent=True)
    fig.clf()
    plt.close()


@dask.delayed
def draw_covid2(data, country1, country2, measure, time):
    """
    Prepare basic plot.
    :param country1: str country name
    :param country2: str another country name
    :param measure: str measurement to be plotted
    :param time: str time range for plot
    """
    data, time_prior = prepare_covid_data(data, time)
    data1 = data[data['location'] == country1]
    data2 = data[data['location'] == country2]
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.set_xlim([time_prior, datetime.now()])
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    ax.plot(data1[measure], 'r.-', markersize=25, label=country1, linewidth=3)
    ax.plot(data2[measure], 'y.-', markersize=25, label=country2, linewidth=3)
    ax.grid()
    plt.xticks(rotation=30, )
    plt.title(measure + " in " + country1 + " and " + country2, fontdict={'fontsize': 40, 'color': "white"})
    plt.legend()
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_fontsize(30)
    ax.yaxis.label.set_fontsize(30)
    ax.tick_params(colors='white')
    fig.savefig('static/images/covid2.png', dpi=300, bbox_inches='tight', transparent=True)
    fig.clf()
    plt.close()


@dask.delayed
def get_rank(data, measure):
    """
    Prepare basic plot.
    :param measure: str measurement to be plotted
    """
    time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    time = time - timedelta(hours=24)
    data = data[data.index == time]
    data = data[~data.location.isin(GROUPS)]
    ranking = data.sort_values(measure, ascending=False)[['location', measure]]
    ranking.set_index('location', inplace=True)
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.bar(ranking[:5].index, ranking[:5][measure].values)
    plt.xticks(rotation=30, )
    plt.title("Top 5 " + measure, fontdict={'fontsize': 40, 'color': "white"})
    # plt.legend()
    ax.set_ylabel(measure)
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_fontsize(30)
    ax.yaxis.label.set_fontsize(30)
    ax.tick_params(colors='white')
    fig.savefig('static/images/covid3.png', dpi=300, bbox_inches='tight', transparent=True)
    fig.clf()
    plt.close()


@dask.delayed
def draw_covid_shares(data, shares, long_name, measure, time):
    """
    Prepare basic plot.
    """
    data, time_prior = prepare_covid_data(data, time)
    data_share = get_stock_data(shares, long_name, int(time_prior.year), int(time_prior.month), int(time_prior.day))
    data = data[data['location'] == 'World']
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.set_xlim([time_prior, datetime.now()])
    plt.title(long_name + " price", fontdict={'fontsize': 40, 'color': "white"})
    ax.grid()
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_fontsize(30)
    ax.yaxis.label.set_fontsize(30)
    ax.tick_params(colors='white')
    ax2 = ax.twinx()
    lns2 = ax2.plot(data[measure], 'g-', label=measure + ' in the whole world', linewidth=3)
    lns1 = ax.plot(data_share["Close"], 'r.-', markersize=25, label=long_name, linewidth=8)
    plt.xticks(rotation=30, )
    ax2.xaxis.label.set_color('white')
    ax2.yaxis.label.set_color('white')
    ax2.tick_params(colors='white')
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)
    fig.savefig('static/images/covid5.png', dpi=300, bbox_inches='tight', transparent=True)
    fig.clf()
    plt.close()


@dask.delayed
def draw_covid_currency(data, currency_codes, country1, country2, measure, time):
    """
    Prepare basic plot.
    :param country1: str country name
    :param country2: str another country name
    :param measure: str measurement to be plotted
    :param time: str time range for plot
    """
    data, time_prior = prepare_covid_data(data, time)
    data1 = data[data['location'] == country1]
    data2 = data[data['location'] == country2]
    plt.rcParams.update({'font.size': 20})
    try:
        currency = get_currencies(currency_codes, country1, country2, str(time_prior)[:10], str(datetime.now())[:10])
    except:
        currency = None

    fig, ax = plt.subplots(figsize=(20, 8))
    ax.set_xlim([time_prior, datetime.now()])
    if currency is not None:
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        lns1 = ax.plot(data1[measure], 'r.-', markersize=25, label=country1, linewidth=3)
        lns2 = ax.plot(data2[measure], 'y.-', markersize=25, label=country2, linewidth=3)
        ax2 = ax.twinx()
        lns3 = ax2.plot(currency, 'go-', markersize=25, label=currency_codes[country1] + ' to ' + \
                                                              currency_codes[country2], linewidth=8)
        ax.grid()
        plt.xticks(rotation=30, )
        plt.title(measure + " and currencies " + " in " + country1 + " and " + country2,
                  fontdict={'fontsize': 40, 'color': "white"})
        lns = lns1 + lns2 + lns3
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc=0)
        ax2.xaxis.label.set_color('white')
        ax2.yaxis.label.set_color('white')
        ax2.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_fontsize(30)
    ax.yaxis.label.set_fontsize(30)
    ax.tick_params(colors='white')
    fig.savefig('static/images/covid4.png', dpi=300, bbox_inches='tight', transparent=True)
    fig.clf()
    plt.close()


def prepare_measures(df, one_column, second_column, scale_nominator=1.0, scale_denominator=1.0):
    """
        Calculate new column one_column_vs_second_column.
    """
    new_column_name = one_column + "_vs_" + second_column
    df[new_column_name] = (df[one_column] * scale_nominator) / (df[second_column] * scale_denominator)
    return df, new_column_name


@dask.delayed
def plot_one_measures_in_advanced_analysis(data_to_plot, time_prior, title, country, path_to_save):
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.set_xlim([time_prior, datetime.now()])
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    ax.plot(data_to_plot, 'r.-', markersize=25, linewidth=3)
    ax.grid()
    plt.xticks(rotation=30, )
    plt.title(title + " in " + country, fontdict={'fontsize': 40, 'color': "white"})
    # plt.legend()
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_fontsize(30)
    ax.yaxis.label.set_fontsize(30)
    ax.tick_params(colors='white')
    fig.savefig(path_to_save, dpi=300, bbox_inches='tight', transparent=True)
    fig.clf()
    plt.close()


@dask.delayed
def draw_measures_advanced_analysis(n, data, country, time):
    data1, time_prior = prepare_covid_data(data, time)
    data1 = data1[data1['location'] == country]
    data1, column_name1 = prepare_measures(data1, dict_with_measures_to_advanced_analysis[n][0],
                                           dict_with_measures_to_advanced_analysis[n][1])
    plot_one_measures_in_advanced_analysis(data1[column_name1], time_prior,
                                           dict_with_measures_to_advanced_analysis[n][2], country,
                                           'static/images/covid{}.png'.format(n + 4))


def new_cases(covid_data, no_of_days=45):
    today = datetime.now()
    time_prior = today - timedelta(days=no_of_days)
    new_cases_pl = covid_data[covid_data['location'] == 'Poland']
    new_cases_pl = new_cases_pl[time_prior:]
    return new_cases_pl


def new_deaths(covid_data, no_of_days=45):
    today = datetime.now()
    time_prior = today - timedelta(days=no_of_days)
    new_deaths_pl = covid_data[covid_data['location'] == 'Poland']
    new_deaths_pl = new_deaths_pl[time_prior:]
    return new_deaths_pl


@dask.delayed
def draw_xgboost_new_cases(covid_data):
    new_cases_pl = new_cases(covid_data, no_of_days=120)
    pred_df1 = new_cases_xgboost(new_cases_pl)
    title_on_xgboost_pred = "Predicting new cases with xgboost"
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    ax.plot(new_cases_pl["new_cases"], 'r.-', markersize=25, linewidth=3, label='Historical new cases')
    ax.plot(pred_df1["prediction"], 'y.-', markersize=25, linewidth=3, label='Prediction (xgboost model)')
    ax.grid()
    plt.xticks(rotation=30, )
    plt.title(title_on_xgboost_pred, fontdict={'fontsize': 25, 'color': "white"})
    plt.legend()
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_fontsize(30)
    ax.yaxis.label.set_fontsize(30)
    ax.tick_params(colors='white')
    fig.savefig('static/images/pred3.png', dpi=300, bbox_inches='tight', transparent=True)
    fig.clf()
    plt.close()


@dask.delayed
def draw_xgboost_new_deaths(covid_data):
    new_deaths_pl = new_deaths(covid_data, no_of_days=120)
    pred_df2 = new_deaths_xgboost(new_deaths_pl)
    title_on_xgboost_pred = "Predicting new deaths with xgboost"
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    ax.plot(new_deaths_pl["new_deaths"], 'r.-', markersize=25, linewidth=3, label='Historical new deaths')
    ax.plot(pred_df2["prediction"], 'y.-', markersize=25, linewidth=3, label='Prediction (xgboost model)')
    ax.grid()
    plt.xticks(rotation=30, )
    plt.title(title_on_xgboost_pred, fontdict={'fontsize': 25, 'color': "white"})
    plt.legend()
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_fontsize(30)
    ax.yaxis.label.set_fontsize(30)
    ax.tick_params(colors='white')
    fig.savefig('static/images/pred4.png', dpi=300, bbox_inches='tight', transparent=True)
    fig.clf()
    plt.close()
