import pickle
from datetime import timedelta, datetime
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta

from DEApp.currency_code import get_currencies
from DEApp.data_loader import COVID_DATA, GROUPS, CURRENCY_CODES
from DEApp.lstm_utils import prepare_lstm_pred_new_cases, prepare_lstm_pred_new_deaths
from DEApp.shares_code import get_stock_data

LSTM_NEW_CASES_TEST_MAE = 1794.41
LSTM_NEW_DEATHS_TEST_MAE = 95.53

dict_with_measures_to_advanced_analysis = \
    {1: ["new_vaccinations", "new_cases", "New vaccinations vs new cases"],
     2: ["new_deaths", "new_cases", "New deaths vs new cases"],
     3: ["new_tests", "new_cases", "New tests vs new cases"],
     4: ["new_vaccinations", "new_deaths", "New vaccinations vs new deaths"],
     5: ["people_fully_vaccinated_per_hundred", "new_cases_per_million", "Fully vaccinated vs new cases"]
     # 6: []
     }


def prepare_covid_data(time):
    data = COVID_DATA
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


def draw_covid1(country, measure, time):
    """
    Prepare basic plot.
    :param country: str country name
    :param measure: str measurement to be plotted
    :param time: str time range for plot
    """
    data, time_prior = prepare_covid_data(time)
    data = data[data['location'] == country]
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.set_xlim([time_prior, datetime.now()])
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    ax.plot(data[measure].rolling(window='3d').mean(), 'y-', linewidth=8, label='running average (3 days)')
    ax.plot(data[measure].rolling(window='7d').mean(), 'g-', linewidth=8, label='running average (week)')
    ax.plot(data[measure], 'r.-', markersize=25, linewidth=3)
    ax.grid()
    plt.xticks(rotation=30,)
    plt.title(measure + " in " + country, fontdict={'fontsize': 40, 'color': "white"})
    plt.legend()
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_fontsize(30)
    ax.yaxis.label.set_fontsize(30)
    ax.tick_params(colors='white')
    fig.savefig('static/images/covid1.png', dpi=300, bbox_inches='tight', transparent=True)
    fig.clf()


def draw_covid2(country1, country2, measure, time):
    """
    Prepare basic plot.
    :param country1: str country name
    :param country2: str another country name
    :param measure: str measurement to be plotted
    :param time: str time range for plot
    """
    data, time_prior = prepare_covid_data(time)
    data1 = data[data['location'] == country1]
    data2 = data[data['location'] == country2]
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.set_xlim([time_prior, datetime.now()])
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    ax.plot(data1[measure], 'r.-', markersize=25, label=country1, linewidth=3)
    ax.plot(data2[measure], 'y.-', markersize=25, label=country2, linewidth=3)
    ax.grid()
    plt.xticks(rotation=30,)
    plt.title(measure + " in " + country1 + " and " + country2, fontdict={'fontsize': 40, 'color': "white"})
    plt.legend()
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_fontsize(30)
    ax.yaxis.label.set_fontsize(30)
    ax.tick_params(colors='white')
    fig.savefig('static/images/covid2.png', dpi=300, bbox_inches='tight', transparent=True)
    fig.clf()


def get_rank(measure):
    """
    Prepare basic plot.
    :param measure: str measurement to be plotted
    """
    data = COVID_DATA
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


def draw_covid_shares(long_name, measure, time):
    """
    Prepare basic plot.
    """
    data, time_prior = prepare_covid_data(time)
    data_share = get_stock_data(long_name, int(time_prior.year), int(time_prior.month), int(time_prior.day))
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
    ax2.plot(data[measure], 'g-', label=measure + ' in the whole world', linewidth=3)
    ax.plot(data_share["Close"], 'r.-', markersize=25, label=long_name, linewidth=8)
    plt.xticks(rotation=30, )
    ax2.xaxis.label.set_color('white')
    ax2.yaxis.label.set_color('white')
    ax2.tick_params(colors='white')
    plt.legend()
    fig.savefig('static/images/covid5.png', dpi=300, bbox_inches='tight', transparent=True)
    fig.clf()


def draw_covid_currency(country1, country2, measure, time):
    """
    Prepare basic plot.
    :param country1: str country name
    :param country2: str another country name
    :param measure: str measurement to be plotted
    :param time: str time range for plot
    """
    data, time_prior = prepare_covid_data(time)
    data1 = data[data['location'] == country1]
    data2 = data[data['location'] == country2]
    plt.rcParams.update({'font.size': 20})
    try:
        currency = get_currencies(country1, country2, str(time_prior)[:10], str(datetime.now())[:10])
    except:
        currency = None

    fig, ax = plt.subplots(figsize=(20, 8))
    ax.set_xlim([time_prior, datetime.now()])
    if currency is not None:
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        ax.plot(data1[measure], 'r.-', markersize=25, label=country1, linewidth=3)
        ax.plot(data2[measure], 'y.-', markersize=25, label=country2, linewidth=3)
        ax2 = ax.twinx()
        ax2.plot(currency, 'go-', markersize=25, label=CURRENCY_CODES[country1] + ' to ' +\
                 CURRENCY_CODES[country2], linewidth=8)
        ax.grid()
        plt.xticks(rotation=30, )
        plt.title(measure + " and currencies " + " in " + country1 + " and " + country2, fontdict={'fontsize': 40, 'color': "white"})
        plt.legend()
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


def prepare_measures(df, one_column, second_column, scale_nominator=1.0, scale_denominator=1.0):
    """
        Calculate new column one_column_vs_second_column.
        :param df: df with data
        :param one_column: name of first column (nominator)
        :param second_column: name of second column (denominator)
        """
    new_column_name = one_column + "_vs_" + second_column
    df[new_column_name] = (df[one_column]*scale_nominator)/(df[second_column]*scale_denominator)
    return df, new_column_name


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


def draw_measures_advanced_analysis(country, time):
    """
        Prepare basic plot.
        :param country: str country name
        :param time: str time range for plot
        """
    data, time_prior = prepare_covid_data(time)
    data = data[data['location'] == country]\

    data, column_name1 = prepare_measures(data, dict_with_measures_to_advanced_analysis[1][0], dict_with_measures_to_advanced_analysis[1][1])
    data, column_name2 = prepare_measures(data, dict_with_measures_to_advanced_analysis[2][0], dict_with_measures_to_advanced_analysis[2][1])
    data, column_name3 = prepare_measures(data, dict_with_measures_to_advanced_analysis[3][0],
                                          dict_with_measures_to_advanced_analysis[3][1])
    data, column_name4 = prepare_measures(data, dict_with_measures_to_advanced_analysis[4][0],
                                          dict_with_measures_to_advanced_analysis[4][1])
    data, column_name5 = prepare_measures(data, dict_with_measures_to_advanced_analysis[5][0],
                                          dict_with_measures_to_advanced_analysis[5][1], scale_nominator=10000)

    plot_one_measures_in_advanced_analysis(data[column_name1], time_prior,
                                           dict_with_measures_to_advanced_analysis[1][2], country, 'static/images/covid5.png')
    plot_one_measures_in_advanced_analysis(data[column_name2], time_prior,
                                           dict_with_measures_to_advanced_analysis[2][2], country,
                                           'static/images/covid6.png')
    plot_one_measures_in_advanced_analysis(data[column_name3], time_prior,
                                           dict_with_measures_to_advanced_analysis[3][2], country,
                                           'static/images/covid7.png')
    plot_one_measures_in_advanced_analysis(data[column_name4], time_prior,
                                           dict_with_measures_to_advanced_analysis[4][2], country,
                                           'static/images/covid8.png')

    plot_one_measures_in_advanced_analysis(data[column_name4], time_prior,
                                           dict_with_measures_to_advanced_analysis[5][2], country,
                                           'static/images/covid9.png')


def draw_predictions_lstm_new_cases():
    new_cases_pl, pred = prepare_lstm_pred_new_cases()
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    ax.plot(new_cases_pl, 'b.-', markersize=25, linewidth=3, label='historical new cases')
    ax.plot(pred, 'r.-', markersize=25, linewidth=3, label='predicted new cases')
    ax.grid()
    plt.xticks(rotation=30, )
    plt.title("LSTM prediction for new cases in Poland",
              fontdict={'fontsize': 40, 'color': "white"})
    plt.legend()
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_fontsize(30)
    ax.yaxis.label.set_fontsize(30)
    ax.tick_params(colors='white')
    fig.savefig('static/images/pred1.png', dpi=300, bbox_inches='tight', transparent=True)
    fig.clf()


def draw_predictions_lstm_new_deaths():
    new_cases_pl, pred = prepare_lstm_pred_new_deaths()
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    ax.plot(new_cases_pl, 'b.-', markersize=25, linewidth=3, label='historical new deaths')
    ax.plot(pred, 'r.-', markersize=25, linewidth=3, label='predicted new deaths')
    ax.grid()
    plt.xticks(rotation=30, )
    plt.title("LSTM prediction for new deaths in Poland",
              fontdict={'fontsize': 40, 'color': "white"})
    plt.legend()
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_fontsize(30)
    ax.yaxis.label.set_fontsize(30)
    ax.tick_params(colors='white')
    fig.savefig('static/images/pred2.png', dpi=300, bbox_inches='tight', transparent=True)
    fig.clf()
