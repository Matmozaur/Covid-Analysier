from django.shortcuts import render

from DEApp.data_loader import COLUMNS_COVID_USABLE, TIMES_COVID, COUNTRIES, SHARES_NAMES
from DEApp.covid_draw import draw_covid1, draw_covid2, get_rank, draw_covid_currency, draw_covid_shares, \
    draw_measures_advanced_analysis, draw_predictions_lstm_new_deaths, \
    draw_predictions_lstm_new_cases, draw_xgboost_new_cases, draw_xgboost_new_deaths

context = dict()


def index(request):
    context['countries'] = COUNTRIES
    context['measurements'] = COLUMNS_COVID_USABLE
    context['times'] = TIMES_COVID
    context['shares'] = SHARES_NAMES
    context['selected_country'] = 'Poland'
    context['selected_another_country'] = 'Germany'
    context['selected_measurement'] = 'new_cases'
    context['selected_time'] = 'month'
    context['selected_share'] = 'Apple'
    draw_covid1(context['selected_country'], context['selected_measurement'], context['selected_time'])
    draw_covid2(context['selected_country'], context['selected_another_country'], context['selected_measurement'],
                context['selected_time'])
    get_rank(context['selected_measurement'])
    draw_covid_currency(context['selected_country'], context['selected_another_country'],
                        context['selected_measurement'],
                        context['selected_time'])
    draw_covid_shares(long_name=context['selected_share'], measure=context['selected_measurement'], time=context['selected_time'])
    return render(request, 'DEApp/DEApp.html', context)


def covid_plot(request):
    if request.method == 'POST':
        context['selected_country'] = request.POST.get("country")
        context['selected_another_country'] = request.POST.get("another_country")
        context['selected_measurement'] = request.POST.get("measurement")
        context['selected_time'] = request.POST.get("time")
        context['selected_share'] = request.POST.get("shares")
    draw_covid1(context['selected_country'], context['selected_measurement'], context['selected_time'])
    draw_covid2(context['selected_country'], context['selected_another_country'], context['selected_measurement'],
                context['selected_time'])
    get_rank(context['selected_measurement'])
    draw_covid_currency(context['selected_country'], context['selected_another_country'],
                        context['selected_measurement'],
                        context['selected_time'])
    draw_covid_shares(long_name=context['selected_share'], measure=context['selected_measurement'], time=context['selected_time'])
    return render(request, 'DEApp/DEApp.html', context)


def predictions(request):
    draw_predictions_lstm_new_deaths()
    draw_predictions_lstm_new_cases()
    draw_xgboost_new_cases()
    draw_xgboost_new_deaths()
    return render(request, 'DEApp/Predictions.html', context)


def advanced_analysis(request):
    context['countries'] = COUNTRIES
    context['times'] = TIMES_COVID
    if request.method == 'POST':
        context['selected_country'] = request.POST.get("country")
        context['selected_time'] = request.POST.get("time")
    else:
        context['selected_country'] = 'Poland'
        context['selected_time'] = 'month'
    draw_measures_advanced_analysis(context['selected_country'], context['selected_time'])
    # get_rank(context['selected_measurement'])
    return render(request, 'DEApp/AdvancedAnalysis.html', context)
