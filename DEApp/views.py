from django.shortcuts import render
from DEApp.data_loader import COLUMNS_COVID_USABLE, TIMES_COVID, COUNTRIES, SHARES_NAMES
from DEApp.covid_draw import draw_covid1, draw_covid2, get_rank, draw_covid_currency, draw_covid_shares, \
    draw_measures_advanced_analysis, draw_xgboost_new_cases, draw_xgboost_new_deaths
import dask
from DEApp.data_loader import COVID_DATA, SHARES, CURRENCY_CODES

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
    tasks = [draw_covid1(COVID_DATA, context['selected_country'], context['selected_measurement'],
                         context['selected_time']),
             draw_covid2(COVID_DATA, context['selected_country'], context['selected_another_country'],
                         context['selected_measurement'],
                         context['selected_time']),
             get_rank(COVID_DATA, context['selected_measurement']),
             draw_covid_currency(COVID_DATA, CURRENCY_CODES, context['selected_country'],
                                 context['selected_another_country'],
                                 context['selected_measurement'],
                                 context['selected_time']),
             draw_covid_shares(COVID_DATA, SHARES, long_name=context['selected_share'],
                               measure=context['selected_measurement'],
                               time=context['selected_time'])]

    dask.compute(tasks, scheduler='distributed')
    return render(request, 'DEApp/DEApp.html', context)


def covid_plot(request):
    if request.method == 'POST':
        context['selected_country'] = request.POST.get("country")
        context['selected_another_country'] = request.POST.get("another_country")
        context['selected_measurement'] = request.POST.get("measurement")
        context['selected_time'] = request.POST.get("time")
        context['selected_share'] = request.POST.get("shares")

    tasks = [draw_covid1(COVID_DATA, context['selected_country'], context['selected_measurement'],
                         context['selected_time']),
             draw_covid2(COVID_DATA, context['selected_country'], context['selected_another_country'],
                         context['selected_measurement'],
                         context['selected_time']),
             get_rank(COVID_DATA, context['selected_measurement']),
             draw_covid_currency(COVID_DATA, CURRENCY_CODES, context['selected_country'],
                                 context['selected_another_country'],
                                 context['selected_measurement'],
                                 context['selected_time']),
             draw_covid_shares(COVID_DATA, SHARES, long_name=context['selected_share'],
                               measure=context['selected_measurement'],
                               time=context['selected_time'])]

    dask.compute(tasks, scheduler='distributed')
    return render(request, 'DEApp/DEApp.html', context)


def predictions(request):
    tasks = [
        draw_xgboost_new_cases(COVID_DATA),
        draw_xgboost_new_deaths(COVID_DATA)
        ]
    dask.compute(tasks, scheduler='distributed')
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
    tasks = [
        draw_measures_advanced_analysis(1, COVID_DATA, context['selected_country'], context['selected_time']),
        draw_measures_advanced_analysis(2, COVID_DATA, context['selected_country'], context['selected_time']),
        draw_measures_advanced_analysis(3, COVID_DATA, context['selected_country'], context['selected_time']),
        draw_measures_advanced_analysis(4, COVID_DATA, context['selected_country'], context['selected_time']),
        draw_measures_advanced_analysis(5, COVID_DATA, context['selected_country'], context['selected_time'])
    ]
    dask.compute(tasks, scheduler='distributed')
    return render(request, 'DEApp/AdvancedAnalysis.html', context)
