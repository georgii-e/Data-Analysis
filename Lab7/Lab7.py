import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

pd.set_option('display.precision', 3)
pd.set_option("display.max_columns", None)


def adf_test(data):
    result = adfuller(data)
    print("ADF Test Statistic:", result[0])
    print("ADF p-value:", result[1])
    for key, value in result[4].items():
        print(f'Critical Value ({key}): {value}')
    if result[1] < 0.05:
        print("The time series is stationary.")
    else:
        print("The time series is non-stationary.")


def decompose(data):
    seasonal_decompose(data, model="additive").plot()
    plt.show()


def plot_moving_average(series, n, country_name=None):
    rolling_mean = series.rolling(window=n).mean()
    plt.figure(figsize=(15, 5))
    plt.plot(rolling_mean, c='orange', label='Rolling mean trend')
    plt.plot(series[n:], label='Actual values')
    if country_name:
        plt.title(f"New Covid-19 Cases in {country_name} ({n}-Day Moving Average)")
    else:
        plt.title(f"UAH course ({n}-Day Moving Average)")
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Cases')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()


initial_covid_df = pd.read_csv(r'F:\Egor\Уроки\Аналіз даних\Лаб7\CovidData.csv')
covid = initial_covid_df.copy()
covid = covid.loc[:, ['location', 'date', 'new_cases']]
print(covid.head())

covid['date'] = pd.to_datetime(covid.loc[:, 'date'])
covid = covid.loc[(covid.loc[:, 'location'] == 'France') | (covid.loc[:, 'location'] == 'Italy')]
covid = covid.pivot(index='date', columns='location', values='new_cases')
covid.fillna(method='bfill', inplace=True)
print(covid.head())
covid.info()

fig, ax = plt.subplots(figsize=(15, 10))
covid.plot(ax=ax)
plt.title('The number of new Covid-19 infections')
ax.grid()
plt.show()

italy_covid = covid.loc[:, 'Italy']
france_covid = covid.loc[:, 'France']

france_covid.hist(figsize=(8, 4))
plt.title('New cases in France')
plt.show()

plot_moving_average(france_covid, 5, 'France')
plot_moving_average(france_covid, 10, 'France')
plot_moving_average(france_covid, 20, 'France')

decompose(france_covid)

adf_test(france_covid)

italy_covid.hist(figsize=(8, 4))
plt.title('New cases in Italy')
plt.show()

plot_moving_average(italy_covid, 5, 'Italy')
plot_moving_average(italy_covid, 10, 'Italy')
plot_moving_average(italy_covid, 20, 'Italy')

decompose(italy_covid)

adf_test(italy_covid)

initial_currencies_df = pd.read_csv(r'F:\Egor\Уроки\Аналіз даних\Лаб7\EurToUah.csv')
currencies = initial_currencies_df.loc[:, ['Price', 'Date']].copy()
currencies.index = pd.to_datetime(currencies.loc[:, 'Date'])
currencies = currencies.loc[:, 'Price']
currencies.sort_index(inplace=True)
print(currencies.describe())

fig, ax = plt.subplots(figsize=(15, 10))
currencies.plot(ax=ax)
plt.title('UAH course')
ax.grid()
ax.legend()
plt.show()

currencies.hist(figsize=(8, 4))
plt.title('Histogram of the UAH against the EUR')
plt.show()

plot_moving_average(currencies, 5)
plot_moving_average(currencies, 10)
plot_moving_average(currencies, 20)

decompose(currencies)

adf_test(currencies)
