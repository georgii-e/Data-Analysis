import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error


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


initial_df = pd.read_csv(r'F:\Egor\Уроки\Аналіз даних\Лаб7\SeattleWeather.csv')
temps = initial_df.copy()
temps.index = pd.to_datetime(temps.loc[:, 'DATE'])
temps.drop(columns='DATE', inplace=True)
temps.info()
temps['PRCP'].fillna(0, inplace=True)
temps['RAIN'].fillna(False, inplace=True)
temps.loc[:, 'TMAX'] = temps.loc[:, 'TMAX'].apply(lambda x: 5 / 9 * (x - 32))
temps.loc[:, 'TMIN'] = temps.loc[:, 'TMIN'].apply(lambda x: 5 / 9 * (x - 32))

correlations = temps.corr()
# noinspection PyTypeChecker
corrMat = plt.matshow(correlations)
plt.xticks(range(len(correlations.columns)), correlations.columns, rotation=90)
plt.yticks(range(len(correlations.columns)), correlations.columns)
plt.colorbar(corrMat)
plt.show()

seasonal_decompose(temps['PRCP'].resample('M').mean()).plot().set_size_inches(15, 10)
plt.show()

last_years = temps.tail(5 * 365)
seasonal_decompose(last_years['PRCP'].resample('W').mean()).plot().set_size_inches(15, 10)
plt.show()

fig, ax = plt.subplots(2, figsize=(15, 10))
ax[0] = plot_acf(last_years['PRCP'].resample('W').mean(), ax=ax[0], lags=100)
ax[1] = plot_pacf(last_years['PRCP'].resample('W').mean(), ax=ax[1], lags=100)
plt.show()

adf_test(temps['PRCP'])

train_data = temps['PRCP'].loc[:'2017-01-01']
test_data = temps['PRCP'].loc['2017-01-02':]

train_data.describe()

model = ARIMA(train_data, order=(1, 0, 3), freq='D').fit()
print(model.summary())

predictions = model.forecast(steps=len(test_data) + 365)
fig, ax = plt.subplots(figsize=(12, 6))
temps['PRCP'].tail(800).plot(ax=ax)
predictions.plot(ax=ax, label='Predicted')
ax.legend()
plt.show()

mse = mean_squared_error(test_data, predictions[:len(test_data)])
print(f"Mean Squared Error: {mse :.4f}")
