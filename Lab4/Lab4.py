import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

pd.set_option('display.precision', 3)
pd.set_option("display.max_columns", None)


def str_to_float(data_frame, column_name):
    data_frame[column_name] = abs(data_frame[column_name].apply(
        lambda x: x.replace(',', '.') if isinstance(x, str) else np.NaN).astype('float'))


def normal_distribution_test(data_frame, column_name, alpha=0.05):  # D’Agostino-Pearson method
    stat, p = stats.normaltest(data_frame[column_name])
    print(f'Statistics = {stat:.3f}, p = {p:.3f}')
    if p > alpha:
        print('The data follows a normal distribution')
    else:
        print('Data does not follow a normal distribution')


def median_and_mean_difference(data_frame, column_name):
    print(f'Difference between median and mean: '
          f'{(data_frame[column_name].mean() - data_frame[column_name].median()) / data_frame[column_name].mean() * 100:.2f}%, '
          f'mean = {int(data_frame[column_name].mean())}, median = {int(data_frame[column_name].median())}')


initial_df = pd.read_csv(r'F:\Egor\Уроки\Аналіз даних\Лаб3\Data2.csv', sep=';', encoding='cp1252')
df = initial_df.copy()
df.rename(columns={'Populatiion': 'Population'}, inplace=True)
str_to_float(df, 'GDP per capita')
str_to_float(df, 'CO2 emission')
str_to_float(df, 'Area')
df.fillna(df.mean(numeric_only=True), inplace=True)  # fulfill nan values
print(df.head())
print('\n\n')

normal_distribution_test(df, 'Population')
normal_distribution_test(df, 'GDP per capita')
normal_distribution_test(df, 'CO2 emission')
normal_distribution_test(df, 'Area')
print('\n\n')

median_and_mean_difference(df, 'Population')
median_and_mean_difference(df, 'GDP per capita')
median_and_mean_difference(df, 'CO2 emission')
median_and_mean_difference(df, 'Area')
print('\n\n')

regions = pd.unique(df['Region'])
for region in regions:
    regional_emissions = df[df['Region'] == region]['CO2 emission']
    statistic, p_value = stats.shapiro(regional_emissions)  # Shapiro-Wilk method, cause in North America <8 countries
    if p_value > 0.05:
        print(f'{region} region has a normal distribution of CO2 emissions')
    else:
        print(f'{region} region does not have a normal distribution of CO2 emissions')

print('\n\n')

fig, ax = plt.subplots()
labels = pd.unique(df['Region'])
wedges, texts, autotexts = ax.pie(df.groupby('Region').sum(numeric_only=True)['Population'], labels=labels,
                                  autopct='%.2f%%')
ax.set_title('Population by region')
ax.legend(wedges, labels, title='Regions', loc='best', bbox_to_anchor=(1, 1))
plt.show()
