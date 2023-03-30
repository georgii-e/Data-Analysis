import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.precision', 3)


def str_to_float(data_frame, column_name):
    data_frame[column_name] = abs(data_frame[column_name].apply(
        lambda x: x.replace(',', '.') if isinstance(x, str) else np.NaN).astype('float'))


def make_boxplot(data_frame, column_name):
    plt.figure()
    plt.title(f'Boxplot for {column_name}')
    plt.boxplot(data_frame[column_name])
    # plt.show()


def make_histogram(data_frame, column_name):
    plt.figure()
    plt.title(f'Histogram for {column_name}')
    plt.hist(data_frame[column_name])
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    # plt.show()


initial_df = pd.read_csv(r'F:\Egor\Уроки\Аналіз даних\Лаб3\Data2.csv', sep=';', encoding='cp1252')
df = initial_df.copy()
df.rename(columns={'Populatiion': 'Population'}, inplace=True)
str_to_float(df, 'GDP per capita')
str_to_float(df, 'CO2 emission')
str_to_float(df, 'Area')
print(df.head(5))
df.info()

df.fillna(df.mean(numeric_only=True), inplace=True)  # additional task 1
df.info()
make_boxplot(df, 'GDP per capita')
make_histogram(df, 'GDP per capita')

df['Population density'] = df['Population'] / df['Area']

print("Country with biggest GDP per capita:\n", df.iloc[df['GDP per capita'].idxmax()], end="\n\n")  # additional task 2
print("Country with smallest area:\n", df.iloc[df['Area'].idxmin()], end="\n\n")

grouped_regions_mean = df.groupby('Region').mean(numeric_only=True)
max_area_index = grouped_regions_mean.index.get_loc(grouped_regions_mean['Area'].idxmax())
print("Region with highest average country area:\n", grouped_regions_mean.iloc[max_area_index],
      end="\n\n")  # additional task 3

print("Country with biggest population density:\n", df.iloc[df['Population density'].idxmax()], end="\n\n")
print("Country with biggest population density in Europe and Central Asia:\n",  # additional task 4
      df.iloc[df.loc[df['Region'] == 'Europe & Central Asia', 'Population density'].idxmax()], end="\n\n")

grouped_regions_gdp = df.groupby('Region')['GDP per capita'].agg(['mean', 'median'])
regions = grouped_regions_gdp.index[grouped_regions_gdp['mean'] == grouped_regions_gdp['median']]
if len(regions):
    print(f'The average and median of GDP are the same in {" ,".join(regions)}')  # additional task 5
else:
    print('The average and median of GDP are different in all regions.\n')

df.sort_values('GDP per capita', ascending=False, inplace=True)
top_5_gdp = df.head(5)
last_5_gdp = df.tail(5)
print('Top 5 countries by GDP:')
for index, row in top_5_gdp.iterrows():  # additional task 6
    print(f'{row.loc["Country Name"]}: {row.loc["GDP per capita"]}')
print('\nLast 5 countries by GDP:')
for index, row in last_5_gdp.iterrows():
    print(f'{row.loc["Country Name"]}: {row.loc["GDP per capita"]}')

df.sort_values('CO2 emission', ascending=False, inplace=True)
top_5_c02 = df.head(5)
last_5_c02 = df.tail(5)
print('\nTop 5 countries by C02 emission:')
for index, row in top_5_c02.iterrows():  # additional task 6
    print(f'{row.loc["Country Name"]}: {row.loc["CO2 emission"]}')
print('\nLast 5 countries by C02 emission:')
for index, row in last_5_c02.iterrows():
    print(f'{row.loc["Country Name"]}: {row.loc["CO2 emission"]}')
