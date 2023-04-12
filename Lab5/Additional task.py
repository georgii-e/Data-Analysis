import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def str_to_float(data_frame, column_name):
    data_frame[column_name] = abs(data_frame[column_name].apply(
        lambda x: x.replace(',', '.') if isinstance(x, str) else np.NaN).astype('float'))


initial_df1 = pd.read_csv(r'F:\Egor\Уроки\Аналіз даних\Лаб5\Data4.csv', sep=';', encoding='windows-1251')
df = initial_df1.copy()
df.rename(columns={'Unnamed: 0': 'Country'}, inplace=True)
str_to_float(df, 'Cql')
str_to_float(df, 'Ie')
str_to_float(df, 'Iec')
str_to_float(df, 'Is')
# df.info()

print(df.loc[:, ['Cql', 'Ie', 'Iec', 'Is']].corr())

pd.plotting.scatter_matrix(df)
plt.show()

y = df.loc[:, 'Cql']

linear_regression_all = LinearRegression()
linear_regression_all.fit(df.loc[:, ['Ie', 'Iec', 'Is']], y)

linear_regression_Is = LinearRegression()
linear_regression_Is.fit(df.loc[:, ['Is']], y)  # most correlated parameter

polynomial_regression = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
polynomial_regression.fit(df.loc[:, ['Ie', 'Iec', 'Is']], y)

initial_df2 = pd.read_csv(r'F:\Egor\Уроки\Аналіз даних\Лаб5\Data4t.csv', sep=';', encoding='windows-1251')
test_df = initial_df2.copy()
test_df.rename(columns={'Unnamed: 0': 'Country'}, inplace=True)
str_to_float(test_df, 'Cql')
str_to_float(test_df, 'Ie')
str_to_float(test_df, 'Iec')
str_to_float(test_df, 'Is')

linear_regression_all_predicted = linear_regression_all.predict(test_df.loc[:, ['Ie', 'Iec', 'Is']])
linear_regression_Is_predicted = linear_regression_Is.predict(test_df.loc[:, ['Is']])
polynomial_regression_predicted = polynomial_regression.predict(test_df.loc[:, ['Ie', 'Iec', 'Is']])

expected = test_df.loc[:, 'Cql']

lin_all_mse = mean_squared_error(expected, linear_regression_all_predicted)
lin_is_mse = mean_squared_error(expected, linear_regression_Is_predicted)
poly_mse = mean_squared_error(expected, polynomial_regression_predicted)

lin_all_r2 = r2_score(expected, linear_regression_all_predicted)
lin_is_r2 = r2_score(expected, linear_regression_Is_predicted)
poly_r2 = r2_score(expected, polynomial_regression_predicted)

lin_all_mae = mean_absolute_error(expected, linear_regression_all_predicted)
lin_is_mae = mean_absolute_error(expected, linear_regression_Is_predicted)
poly_mae = mean_absolute_error(expected, polynomial_regression_predicted)

print(f'Mean squared error for linear regression with all parameters: {lin_all_mse:.5f}, R-squared metric: {lin_all_r2:.3f}, mean absolute error: {lin_all_mae:.3f}')
print(f'Mean squared error for linear regression with only "Is" parameter: {lin_is_mse:.5f}, R-squared metric: {lin_is_r2:.3f}, mean absolute error: {lin_is_mae:.3f}')
print(f'Mean squared error for polynomial regression: {poly_mse:.5f}, R-squared metric: {poly_r2:.3f}, mean absolute error: {poly_mae:.3f}')
