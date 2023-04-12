import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pd.set_option('display.precision', 3)
pd.set_option("display.max_columns", None)
initial_df = pd.read_csv(r'F:\Egor\Уроки\Аналіз даних\Лаб5\winequality-red.csv', encoding='cp1252')
df = initial_df.copy()
# df.info()

X = df.drop('quality', axis='columns')
y = df.loc[:, 'quality']
correlations = df.corr()
most_correlated_param = correlations.loc['quality'].drop('quality').idxmax()
print(f'Parameter that correlates the most with the quality: {most_correlated_param}, '
      f'with coefficient {correlations.loc[most_correlated_param, "quality"]:.2f}\n')
# noinspection PyTypeChecker
corrMat = plt.matshow(correlations)
plt.xticks(range(len(correlations.columns)), correlations.columns, rotation=90)
plt.yticks(range(len(correlations.columns)), correlations.columns)
plt.colorbar(corrMat)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=11)

linear_regression = LinearRegression()
linear_regression.fit(X_train.loc[:, [most_correlated_param]], y_train)
lin_predicted = linear_regression.predict(X_test.loc[:, [most_correlated_param]])
expected = y_test
for p, e in zip(lin_predicted[::100], expected[::100]):
    print(f'predicted: {p:.2f}, expected: {e:.2f}, difference: {(e - p) / p * 100:.2f}%')
print('\n')

multivariate_regression = LinearRegression()
multivariate_regression.fit(X_train, y_train)
multi_predicted = multivariate_regression.predict(X_test)
for p, e in zip(multi_predicted[::100], expected[::100]):
    print(f'predicted: {p:.2f}, expected: {e:.2f}, difference: {(e - p) / p * 100:.2f}%')
print('\n')

polynomial_regression = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
polynomial_regression.fit(X_train, y_train)
poly_predicted = polynomial_regression.predict(X_test)
for p, e in zip(poly_predicted[::100], expected[::100]):
    print(f'predicted: {p:.2f}, expected: {e:.2f}, difference: {(e - p) / p * 100:.2f}%')
print('\n')

lin_mse = mean_squared_error(expected, lin_predicted)
multi_mse = mean_squared_error(expected, multi_predicted)
poly_mse = mean_squared_error(expected, poly_predicted)

lin_r2 = r2_score(expected, lin_predicted)
multi_r2 = r2_score(expected, multi_predicted)
poly_r2 = r2_score(expected, poly_predicted)

lin_mae = mean_absolute_error(expected, lin_predicted)
multi_mae = mean_absolute_error(expected, multi_predicted)
poly_mae = mean_absolute_error(expected, poly_predicted)


print(f'Mean squared error for linear regression with {most_correlated_param} parameter: {lin_mse:.3f}, R-squared metric: {lin_r2:.3f}, mean absolute error: {lin_mae:.3f}')
print(f'Mean squared error for multivariate regression: {multi_mse:.3f}, R-squared metric: {multi_r2:.3f}, mean absolute error: {multi_mae:.3f}')
print(f'Mean squared error for polynomial regression: {poly_mse:.3f}, R-squared metric: {poly_r2:.3f}, mean absolute error: {poly_mae:.3f}')
