import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def pie_plot():
    labels = ['Heart disease is observed', 'No heart disease is observed']
    plt.figure(figsize=(6, 6))
    distribution = heart_disease['TenYearCHD'].value_counts()
    print(f'Total participants with coronary heart disease: {distribution[1]}')
    print(f'Total participants without coronary heart disease: {distribution[0]}')
    plt.pie(distribution, labels=labels, autopct='%0.2f%%')
    plt.title('Proportion of participants with and without heart disease')
    plt.show()


def correlations():
    correlations = heart_disease.corr()
    most_correlated_param = correlations.loc['TenYearCHD'].drop('TenYearCHD').idxmax()
    print(f'Parameter that correlates the most with the possibility of coronary heart disease: {most_correlated_param}')
    print(f'Сorresponding coefficient: {correlations.loc[most_correlated_param, "TenYearCHD"] * 100:.2f}%')
    # noinspection PyTypeChecker
    corr_plt = plt.matshow(correlations)
    plt.xticks(range(len(correlations.columns)), correlations.columns, rotation=90)
    plt.yticks(range(len(correlations.columns)), correlations.columns)
    plt.colorbar(corr_plt)
    plt.show()


def choose_hyperparameters_log(log_reg):
    param_grid_1 = {
        'C': [0.5, 1, 5, 10],
        'solver': ['liblinear', 'newton-cholesky'],
        'max_iter': [100, 300, 500]
    }

    param_grid_2 = {
        'C': [0.5, 1, 5, 10],
        'solver': ['lbfgs'],
        'max_iter': [2500, 3000, 5000]
    }
    grid_search = GridSearchCV(estimator=log_reg, param_grid=param_grid_1)
    grid_search.fit(X_train, y_train)

    print("Best hyperparameters from first grid:", grid_search.best_params_)
    best_model_1 = grid_search.best_estimator_
    accuracy_1 = best_model_1.score(X_test, y_test)
    print(f'Test accuracy with first model: {accuracy_1 * 100:.2f}%')

    grid_search = GridSearchCV(estimator=log_reg, param_grid=param_grid_2)
    grid_search.fit(X_train, y_train)

    print("Best hyperparameters from second grid:", grid_search.best_params_)
    best_model_2 = grid_search.best_estimator_
    accuracy_2 = best_model_2.score(X_test, y_test)
    print(f'Test accuracy with second model: {accuracy_2 * 100:.2f}%')

    if accuracy_1 > accuracy_2:
        return best_model_1
    return best_model_2


def choose_hyperparameters_dec(dec_tree):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10],
    }
    grid_search = GridSearchCV(estimator=dec_tree, param_grid=param_grid)
    grid_search.fit(X_train, y_train)
    print("Best hyperparameters from the grid:", grid_search.best_params_)
    best_model = grid_search.best_estimator_
    accuracy = best_model.score(X_test, y_test)
    print(f'Test accuracy with best Decision Tree model: {accuracy * 100:.2f}%')

    return best_model


def choose_hyperparameters_rand(rand_forest):
    param_grid = {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': [None, 5, 10]
    }
    grid_search = GridSearchCV(estimator=rand_forest, param_grid=param_grid)
    grid_search.fit(X_train, y_train)
    print("Best hyperparameters from the grid:", grid_search.best_params_)
    best_model = grid_search.best_estimator_
    accuracy = best_model.score(X_test, y_test)
    print(f'Test accuracy with best Random Forest model: {accuracy * 100:.2f}%')

    return best_model


def choose_hyperparameters_svc(svc):
    param_grid = {
        'C': [1, 3, 5, 10],
        'degree': [3, 4, 5],
        'gamma': ['scale', 'auto'],
    }
    grid_search = GridSearchCV(estimator=svc, param_grid=param_grid)
    grid_search.fit(X_train, y_train)
    print("Best hyperparameters from the grid:", grid_search.best_params_)
    best_model = grid_search.best_estimator_
    accuracy = best_model.score(X_test, y_test)
    print(f'Test accuracy with best SVC model: {accuracy * 100:.2f}%')

    return best_model


def confusion_matrix_plot(classifier, test_x, test_y):
    y_pred = classifier.predict(test_x)
    cm = confusion_matrix(test_y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()


initial_df = pd.read_csv(r"F:\Egor\Уроки\Аналіз даних\Курсова робота\Framingham.csv")
heart_disease = initial_df.copy()
heart_disease.info()
print(heart_disease.describe())

pie_plot()
correlations()

X = heart_disease.drop(columns='TenYearCHD')
y = heart_disease.loc[:, 'TenYearCHD']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

default_log_reg = LogisticRegression(solver='liblinear')  # was default solver in previous version of linear regression;
# no convergence with lbfgs solver and default max_iter parameter
default_log_reg.fit(X_train, y_train)
log_accuracy = default_log_reg.score(X_test, y_test)
print(f'Default logistic regression model accuracy: {log_accuracy * 100:.2f}%')
f1 = f1_score(y_test, default_log_reg.predict(X_test))
print("F1 Score for default decision tree model:", f1)

logistic_regression = LogisticRegression()
optimal_log_reg = choose_hyperparameters_log(logistic_regression)
f1 = f1_score(y_test, optimal_log_reg.predict(X_test))
print("F1 Score for optimal decision tree model:", f1)
confusion_matrix_plot(optimal_log_reg, X_test, y_test)

default_dec_tree = DecisionTreeClassifier()
default_dec_tree.fit(X_train, y_train)
dtree_accuracy = default_dec_tree.score(X_test, y_test)
print(f'Default decision tree model accuracy: {dtree_accuracy * 100:.2f}%')
f1 = f1_score(y_test, default_dec_tree.predict(X_test))
print("F1 Score for default decision tree model:", f1)

decision_tree = DecisionTreeClassifier()
optimal_dec_tree = choose_hyperparameters_dec(decision_tree)
f1 = f1_score(y_test, optimal_dec_tree.predict(X_test))
print("F1 Score for optimal decision tree model:", f1)
confusion_matrix_plot(optimal_dec_tree, X_test, y_test)

default_rand_forest = RandomForestClassifier()
default_rand_forest.fit(X_train, y_train)
rforest_accuracy = default_rand_forest.score(X_test, y_test)
print(f'Default random forest model accuracy: {rforest_accuracy * 100:.2f}%')
f1 = f1_score(y_test, default_rand_forest.predict(X_test))
print("F1 Score for default random forest model:", f1)

random_forest = RandomForestClassifier()
optimal_rand_forest = choose_hyperparameters_rand(random_forest)
f1 = f1_score(y_test, optimal_rand_forest.predict(X_test))
print("F1 Score for optimal random forest model:", f1)
confusion_matrix_plot(optimal_rand_forest, X_test, y_test)

default_svc = SVC()
default_svc.fit(X_train, y_train)
svc_accuracy = default_svc.score(X_test, y_test)
print(f'Default SVC model accuracy: {svc_accuracy * 100:.2f}%')
f1 = f1_score(y_test, default_svc.predict(X_test))
print("F1 Score for default SVC model:", f1)

SVC = SVC()
optimal_svc = choose_hyperparameters_svc(SVC)
f1 = f1_score(y_test, optimal_svc.predict(X_test))
print("F1 Score for optimal SVC model:", f1)
confusion_matrix_plot(optimal_svc, X_test, y_test)
