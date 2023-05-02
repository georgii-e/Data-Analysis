import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

initial_df = pd.read_csv(r'F:\Egor\Уроки\Аналіз даних\Лаб6\titanic.csv')
titanic = initial_df.copy()
titanic.info()
titanic = titanic.drop(columns=['PassengerId', 'Name', 'Cabin', 'Ticket'])
titanic.loc[:, 'Age'].fillna(titanic.loc[:, 'Age'].mean(numeric_only=True), inplace=True)
titanic.loc[:, 'Embarked'].fillna(titanic.loc[:, 'Embarked'].mode()[0], inplace=True)
titanic.info()
all_features = pd.get_dummies(titanic)
X = all_features.drop(columns='Survived')
y = all_features.loc[:, 'Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

log_reg = LogisticRegression(solver='liblinear')
log_reg.fit(X_train, y_train)
log_accuracy = log_reg.score(X_test, y_test)
print(f'Logistic regression model accuracy: {log_accuracy * 100:.2f}%')

dtree = DecisionTreeClassifier(max_depth=3)
dtree.fit(X_train, y_train)
dtree_accuracy = dtree.score(X_test, y_test)
print(f'Decision tree model accuracy: {dtree_accuracy * 100:.2f}%')

rf = RandomForestClassifier(max_depth=5)
rf.fit(X_train, y_train)
rf_accuracy = rf.score(X_test, y_test)
print(f'Random forest model accuracy: {rf_accuracy * 100:.2f}%')

gb = GradientBoostingClassifier(max_depth=3)
gb.fit(X_train, y_train)
gb_accuracy = gb.score(X_test, y_test)
print(f'Gradient boosting model accuracy: {gb_accuracy * 100:.2f}%')
