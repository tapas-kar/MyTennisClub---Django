# import libraries

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier


# QUESTION 1

# 1.1
data = pd.read_csv('data.csv', encoding='utf-8')

print(data.columns)

X = data.loc[:, ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5']]

print(X.shape)

y = data.loc[:, 'Category']

print(y.shape)

# 1.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

print(X_train.shape)

# 1.3
sc = StandardScaler()

X_train_std = sc.fit_transform(X_train)

X_test_std = sc.fit_transform(X_test)

print(X_test_std.shape)


# QUESTION 2

svm = SVC(C=1.0, kernel='rbf', gamma='auto', random_state=1)

svm.fit(X_train_std, y_train)

y_pred = svm.predict(X_test_std)

print("SVC Misclassified examples: %d" % (y_pred != y_test).sum())

f1_score_ = f1_score(y_true=y_test, y_pred=y_pred, average='micro')

print("SVC F1 score is ", f1_score_)



# QUESTION 3

forest = RandomForestClassifier(n_estimators=500, criterion="gini", random_state=1, n_jobs=2)

forest.fit(X_train_std, y_train)

y_pred = forest.predict(X_test_std)

print("Random Forest Misclassified examples: %d" % (y_pred != y_test).sum())

f1_score_ = f1_score(y_true=y_test, y_pred=y_pred, average='micro')

print("Random Forest Classifier F1 score is ", f1_score_)

