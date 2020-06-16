import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data
y = iris.target
# # print(np.unique(y))

# sc = StandardScaler()
#
# X_train_std = sc.fit_transform(X_train)
# X_test_std = sc.fit_transform(X_test)
#
# svm = SVC(kernel="linear", C=1.0, random_state=1)
# svm.fit(X_train_std, y_train)
#
# y_pred = svm.predict(X_test_std)
# print("Misclassified examples: %d" % (y_test != y_pred).sum())

# wine = datasets.load_wine()
#
# X2 = wine.data
# y2 = wine.target
#
# # print(np.unique(y2))
#
# X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, stratify=y2)
#
# sc2 = StandardScaler()
#
# X_train_std2 = sc2.fit_transform(X2_train)
# X_test_std2 = sc2.fit_transform(X2_test)
#
#
# svm2 = SVC(kernel="rbf", C=7.0, random_state=1)
# svm2.fit(X_train_std2, y2_train)
#
# y_pred = svm2.predict(X_test_std2)
#
# print("Misclassified examples: %d" % (y2_test != y_pred).sum())

print(X)
print(y)

tree_model = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

abc = tree_model.fit(X_train, y_train)

y_pred = tree_model.predict(X_test)

print("Misclassified examples: %d" % (y_pred != y_test).sum())

feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
dot_data = export_graphviz(abc, out_file= None, filled=True, rounded=True, special_characters=True,
                           feature_names=feature_cols, class_names=['0', '1', '2'])

graph = pydotplus.graph_from_dot_data(dot_data)

print(graph)
# graph.write_png('iris.png')

print(tree_model.feature_importances_)

