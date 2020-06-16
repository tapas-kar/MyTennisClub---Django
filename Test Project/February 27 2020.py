# Random forest

from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()

X = iris.data
y = iris.target

print(np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

print(y_train)

print(y_test)

forest = RandomForestClassifier(criterion="gini", n_estimators=500, random_state=1, n_jobs=2)

forest.fit(X_train, y_train)

y_pred = forest.predict(X_test)

print(y_pred)

print("Misclassified example: %d" % (y_test != y_pred).sum())

print(forest.feature_importances_)

# print("Confusion Matrix")
confusion_matrix(y_true=y_test, y_pred=y_pred)

# print("Precision Matrix")
# precision_score(y_true=y_test, y_pred=y_pred, average=None)

# print("Recall Score")
# recall_score(y_true=y_test, y_pred=y_pred, average=None)

# print("F1 Score")
# f1_score(y_true=y_test, y_pred=y_pred, average='micro')

print("Classification Report:", classification_report(y_true=y_test, y_pred=y_pred))



#### KNN Classifier

knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')

sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

knn.fit(X_train_std, y_train)

y_pred = knn.predict(X_test)

print("Misclassified examples: %d" % (y_test != y_pred).sum())

