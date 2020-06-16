import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

data = pd.read_csv(r'C:\Users\tapas\PycharmProjects\Test Project\final.csv', encoding='utf-8')

X = data.loc[:, ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10']]
y = data.loc[:, ['Group']]

# print(X)
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

print(y)
sc = StandardScaler()

X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)

y_train = y_train.loc[:, ['Group']]

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score


# print(np.unique(y))

# knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
#
# knn.fit(X_train_std, y_train)
#
# y_pred = knn.predict(X_test)
#
# print("Misclassified examples: %d" % (y_pred != y_test).sum())
#
# f1_score(y_true=y_test, y_pred=y_pred)
#
# print(f1_score())


# from sklearn.cluster import KMeans
#
# km = KMeans(n_clusters=4, n_init=10, max_iter=300, init='k-means++', tol=1e-04, random_state=0)
#
# y_km = km.fit_predict(X_train_std)
#
# print(km.inertia_, km.cluster_centers_)

from sklearn.neural_network import MLPClassifier

nn = MLPClassifier(hidden_layer_sizes=100, activation='logistic', alpha=0.01, max_iter=200, learning_rate_init=0.005,
                   batch_size=100, shuffle=True, random_state=1)

nn.fit(X_train_std, y_new)

y_test_pred = nn.predict(X_test_std)

acc = (np.sum(y_test == y_test_pred).astype(np.float)/ X_test_std.shape[0])

print(acc)


