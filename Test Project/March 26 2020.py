import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

iris = pd.read_csv('C:\COSC 4381 - AI in Python\Class Jupyter notebooks\iris.data', header=None, encoding='utf-8')

print(iris.tail())

iris.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

X = iris.loc[:, ['sepal_length', 'petal_length']]

sc = StandardScaler()

X_std = sc.fit_transform(X)

plt.scatter(X_std[:, 0], X_std[:, 1], c='white', marker='o', edgecolors='black', s=50)
plt.grid()
plt.tight_layout()
plt.show()

km = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=0)

y_km = km.fit_predict(X_std)
