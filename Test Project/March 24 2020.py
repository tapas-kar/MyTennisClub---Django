from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

iris = datasets.load_iris()

X = iris.data
y = iris.target

# print(X)

sc = StandardScaler()

X_std = sc.fit_transform(X)

km = KMeans(n_clusters=3, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)

y_km = km.fit_predict(X_std)

plt.scatter(X_std[y_km == 0, 0], X_std[y_km == 0, 1], s=50, c='lightgreen', marker='s', edgecolor='black'
            , label='Cluster 1')
plt.scatter(X_std[y_km == 1, 0], X_std[y_km == 1, 1], s=50, c='orange', marker='o', edgecolor='black'
            , label='Cluster 2')
plt.scatter(X_std[y_km == 2, 0], X_std[y_km == 2, 1], s=50, c='lightblue', marker='v', edgecolor='black'
            , label='Cluster 3')

plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=250, marker='*', c='red'
            , edgecolors='black', label='Centroids')

plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()
plt.show()

