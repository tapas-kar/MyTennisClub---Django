from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

boston_df = pd.read_csv('C:\COSC 4381 - AI in Python\Class Jupyter notebooks\\boston_house_prices.txt')


desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 15)

# print(boston_df)

# print(boston_df.columns)

# plt.show(sns.pairplot(boston_df))

# plt.show(sns.pairplot(boston_df.loc[:, ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']]))

# print(boston_df.corr())

# plt.show(sns.heatmap(boston_df.corr(), annot=True))

# plt.show(sns.heatmap(boston_df.loc[:, ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']].corr(), annot=True))

############# LINEAR REGRESSION

class LinearRegressionGD(object):
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            cost = 0

            for xi, target in zip(X, y):
                output = self.net_input(xi)
                errors = (target - output)

                for z in range(len(xi)):
                    self.w_[1+z] += self.eta * errors * xi[z]
                self.w_[0] += self.eta * errors

                cost += (errors**2)/2.0

            self.cost_.append(cost)
            print(f"cost = {cost} and iteration = {i}")
        return self

    def net_input(self, X):
        sum_ = 0
        for xi in range(len(X)):
            sum_ += X[xi] * self.w_[1+xi]
        sum_ += self.w_[0]
        return sum_

    def predict(self, X):
        return self.net_input(X)


X = boston_df[['RM']].values

y = boston_df['MEDV'].values

# print(X.shape, y.shape)

sc = StandardScaler()

X_std = sc.fit_transform(X)

y_new = y[:, np.newaxis]

# print(y_new)

y_std = sc.fit_transform(y_new)

# important for lr.cost_ to work, otherwise it does not detect it
y_std = y_std.flatten()

lr = LinearRegressionGD()

lr.fit(X_std, y_std)

plt.show(plt.plot(range(1, lr.n_iter+1), lr.cost_))

for i in range(len(y_std)):
    if y_std[i] != lr.predict(X_std[i]):
        print("Error!")
    else:
        print("Correct!")

