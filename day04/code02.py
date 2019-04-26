# 多项式回归
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X ** 2 * +X + 2 + np.random.rand(m, 1)

poly_features = PolynomialFeatures(degree=2, include_bias=False)
# 转换训练集，将每个特征的平方作为新特征加入训练集
X_poly = poly_features.fit_transform(X)

# 用扩展后的数据，匹配一个LinearRegression模型

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

plt.plot(X, y, '.')
plt.plot(X, lin_reg.predict(X_poly), color="red")
plt.show()
