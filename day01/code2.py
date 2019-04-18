import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.linear_model import SGDClassifier

np.set_printoptions(threshold=np.inf)

# 准备训练集和测试集
mnist = fetch_mldata("MNIST original")
X, y = mnist["data"], mnist["target"]
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
y_train_5 = (y_train == 5)  # True for all 5s, False for all other digits.
y_test_5 = (y_test == 5)

# SGD分类
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)
print(sgd_clf.predict(X_test))
