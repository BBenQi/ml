from sklearn.datasets import fetch_mldata
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import SGDClassifier
# from sklearn.metrics import confusion_matrix

# 准备训练集和测试集
mnist = fetch_mldata('MNIST original')
X, y = mnist["data"], mnist["target"]
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)  # 乱序
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# 训练一个二元分类器
# 处理训练集和测试集标签，改为二分类

# y_train_5 = (y_train == 5)
# y_test_5 = (y_test == 5)
# sgd_clf = SGDClassifier(random_state=42)
# sgd_clf.fit(X_train, y_train)
#
# # result = sgd_clf.predict(X_test) == y_test
# # print(np.sum(result == True) / result.shape[0])
# # 正确率 95.5%
# # print(sgd_clf.decision_function(X_test))
# print(confusion_matrix(y_test, sgd_clf.predict(X_test)))

# 多标签分类
y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
print(y_test[876])
print(knn_clf.predict([X_test[0]]))