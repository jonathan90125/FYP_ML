import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 生成虚拟数据集
# X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# np.savetxt("new.csv", X, delimiter=',')


my_matrix = np.loadtxt(open("new.csv","rb"), delimiter=",", skiprows=0)
y = np.loadtxt(open("newy.csv","rb"), delimiter=",", skiprows=0)
# 将数据集分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(my_matrix, y, test_size=0.2, random_state=42)




my_matrix=my_matrix[:,[3,4]]
plt.figure("show")
plt.scatter(my_matrix[:,0],my_matrix[:,1],color=0)
plt.show()

# 定义XGBoost分类器
model = xgb.XGBClassifier(
    learning_rate=0.1,
    n_estimators=100,
    max_depth=3,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27
)

# 在训练数据上训练模型
model.fit(X_train, y_train)

# 在测试数据上进行预测
y_pred = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
