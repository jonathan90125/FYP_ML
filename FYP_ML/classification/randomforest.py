from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 生成虚拟数据集
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=23)

# 将数据集分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义随机森林分类器
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 在训练数据上训练模型
model.fit(X_train, y_train)

# 在测试数据上进行预测
y_pred = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
