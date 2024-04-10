import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 创建一些随机的回归数据集
X = np.loadtxt(open("X.csv","rb"), delimiter=",", skiprows=0)
y = np.loadtxt(open("y.csv","rb"), delimiter=",", skiprows=0)

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 初始化RandomForestRegressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# 在训练集上拟合模型
rf_regressor.fit(X_train, y_train)

# 使用训练好的模型进行预测
predictions = rf_regressor.predict(X_test)

# 评估模型性能
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

# 绘制实际值与预测值的散点图
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, color='blue', alpha=0.5)
plt.xlabel('Actual Values')
plt.ylabel('Predictions')
plt.title('Actual vs Predicted Values')
plt.show()

# 获取特征的重要性
feature_importance = rf_regressor.feature_importances_

# 绘制特征重要性条形图
plt.figure(figsize=(8, 6))
plt.bar(np.arange(len(feature_importance)), feature_importance, color='green')
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.show()
