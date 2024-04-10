import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 创建随机的回归数据集
X = np.loadtxt(open("X.csv","rb"), delimiter=",", skiprows=0)
y = np.loadtxt(open("y.csv","rb"), delimiter=",", skiprows=0)
# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化XGBRegressor
xgb_regressor = XGBRegressor(objective='reg:squarederror', random_state=42)

# 在训练集上拟合模型
xgb_regressor.fit(X_train, y_train)

# 使用训练好的模型进行预测
predictions = xgb_regressor.predict(X_test)

# 评估模型性能
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

# 绘制实际值与预测值的散点图
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, color='blue', alpha=0.5)
plt.xlabel('Actual Values')
plt.ylabel('Predictions')
plt.title('Actual vs Predicted Values (XGBoost)')
plt.show()
