import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# 创建一个虚拟的回归数据集
X = np.loadtxt(open("X.csv","rb"), delimiter=",", skiprows=0)
y = np.loadtxt(open("y.csv","rb"), delimiter=",", skiprows=0)
y=y.reshape(-1, 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y)
# 划分训练集和测试集
X_train, X_test = X[:800], X[800:]
y_train, y_test = y[:800], y[800:]

# 训练 Random Forest 模型
rf_model = RandomForestRegressor(n_estimators=100)
rf_model.fit(X_train, y_train.ravel())

# 训练 XGBoost 模型
xgb_model = XGBRegressor(n_estimators=100)
xgb_model.fit(X_train, y_train)

# 定义 BP 算法模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(10, 20)  # 输入层到隐藏层1
        self.fc2 = nn.Linear(20, 10)  # 隐藏层1到隐藏层2
        self.fc3 = nn.Linear(10, 1)   # 隐藏层2到输出层

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 初始化 BP 算法模型
bp_model = MLP()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(bp_model.parameters(), lr=0.01)

# 训练 BP 算法模型
X_tensor = torch.tensor(X_train, dtype=torch.float32)
y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = bp_model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

# 集成三个模型的预测结果
rf_predictions = rf_model.predict(X_test)
xgb_predictions = xgb_model.predict(X_test)
with torch.no_grad():
    bp_predictions = bp_model(torch.tensor(X_test, dtype=torch.float32)).numpy()

# 进行投票/平均
ensemble_predictions = (rf_predictions + xgb_predictions + bp_predictions.squeeze()) / 3
predicted_original1 = scaler.inverse_transform(y_test)
predicted_original2 = scaler.inverse_transform(ensemble_predictions.reshape(1, -1))
# 绘制预测结果的散点图
plt.scatter(predicted_original1, predicted_original2, label='Predicted vs. Actual', color='red')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs. Actual Values')
plt.legend()
plt.show()
