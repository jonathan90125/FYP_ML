import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 创建一个虚拟的回归数据集
np.random.seed(0)
# X = np.random.rand(1000, 10) * 10
# y = 3 * X[:, 0].reshape(-1, 1) + np.random.randn(1000, 1) * 2  # 使用 X 的第一列作为标签，添加噪声
# print(y)
X = np.loadtxt(open("X.csv","rb"), delimiter=",", skiprows=0)
y = np.loadtxt(open("y.csv","rb"), delimiter=",", skiprows=0)
y=y.reshape(-1, 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y)

# 将数据转换为PyTorch的张量，并且将目标张量的形状改为 (1000, 1)
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# 定义多层感知器模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(10, 200)  # 输入层到隐藏层1
        self.fc2 = nn.Linear(200, 30)  # 隐藏层1到全连接层
        self.fc3 = nn.Linear(30, 10)  # 全连接层到隐藏层2
        self.fc4 = nn.Linear(10, 1)   # 隐藏层2到输出层

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = self.fc4(x)
        return x

# 初始化模型、损失函数和优化器
model = MLP()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 20000
losses = []
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 绘制损失曲线
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# 绘制预测结果的散点图
with torch.no_grad():
    predicted = model(X_tensor).detach().numpy()
predicted_original1 = scaler.inverse_transform(predicted)
predicted_original2 = scaler.inverse_transform(y)

plt.scatter(predicted_original1, predicted_original2, label='Predicted vs. Actual', color='red')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs. Actual Values')
plt.legend()
plt.show()
