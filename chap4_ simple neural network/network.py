import torch

def function(x):
    y = torch.sin(x) + torch.exp(x)
    return y
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 定义目标函数
def target_function(x):
    # return np.sin(3 * x) + 0.3 * x**3 + 0.5 * np.exp(-x**2)
    y = torch.sin(x) + torch.exp(x)
    return y
# 生成数据
n_train = 1000
n_test = 500

# 训练数据（随机采样）
x_train = np.random.uniform(-3, 3, size=(n_train, 1))
y_train = target_function(x_train)

# 测试数据（均匀采样）
x_test = np.linspace(-3, 3, n_test).reshape(-1, 1)
y_test = target_function(x_test)

# 转换为Tensor
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 定义神经网络
class FunctionApproximator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 512)
        self.layer2 = nn.Linear(512, 256)
        self.output_layer = nn.Linear(256, 1)
        
        # 权重初始化
        nn.init.kaiming_normal_(self.layer1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.layer2.weight, nonlinearity='relu')

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.output_layer(x)

# 初始化模型、损失函数和优化器
model = FunctionApproximator()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
num_epochs = 2000
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    # 训练模式
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    
    # 评估模式
    model.eval()
    with torch.no_grad():
        test_pred = model(x_test_tensor)
        test_loss = criterion(test_pred, y_test_tensor)
        test_losses.append(test_loss.item())
    
    if (epoch+1) % 200 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

# 可视化结果
model.eval()
with torch.no_grad():
    y_pred = model(x_test_tensor).numpy()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(x_test, y_test, label='True Function', linewidth=2)
plt.plot(x_test, y_pred, '--', label='Prediction', linewidth=2)
plt.scatter(x_train, y_train, color='red', alpha=0.1, label='Training Data')
plt.title('Function Fitting Result')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.subplot(1, 2, 2)
plt.semilogy(train_losses, label='Training Loss')
plt.semilogy(test_losses, label='Testing Loss')
plt.title('Training Process')
plt.xlabel('Epoch')
plt.ylabel('Loss (log scale)')
plt.legend()
plt.tight_layout()
plt.show()
