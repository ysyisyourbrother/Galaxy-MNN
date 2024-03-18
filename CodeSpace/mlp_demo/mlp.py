import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.onnx as onnx

# 检查是否有可用的GPU设备
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义多层感知器模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(hidden_sizes)):
            if i == 0:
                self.layers.append(nn.Linear(input_size, hidden_sizes[i]))
            else:
                self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.layers.append(nn.ReLU())
        self.output_layer = nn.Linear(hidden_sizes[-1], num_classes)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

# 定义参数
input_size = 784
hidden_sizes = [128, 64]
num_classes = 10

# 创建模型实例并将其移动到GPU
model = MLP(input_size, hidden_sizes, num_classes)
#model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# 加载MNIST数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 将数据移动到GPU
        # images = images.reshape(-1, input_size).to(device)
        # labels = labels.to(device)
        images = images.reshape(-1, input_size)
        labels = labels
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印训练信息
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# 保存模型
torch.save(model.state_dict(), 'mlp_minist_cpu.pth')
print("model saved")

# 加载测试数据集
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False)

# 使用模型进行图像识别并计算准确性
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        # 将数据移动到GPU
        #images = images.reshape(-1, input_size).to(device)
        images = images.reshape(-1, input_size)
        labels = labels
        # 在GPU上进行推理
        outputs = model(images)
        _, predicted_labels = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted_labels == labels).sum().item()

# 输出准确性
accuracy = correct / total * 100
print('Accuracy on test set: {:.2f}%'.format(accuracy))