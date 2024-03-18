import torch
import torch.onnx
from mlp_minist import MLP
# 定义参数
input_size = 784  # 输入特征维度，这里以MNIST数据集为例，输入图像大小为28x28，展开后为784
hidden_sizes = [128, 64]  # 隐层大小，这里使用两层隐层，分别为128和64个神经元
num_classes = 10  # 分类类别数量，这里以MNIST数据集为例，共10个类别（0到9）
# 加载模型
model = MLP(input_size, hidden_sizes, num_classes)
model.load_state_dict(torch.load('mlp_minist_cpu.pth'))
model.eval()

temp_data = torch.randn(1,784)    # 1是batch_size
 
# 定义输入输出的名字，随便定，不过之后的操作好像会用到
input_names = ["input"]
output_names = ["output"]

#以下内容未调试

"""

# 将模型设置为训练模式
model.train()

# 使用训练数据进行前向传播，以确保模型中的训练相关操作被正确执行
with torch.no_grad():
    model(temp_data)  

torch.onnx.export(model, temp_data, "trainedModels/onnx/test.onnx", verbose=True, input_names=input_names, output_names=output_names)
print("onnx is ok")
    
"""


# 转换模型
torch.onnx.export(model, temp_data, "mlp_minist_cpu.onnx", export_params = True, input_names=input_names, output_names=output_names)
print("onnx is ok")

