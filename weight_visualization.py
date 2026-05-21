import os
import json
import gzip
import pickle
from struct import unpack
import numpy as np
import matplotlib.pyplot as plt
import mynn as nn

# 指向你的模型 pickle 文件，脚本会自动推导同目录下的 json 元数据
model_path = r'best_models/cnn_bn_sgd_const_20260521_184521_BN/best_model.pickle'  # 替换为你的模型路径

model_dir = os.path.dirname(model_path)
json_path = os.path.join(model_dir, 'model_metadata.json')

# 1. 自动读取元数据
if os.path.exists(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    model_type = metadata["model_meta"]["model_type"]  # "Model_CNN" 或 "Model_MLP"
    hyperparams = metadata.get("hyperparameters", {})
    print(f"成功识别元数据！模型架构为: {model_type}")
else:
    # 若找不到 json，默认当作经典的 MLP 处理
    print("未找到元数据 json，默认按标准 MLP 处理")
    model_type = "Model_MLP"
    hyperparams = {"size_list": [784, 600, 10], "act_func": "ReLU", "drop_rate": 0.0}

# 2. 动态实例化对应的模型并加载权重
if "CNN" in model_type:
    model = nn.models.Model_CNN(
        weight_decay=True, 
        weight_decay_lambda=0, 
        use_bn=hyperparams.get("use_bn", False)
    )
else:
    model = nn.models.Model_MLP(
        size_list=hyperparams.get("size_list", [784, 600, 10]),
        act_func=hyperparams.get("act_func", "ReLU"),
        drop_rate=0.0  # 可视化评估阶段，关闭 Dropout
    )

model.load_model(model_path)
print("--> 权重矩阵加载并建立内存映射成功。")

# 3. 载入 MNIST 测试集数据
test_images_path = r'./dataset/MNIST/t10k-images-idx3-ubyte.gz'
test_labels_path = r'./dataset/MNIST/t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)
    
with gzip.open(test_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = test_imgs / 255.0  # 标准化到 [0, 1]

# 4. 根据模型类型自适应变换测试集维度（确保如果你想跑前向验证，它不会报错）
if "CNN" in model_type:
    test_imgs_fed = test_imgs.reshape(-1, 1, 28, 28)
else:
    test_imgs_fed = test_imgs.reshape(-1, 28 * 28)

# 解开下行注释可以验证模型前向推理
# logits = model(test_imgs_fed)


# 5. 提取所有可优化层的权重进行可视化
weight_layers = [layer for layer in model.layers if getattr(layer, 'optimizable', False)]

if "CNN" in model_type:
    print("正在绘制 CNN 卷积核特征...")
    # 提取第一层卷积层的权重，通常形状为 [out_channels, in_channels, k_h, k_w]
    W1 = weight_layers[0].params['W']
    out_channels = W1.shape[0]
    
    # 动态构建网格矩阵画图
    grid_size = int(np.ceil(np.sqrt(out_channels)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(6, 6))
    axes = axes.flatten()
    
    for i in range(out_channels):
        # 针对 MNIST，输入通道 in_channels=1，我们提取第 0 个通道的二维卷积核
        kernel = W1[i, 0] 
        axes[i].imshow(kernel, cmap='bwr') # bwr(蓝白红)色盘非常适合观察正负权重
        axes[i].axis('off')
        axes[i].set_title(f"Kernel {i}", fontsize=8)
        
    # 隐藏多余的子图
    for j in range(out_channels, len(axes)):
        axes[j].axis('off')
        
    plt.suptitle(f"{model_type} - First Conv2d Layer Weights", fontsize=14)
    plt.tight_layout()


else:
    print("正在绘制 MLP 隐藏层...")
    W1 = weight_layers[0].params['W']  # 形状: [784, 600]
    W2 = weight_layers[1].params['W']  # 形状: [600, 10]
    
    # 截取前 64 个具有代表性的隐含层神经元。
    num_features_to_show = min(64, W1.shape[1])
    grid_size = int(np.ceil(np.sqrt(num_features_to_show)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
    axes = axes.flatten()
    
    for i in range(num_features_to_show):
        # W1 每一列代表一个隐层神经元与输入 784 维向量连接的权重，将其重新拼回 28x28
        feature_template = W1[:, i].reshape(28, 28)
        axes[i].imshow(feature_template, cmap='bwr')
        axes[i].axis('off')
    
    plt.suptitle(f"{model_type} - First Hidden Layer Feature Templates (Top {num_features_to_show})", fontsize=12)
    plt.tight_layout()

plt.show()
