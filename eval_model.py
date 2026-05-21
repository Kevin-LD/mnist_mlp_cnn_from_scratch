import os
import gzip
import json
import pickle
from struct import unpack
import numpy as np
import mynn as nn

RUN_DIR = r'best_models/cnn_nobn_sgd_20260521_175920'

metadata_path = os.path.join(RUN_DIR, 'model_metadata.json')
weights_path = os.path.join(RUN_DIR, 'best_model.pickle')

# 读取元数据
with open(metadata_path, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

model_type = metadata["model_meta"]["model_type"]
print(f"[*] 成功读取元数据 | 模型架构: {model_type}")
print(f"[*] 历史训练记录   | 运行 Epoch: {metadata['model_meta']['saved_epoch']} | 验证集最佳 Acc: {metadata['best_metrics']['dev_acc']:.4f}")

# 安全获取超参数字典（如果存在）
hp = metadata.get("hyperparameters", {})

if model_type == "Model_MLP":
    # 自动读取保存的隐藏层拓扑与激活函数
    model = nn.models.Model_MLP(size_list=hp["size_list"], act_func=hp["act_func"])
    reshape_shape = (-1, 28 * 28)  # MLP 使用 28*28 向量输入
    print(f"[*] 配置还原完成   | 隐层拓扑: {hp['size_list']} | 激活函数: {hp['act_func']}")
    
elif model_type == "Model_CNN":
    # 动态读取 BN 状态。若元数据中不存在该字段，默认设为 False 做到向前兼容
    use_bn = hp.get("use_bn", False)
    model = nn.models.Model_CNN(use_bn=use_bn)
    reshape_shape = (-1, 1, 28, 28)  # CNN 使用 [Batch, Channel, H, W] 四维输入
    print(f"[*] 配置还原完成   | 是否开启 BatchNorm2d: {use_bn}")
    
else:
    raise ValueError(f"未知的模型类型: {model_type}")

# 载入权重
model.load_model(weights_path)
print("[*] 模型权重载入成功。")

# 由于引入了 BN 或 Dropout，推理前必须显式切换到 eval 模式
if hasattr(model, 'eval'):
    model.eval()
    print("[*] 模型模式切换   | 已成功进入 eval 评估状态。")


eval_images_path = r'./dataset/MNIST/t10k-images-idx3-ubyte.gz'
eval_labels_path = r'./dataset/MNIST/t10k-labels-idx1-ubyte.gz'

with gzip.open(eval_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    # 根据模型类型自动决定 reshape 形状
    eval_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(reshape_shape)
    
with gzip.open(eval_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    eval_labs = np.frombuffer(f.read(), dtype=np.uint8)

eval_imgs = eval_imgs / 255.0

# 纯前向推理
logits = model(eval_imgs)
eval_acc = nn.metric.accuracy(logits, eval_labs)

print("\n" + "=" * 50)
print(f" 最终评估报告")
print("-" * 50)
print(f" 模型架构: {model_type}")
if model_type == "Model_CNN":
    print(f" 批 归 一: 使用了 BatchNorm2d" if hp.get("use_bn", False) else f" 批 归 一: 未使用 BN")
print(f" 测试准确率 (Eval Accuracy): {eval_acc * 100:.2f}%")
print("=" * 50)
