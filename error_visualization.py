import os
import json
import gzip
import numpy as np
from struct import unpack
import matplotlib.pyplot as plt
import mynn as nn

model_path = r'best_models/mlp_drop0.0_sgd_20260521_175444_baseline/best_model.pickle'

model_dir = os.path.dirname(model_path)
json_path = os.path.join(model_dir, 'model_metadata.json')

# 自动读取元数据
if os.path.exists(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    model_type = metadata["model_meta"]["model_type"]
    hyperparams = metadata.get("hyperparameters", {})
    print(f"成功识别元数据！当前分析的模型为: {model_type}")
else:
    print("未找到元数据 json：默认按标准 MLP 处理")
    model_type = "Model_MLP"
    hyperparams = {"size_list": [784, 600, 10], "act_func": "ReLU"}

# 2. 动态实例化模型并加载权重
if "CNN" in model_type:
    model = nn.models.Model_CNN(weight_decay=True, weight_decay_lambda=0, use_bn=hyperparams.get("use_bn", False))
else:
    model = nn.models.Model_MLP(size_list=hyperparams.get("size_list", [784, 600, 10]), act_func=hyperparams.get("act_func", "ReLU"), drop_rate=0.0)

model.load_model(model_path)

# 必将模型切换为 eval 模式
if hasattr(model, 'eval'):
    model.eval()
print("--> 模型已成功切换至评估(eval)模式。")

test_images_path = r'./dataset/MNIST/t10k-images-idx3-ubyte.gz'
test_labels_path = r'./dataset/MNIST/t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)
    
with gzip.open(test_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs_normalized = test_imgs / 255.0

if "CNN" in model_type:
    test_imgs_fed = test_imgs_normalized.reshape(-1, 1, 28, 28)
else:
    test_imgs_fed = test_imgs_normalized.reshape(-1, 28 * 28)

print("正在对测试集进行全量推理...")
logits = model(test_imgs_fed)
preds = np.argmax(logits, axis=1)

total_acc = np.mean(preds == test_labs)
print(f"校验测试集总准确率 (Test Accuracy): {total_acc * 100:.2f}%")

error_indices = np.where(preds != test_labs)[0]
print(f"发现错例共计: {len(error_indices)} 个")


# 可视化一：绘制经典的 4x4 错例矩阵
num_errors_to_show = min(16, len(error_indices))
if num_errors_to_show > 0:
    fig, axes = plt.subplots(4, 4, figsize=(9, 9))
    axes = axes.flatten()
    
    # 为了保证实验的多样性，这里随机抽取 16 个错例（也可以去掉 choice 换成顺序截取）
    sampled_error_idx = np.random.choice(error_indices, num_errors_to_show, replace=False)
    
    for i, idx in enumerate(sampled_error_idx):
        # 还原出原始的 28x28 图像
        img_disp = test_imgs[idx].reshape(28, 28)
        axes[i].imshow(img_disp, cmap='gray')
        
        true_label = test_labs[idx]
        pred_label = preds[idx]
        
        # 将真实标签和预测标签打在标题上
        axes[i].set_title(f"True: {true_label} | Pred: {pred_label}", fontsize=10, color='darkred')
        axes[i].axis('off')
        
    plt.suptitle(f"Misclassified Examples Display ({model_type})", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
else:
    print("模型没有出现任何错例")


# 可视化二：计算并绘制混淆矩阵 (Confusion Matrix)
num_classes = 10
cm = np.zeros((num_classes, num_classes), dtype=int)

for t, p in zip(test_labs, preds):
    cm[t, p] += 1

fig, ax = plt.subplots(figsize=(9, 8))
# 使用 Blues 渐变色盘，对角线由于正确率高，颜色最深
cax = ax.matshow(cm, cmap='Blues')
fig.colorbar(cax, fraction=0.046, pad=0.04)

ax.set_xticks(np.arange(num_classes))
ax.set_yticks(np.arange(num_classes))
ax.set_xticklabels(np.arange(num_classes))
ax.set_yticklabels(np.arange(num_classes))

ax.set_xlabel('Predicted Label', labelpad=10, fontsize=12)
ax.set_ylabel('True Label', labelpad=10, fontsize=12)
ax.xaxis.set_label_position('bottom')
ax.xaxis.tick_bottom()

thresh = cm.max() / 2.0
for i in range(num_classes):
    for j in range(num_classes):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=10)

plt.title(f"Confusion Matrix Heatmap ({model_type})", fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()

plt.show()
