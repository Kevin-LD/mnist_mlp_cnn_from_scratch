import matplotlib.pyplot as plt

def plot_metrics(runner, model_name):
    epochs = range(1, len(runner.train_losses) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # 左图：Loss 变化曲线
    axes[0].plot(epochs, runner.train_losses, label='Train Loss', marker='o', color='#1f77b4', linewidth=2)
    axes[0].plot(epochs, runner.dev_losses, label='Dev Loss', marker='s', color='#ff7f0e', linewidth=2, linestyle='--')
    axes[0].set_title(f'{model_name} - Loss Curve', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epochs', fontsize=10)
    axes[0].set_ylabel('Loss', fontsize=10)
    axes[0].set_xticks(epochs)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(frameon=True)
    
    # 右图：Accuracy (Score) 变化曲线
    axes[1].plot(epochs, runner.train_scores, label='Train Acc', marker='o', color='#2ca02c', linewidth=2)
    axes[1].plot(epochs, runner.dev_scores, label='Dev Acc', marker='s', color='#d62728', linewidth=2, linestyle='--')
    axes[1].set_title(f'{model_name} - Accuracy Curve', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epochs', fontsize=10)
    axes[1].set_ylabel('Accuracy', fontsize=10)
    axes[1].set_xticks(epochs)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend(frameon=True)
    
    plt.tight_layout()
    
    # 自动创建或识别并保存在本地一份图片成果
    output_png = f"./figs/{model_name.lower()}_training_metrics.png"
    plt.savefig(output_png, dpi=300)
    print(f"\n[可视化] 训练图表已成功保存至本地: {output_png}")
    plt.show()
