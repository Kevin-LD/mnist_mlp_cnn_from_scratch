import os
import gzip
import pickle
from struct import unpack
import numpy as np
from datetime import datetime
import mynn as nn
from draw_tools import MyPlot

# 全局配置开关
model_type = "MLP"       # 可选: "CNN" 或 "MLP"
drop_rate = 0            # (MLP 中) Dropout 丢弃率。0 表示关闭
use_momentum = False     # 是否使用动量 SGD
use_bn = False           # 是否在 CNN 中加入 2D 批归一化层

# 调度器控制
use_scheduler = False     # 是否启用学习率衰减调度器
end_lr = 0.0             # 线性下降结束时的最终学习率 (通常设为 0.0)

current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
opt_suffix = "moment" if use_momentum else "sgd"
sched_suffix = "linear" if use_scheduler else "const"

# 动态组合本地存储文件夹名称
if model_type == "MLP":
    save_dir_name = f"{model_type.lower()}_drop{drop_rate}_{opt_suffix}_{sched_suffix}_{current_time_str}"
else:
    bn_suffix = "bn" if use_bn else "nobn"
    save_dir_name = f"{model_type.lower()}_{bn_suffix}_{opt_suffix}_{sched_suffix}_{current_time_str}"
    
dynamic_save_dir = os.path.join("./best_models", save_dir_name)

CONFIG = {
    "model_type": model_type,   
    "drop_rate": drop_rate,
    "use_momentum": use_momentum,  
    "use_bn": use_bn,
    
    "use_scheduler": use_scheduler,
    "end_lr": end_lr,
    
    "seed": 42,
    "batch_size": 32,
    "num_epochs": 15,
    "init_lr": 0.6,
    "save_dir": dynamic_save_dir,
    "use_wandb": True,
    "weight_decay": 0,
    "dataset": {
        "images": r'./dataset/MNIST/train-images-idx3-ubyte.gz',
        "labels": r'./dataset/MNIST/train-labels-idx1-ubyte.gz'
    }
}

# 固定随机种子
np.random.seed(CONFIG["seed"])


def load_and_preprocess_mnist(config):
    with gzip.open(config["dataset"]["images"], 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)
        
    with gzip.open(config["dataset"]["labels"], 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        labs = np.frombuffer(f.read(), dtype=np.uint8)

    # 乱序并持久化索引
    idx = np.random.permutation(np.arange(num))
    with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
        
    imgs, labs = imgs[idx], labs[idx]
    
    # 切分验证集
    valid_imgs, valid_labs = imgs[:10000], labs[:10000]
    train_imgs, train_labs = imgs[10000:], labs[10000:]

    train_imgs = train_imgs / 255.0
    valid_imgs = valid_imgs / 255.0

    # 根据模型架构自适应变换维度
    if config["model_type"] == "CNN":
        train_imgs = train_imgs.reshape(-1, 1, 28, 28)  # [Batch, Channel, H, W]
        valid_imgs = valid_imgs.reshape(-1, 1, 28, 28)
    elif config["model_type"] == "MLP":
        train_imgs = train_imgs.reshape(-1, 28 * 28)    # [Batch, Features]
        valid_imgs = valid_imgs.reshape(-1, 28 * 28)
    else:
        raise ValueError(f"未知的模型类型: {config['model_type']}")

    return (train_imgs, train_labs), (valid_imgs, valid_labs)


def main():
    if CONFIG["use_wandb"]:
        import wandb
        opt_name = "Moment" if CONFIG["use_momentum"] else "SGD"
        run_name = f"run_{CONFIG['model_type']}_{opt_name}"
        if CONFIG["model_type"] == "MLP":
            run_name += f"_drop_{CONFIG['drop_rate']}"
        elif CONFIG["model_type"] == "CNN":
            run_name += f"_bn_{CONFIG['use_bn']}"
            
        if CONFIG["use_scheduler"]:
            run_name += f"_LinearLR"
        else:
            run_name += f"_ConstLR"
            
        wandb.init(project="deep-learning-pj2-mnist", name=run_name, config=CONFIG)

    # 处理数据
    train_set, dev_set = load_and_preprocess_mnist(CONFIG)
    train_imgs, train_labs = train_set
    num_classes = int(train_labs.max() + 1)
    
    if CONFIG["model_type"] == "CNN":
        model = nn.models.Model_CNN(
            weight_decay=True, 
            weight_decay_lambda=CONFIG["weight_decay"],
            use_bn=CONFIG["use_bn"]
        )
    else:
        input_dim = train_imgs.shape[-1]
        model = nn.models.Model_MLP(
            size_list=[input_dim, 600, num_classes], 
            act_func='ReLU', 
            drop_rate=CONFIG["drop_rate"],
            weight_decay=True,
            weight_decay_lambda=CONFIG["weight_decay"]
        )

    # 选择优化器
    if CONFIG["use_momentum"]:
        optimizer = nn.optimizer.MomentGD(init_lr=CONFIG["init_lr"], model=model)
    else:
        optimizer = nn.optimizer.SGD(init_lr=CONFIG["init_lr"], model=model)
        
    # 根据配置条件动态选择激活 LinearLR 或保持 ConstantLR
    if CONFIG["use_scheduler"]:
        # 动态计算当前数据集在特定 BatchSize 下单轮的步数
        num_samples = train_imgs.shape[0]
        num_batches = (num_samples + CONFIG["batch_size"] - 1) // CONFIG["batch_size"]
        
        # 累乘总 Epoch，算出整个生命周期总的迭代次数 (Total Steps)
        total_steps = num_batches * CONFIG["num_epochs"]

        scheduler = nn.lr_scheduler.LinearLR(
            optimizer=optimizer, 
            total_steps=total_steps, 
            end_lr=CONFIG["end_lr"]
        )
        print(f"--> [Scheduler 激活] 识别到单轮包含 {num_batches} 个 Batch。")
        print(f"--> 学习率将在接下来总计 {total_steps} 步的训练中，从 {CONFIG['init_lr']} 线性下降至 {CONFIG['end_lr']}。")
    else:
        scheduler = nn.lr_scheduler.ConstantLR(optimizer=optimizer)
        
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=num_classes)

    runner = nn.MyRunner.MyRunner(
        model=model, 
        optimizer=optimizer, 
        metric=nn.metric.accuracy, 
        loss_fn=loss_fn, 
        batch_size=CONFIG["batch_size"], 
        scheduler=scheduler
    )

    runner.train(
        train_set, 
        dev_set, 
        num_epochs=CONFIG["num_epochs"], 
        save_dir=CONFIG["save_dir"]
    )

    MyPlot.plot_metrics(runner, model_name=CONFIG["model_type"])

    if CONFIG["use_wandb"]:
        import wandb
        wandb.finish()

if __name__ == "__main__":
    main()
