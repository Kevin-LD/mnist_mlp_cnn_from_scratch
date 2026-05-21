import os
import gzip
import pickle
from struct import unpack
import numpy as np
from datetime import datetime
import mynn as nn
from draw_tools import MyPlot


model_type = "MLP"  # 可选: "CNN" 或 "MLP"
current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")

save_dir_name = f"{model_type.lower()}_run_{current_time_str}"
dynamic_save_dir = os.path.join("./best_models", save_dir_name)

CONFIG = {
    "model_type": model_type,   
    "seed": 309,
    "batch_size": 32,
    "num_epochs": 5,
    "init_lr": 0.06,
    "save_dir": dynamic_save_dir,
    "use_wandb": True,         
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
        wandb.init(project="deep-learning-pj2-mnist", name=f"run_{CONFIG['model_type']}", config=CONFIG)

    # 处理数据
    train_set, dev_set = load_and_preprocess_mnist(CONFIG)
    train_imgs, train_labs = train_set
    num_classes = int(train_labs.max() + 1)
    
    if CONFIG["model_type"] == "CNN":
        model = nn.models.Model_CNN()
    else:
        input_dim = train_imgs.shape[-1]
        model = nn.models.Model_MLP([input_dim, 600, num_classes], 'ReLU')

    optimizer = nn.optimizer.SGD(init_lr=CONFIG["init_lr"], model=model)
    scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
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
