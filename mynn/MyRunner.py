import numpy as np
import os
import json  # 引入 JSON 模块处理元数据
from tqdm import tqdm

class MyRunner:
    """
    精简优化版训练器：每个 Epoch 独立生命周期进度条，彻底告别冗余刷屏
    """
    def __init__(self, model, optimizer, metric, loss_fn, batch_size=32, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size

        self.train_losses = []
        self.train_scores = []
        self.dev_losses = []
        self.dev_scores = []
        self.best_score = -float('inf')

    def train(self, train_set, dev_set, **kwargs):
        num_epochs = kwargs.get("num_epochs", 10)
        save_dir = kwargs.get("save_dir", "best_model")

        # 检测 W&B 状态
        has_wandb = False
        try:
            import wandb
            if wandb.run is not None:
                has_wandb = True
        except ImportError:
            pass

        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        X, y = train_set
        assert X.shape[0] == y.shape[0], "训练集特征与标签样本数不匹配！"
        num_samples = X.shape[0]
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size

        global_step = 0 
        
        # 外层 Epoch 不再套用 tqdm，保持干净
        for epoch in range(num_epochs):
            idx = np.random.permutation(num_samples)
            X, y = X[idx], y[idx]

            epoch_trn_losses = []
            epoch_trn_scores = []

            # 每个 Epoch 启动一个独立的局部进度条
            pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
            
            for iteration in pbar:
                global_step += 1
                start_idx = iteration * self.batch_size
                end_idx = min(start_idx + self.batch_size, num_samples)
                
                train_X = X[start_idx:end_idx]
                train_y = y[start_idx:end_idx]

                # 前向与反向
                logits = self.model(train_X)
                trn_loss = self.loss_fn(logits, train_y)
                trn_score = self.metric(logits, train_y)

                epoch_trn_losses.append(trn_loss)
                epoch_trn_scores.append(trn_score)

                self.loss_fn.backward()
                self.optimizer.step()
                
                if self.scheduler is not None:
                    self.scheduler.step()

                # 后台高频统计静默同步给 W&B
                if has_wandb:
                    wandb.log({
                        "train/batch_loss": float(trn_loss),
                        "train/batch_score": float(trn_score),
                        "learning_rate": float(self.optimizer.lr) if hasattr(self.optimizer, 'lr') else 0.0
                    }, step=global_step)

                pbar.set_postfix({"loss": f"{trn_loss:.4f}"})

            pbar.close()
            
            dev_score, dev_loss = self.eval(dev_set)
            
            avg_trn_loss = np.mean(epoch_trn_losses)
            avg_trn_score = np.mean(epoch_trn_scores)

            self.train_losses.append(avg_trn_loss)
            self.train_scores.append(avg_trn_score)
            self.dev_losses.append(dev_loss)
            self.dev_scores.append(dev_score)

            if has_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train/epoch_loss": float(avg_trn_loss),
                    "train/epoch_score": float(avg_trn_score),
                    "dev/loss": float(dev_loss),
                    "dev/score": float(dev_score)
                }, step=global_step)

            # 在进度条下方完整打印一行该 Epoch 的总体汇总报告
            print(f"[Epoch {epoch+1} Summary] Train Loss: {avg_trn_loss:.4f} | Train Acc: {avg_trn_score:.4f} | Dev Loss: {dev_loss:.4f} | Dev Acc: {dev_score:.4f}")

            # 权重检查与模型归档
            if dev_score > self.best_score:
                # 1. 保存常规权重文件
                save_path = os.path.join(save_dir, 'best_model.pickle')
                self.save_model(save_path)
                
                # 2. 动态提取模型元数据
                metadata = {
                    "model_meta": {
                        "model_type": self.model.__class__.__name__,
                        "saved_epoch": epoch + 1,
                        "global_step": global_step
                    },
                    "best_metrics": {
                        "dev_acc": float(dev_score),
                        "dev_loss": float(dev_loss),
                        "train_acc": float(avg_trn_score),
                        "train_loss": float(avg_trn_loss)
                    }
                }
                
                # 如果是 MLP 模型，额外把网络层级和激活函数打包存进元数据
                if hasattr(self.model, 'size_list') and self.model.size_list is not None:
                    metadata["hyperparameters"] = {
                        "size_list": self.model.size_list,
                        "act_func": str(self.model.act_func)
                    }
                
                # 保存元数据
                metadata_path = os.path.join(save_dir, 'model_metadata.json')
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=4, ensure_ascii=False)

                print(f"  --> Best score updated: {self.best_score:.4f} --> {dev_score:.4f}. Model & Metadata saved.")
                self.best_score = dev_score

    def eval(self, data_set):
        X, y = data_set
        logits = self.model(X)
        loss = self.loss_fn(logits, y)
        score = self.metric(logits, y)
        return score, loss
    
    def save_model(self, save_path):
        if hasattr(self.model, 'save_model'):
            self.model.save_model(save_path)
        else:
            import pickle
            with open(save_path, 'wb') as f:
                pickle.dump(self.model, f)
