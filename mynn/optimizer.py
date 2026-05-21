from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    # 去掉了 weight decay，改为在 backward 中计算 grad 时直接加上 weight decay 的项
                    layer.params[key] -= self.init_lr * layer.grads[key]


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu=0.9):
        super().__init__(init_lr, model)
        self.mu = mu
        self.v = {}  # 用于存储每个层、每个参数的动量

    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                # 如果是第一次访问该层，为它初始化一个动量字典
                if layer not in self.v:
                    self.v[layer] = {}
                
                for key in layer.params.keys():
                    # 如果该参数还没有对应的动量缓存，初始化为与参数形状相同的全零矩阵
                    if key not in self.v[layer]:
                        self.v[layer][key] = np.zeros_like(layer.params[key])
                    
                    # 累积动量：v = mu * v + grad
                    self.v[layer][key] = self.mu * self.v[layer][key] + layer.grads[key]
                    
                    # 更新参数：param = param - lr * v
                    layer.params[key] -= self.init_lr * self.v[layer][key]
