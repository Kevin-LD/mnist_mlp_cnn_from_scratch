from .op import *
import pickle

class Model_MLP(Layer):
    def __init__(self, size_list, act_func='ReLU', drop_rate=0.0, weight_decay=False, weight_decay_lambda=1e-4):
        super().__init__()
        self.size_list = size_list
        self.act_func = act_func
        self.drop_rate = drop_rate
        
        self.layers = []
        
        # 动态构建扁平化网络层列表
        for i in range(len(size_list) - 1):
            # 1. 添加线性层
            self.layers.append(
                Linear(in_dim=size_list[i], out_dim=size_list[i + 1], initialize_method="kaiming",
                       weight_decay=weight_decay, weight_decay_lambda=weight_decay_lambda)
            )
            
            # 2. 如果不是最后一层输出层，则添加激活层和 Dropout 层
            if i < len(size_list) - 2:
                if act_func == 'ReLU':
                    self.layers.append(ReLU())
                elif act_func == 'Logistic':
                    raise NotImplementedError("Logistic 激活函数暂未实现")
                else:
                    raise ValueError(f"未知的激活函数: {act_func}")
                
                # 插入 Dropout 层
                if drop_rate > 0:
                    self.layers.append(Dropout(drop_rate=drop_rate))

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        """
        前向传播
        """
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        """
        反向传播
        """
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def train(self):
        """切换模型至训练模式（激活 Dropout）"""
        for layer in self.layers:
            if hasattr(layer, 'train'):
                layer.train()

    def eval(self):
        """切换模型至评估/测试模式（关闭 Dropout）"""
        for layer in self.layers:
            if hasattr(layer, 'eval'):
                layer.eval()

    def load_model(self, param_path):
        with open(param_path, 'rb') as f:
            param_list = pickle.load(f)
            
        idx = 0
        for layer in self.layers:
            if layer.optimizable:
                state_dict = param_list[idx]
                layer.W = state_dict['W']
                layer.b = state_dict['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = state_dict['weight_decay']
                layer.weight_decay_lambda = state_dict['lambda']
                idx += 1
        
    def save_model(self, save_path):
        param_list = []
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({
                    'W': layer.params['W'], 
                    'b': layer.params['b'], 
                    'weight_decay': layer.weight_decay, 
                    'lambda': layer.weight_decay_lambda
                })
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)        

class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self, weight_decay=False, weight_decay_lambda=1e-4) -> None:
        super().__init__()
        # 针对 28x28 的单通道 MNIST 图像设计的 CNN 架构
        # Conv1: 输入 1 通道 -> 输出 8 通道, 卷积核 3x3, 步长 2, padding 1 -> 输出尺寸: 14x14
        # Conv2: 输入 8 通道 -> 输出 16 通道, 卷积核 3x3, 步长 2, padding 1 -> 输出尺寸: 7x7
        # Flatten: 16 * 7 * 7 = 784 维向量
        # Linear: 784 -> 10 分类
        self.layers = [
            conv2D(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1, 
                   initialize_method="kaiming", weight_decay=weight_decay, weight_decay_lambda=weight_decay_lambda),
            ReLU(),
            conv2D(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1, 
                   initialize_method="kaiming", weight_decay=weight_decay, weight_decay_lambda=weight_decay_lambda),
            ReLU(),
            Flatten(),
            Linear(in_dim=784, out_dim=10, weight_decay=weight_decay,
                   initialize_method="kaiming", weight_decay_lambda=weight_decay_lambda)
        ]

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        """
        input X: [batch, 1, 28, 28] (MNIST 图像标准化后的四维张量)
        output:  [batch, 10]
        """
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        """
        loss_grad: [batch, 10] 损失函数对网络最终输出的导数
        """
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads
    
    def load_model(self, param_path):
        """
        按顺序恢复所有可训练层的权重
        """
        with open(param_path, 'rb') as f:
            param_list = pickle.load(f)
            
        idx = 0
        for layer in self.layers:
            if layer.optimizable:
                state_dict = param_list[idx]
                layer.W = state_dict['W']
                layer.b = state_dict['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = state_dict['weight_decay']
                layer.weight_decay_lambda = state_dict['lambda']
                idx += 1
        
    def save_model(self, save_path):
        param_list = []
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({
                    'W': layer.params['W'], 
                    'b': layer.params['b'], 
                    'weight_decay': layer.weight_decay, 
                    'lambda': layer.weight_decay_lambda
                })
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
