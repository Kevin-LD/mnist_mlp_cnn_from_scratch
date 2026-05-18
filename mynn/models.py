from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        self.layers = []
        for i in range(len(self.size_list) - 1):
            layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
            layer.W = param_list[i + 2]['W']
            layer.b = param_list[i + 2]['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = param_list[i + 2]['weight_decay']
            layer.weight_decay_lambda = param_list[i+2]['lambda']
            if self.act_func == 'Logistic':
                raise NotImplementedError
            elif self.act_func == 'ReLU':
                layer_f = ReLU()
            self.layers.append(layer)
            if i < len(self.size_list) - 2:
                self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        # 针对 28x28 的单通道 MNIST 图像设计的 CNN 架构
        # Conv1: 输入 1 通道 -> 输出 8 通道, 卷积核 3x3, 步长 2, padding 1 -> 输出尺寸: 14x14
        # Conv2: 输入 8 通道 -> 输出 16 通道, 卷积核 3x3, 步长 2, padding 1 -> 输出尺寸: 7x7
        # Flatten: 16 * 7 * 7 = 784 维向量
        # Linear: 784 -> 10 分类
        self.layers = [
            conv2D(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1, 
                   initialize_method='kaiming', weight_decay=weight_decay, weight_decay_lambda=weight_decay_lambda),
            ReLU(),
            conv2D(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1, 
                   initialize_method='kaiming', weight_decay=weight_decay, weight_decay_lambda=weight_decay_lambda),
            ReLU(),
            Flatten(),
            Linear(in_dim=784, out_dim=10, initialize_method='kaiming', 
                   weight_decay=weight_decay, weight_decay_lambda=weight_decay_lambda)
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
