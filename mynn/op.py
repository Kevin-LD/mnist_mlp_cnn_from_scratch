from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method='kaiming', weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        # 使用 Kaiming (He) 初始化权重 W
        if initialize_method == 'kaiming':
            std = np.sqrt(2.0 / in_dim)
            self.W = np.random.normal(loc=0.0, scale=std, size=(in_dim, out_dim))
            self.b = np.zeros((1, out_dim))
        else:
            self.W = initialize_method(size=(in_dim, out_dim))
            self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X  # 保存输入以便反向传播使用
        out = X @ self.W + self.b
        return out

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer. (这里的 input 是 grad，即损失函数对当前层输出的梯度)
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        # 计算当前层参数的梯度
        # 这里的 self.input 保存了前向传播的输入 X
        self.grads['W'] = self.input.T @ grad
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        
        # 如果开启了权重衰减（L2正则化），加上正则化项的梯度
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.W
            
        # 计算传给前一层的梯度 (dX)
        dx = grad @ self.W.T
        return dx
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        pass

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]
        no padding
        """
        pass

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        pass
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True  # 默认包含 Softmax
        self.predicts = None
        self.labels = None
        self.probs = None
        self.grads = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        self.predicts = predicts
        self.labels = labels
        
        # 根据 has_softmax 决定是否计算 softmax 概率
        if self.has_softmax:
            self.probs = softmax(predicts)
        else:
            self.probs = predicts
            
        batch_size = predicts.shape[0]
        
        # 提取出正确标签对应的预测概率，加 1e-15 防止 log(0)
        target_probs = self.probs[np.arange(batch_size), labels]
        loss = -np.mean(np.log(target_probs + 1e-15))
        return loss
    
    def backward(self):
        # first compute the grads from the loss to the input
        batch_size = self.predicts.shape[0]
        
        # 生成 one-hot 标签
        one_hot = np.zeros_like(self.probs)
        one_hot[np.arange(batch_size), self.labels] = 1.0
        
        if self.has_softmax:
            # 当 Softmax 和 CrossEntropy 结合时，对 logits 的梯度简化为: (P - Y) / N
            self.grads = (self.probs - one_hot) / batch_size
        else:
            # 如果取消了 Softmax（即输入已经是概率），对输入的梯度为: -Y / (P * N)
            self.grads = (-one_hot / (self.probs + 1e-15)) / batch_size
            
        # Then send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition
