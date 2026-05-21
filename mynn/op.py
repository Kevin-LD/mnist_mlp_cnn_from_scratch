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
    def __init__(self, in_dim, out_dim, initialize_method="kaiming", weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.b = np.zeros((1, out_dim))
        # 使用 Kaiming (He) 初始化权重 W
        if initialize_method == 'kaiming':
            std = np.sqrt(2.0 / in_dim)
            self.W = np.random.normal(loc=0.0, scale=std, size=(in_dim, out_dim))
        elif initialize_method == 'normal':
            std = 0.01 
            self.W = np.random.normal(loc=0.0, scale=std, size=(in_dim, out_dim))
        else:
            raise ValueError(f"未知的初始化方法: {initialize_method}")
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
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method="kaiming", weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.b = np.zeros((out_channels, 1, 1))
        
        if initialize_method == 'kaiming':
            fan_in = in_channels * kernel_size * kernel_size
            std = np.sqrt(2.0 / fan_in)
            self.W = np.random.normal(loc=0.0, scale=std, size=(out_channels, in_channels, kernel_size, kernel_size))
            
        elif initialize_method == 'normal':
            std = 0.01 # 卷积层也使用 0.01 的标准差
            self.W = np.random.normal(loc=0.0, scale=std, size=(out_channels, in_channels, kernel_size, kernel_size))
            
        else:
            raise ValueError(f"未知的初始化方法: {initialize_method}")
            
        self.grads = {'W' : None, 'b' : None}
        self.X = None # 用于缓存输入以供反向传播使用
        self.X_pad = None # 用于缓存 Padding 后的输入

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay


    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [out, in, k, k]
        """
        self.X = X
        B, C_in, H, W = self.X.shape
        C_out, _, k, _ = self.W.shape
        
        # 处理 Padding
        if self.padding > 0:
            self.X_pad = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            self.X_pad = X
            
        H_pad, W_pad = self.X_pad.shape[2], self.X_pad.shape[3]
        
        # 计算输出的特征图尺寸
        H_out = (H_pad - k) // self.stride + 1
        W_out = (W_pad - k) // self.stride + 1
        
        # 初始化输出矩阵
        out = np.zeros((B, C_out, H_out, W_out))
        
        # 遍历输出特征图的每一个空间像素点 (h, w)
        for h in range(H_out):
            for w in range(W_out):
                h_start = h * self.stride
                h_end = h_start + k
                w_start = w * self.stride
                w_end = w_start + k
                
                # 提取当前滑动窗口下的切片: [B, C_in, k, k]
                X_slice = self.X_pad[:, :, h_start:h_end, w_start:w_end]
                
                # 利用广播机制计算点乘并对 C_in, k, k 维度求和
                # X_slice: [B, 1, C_in, k, k] * self.W: [1, C_out, C_in, k, k] -> [B, C_out, C_in, k, k]
                out[:, :, h, w] = np.sum(X_slice[:, np.newaxis, :, :, :] * self.W[np.newaxis, :, :, :, :], axis=(2, 3, 4))
                
        # 加上偏置项：[B, C_out, H_out, W_out] + [C_out, 1, 1] 触发 NumPy 的自动右对齐广播机制
        out += self.b
        return out

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        B, C_in, H, W = self.X.shape
        C_out, _, k, _ = self.W.shape
        H_out, W_out = grads.shape[2], grads.shape[3]
        
        # 1. 计算偏置项的梯度：对 Batch 维度和空间维度求和，并配平回 [C_out, 1, 1] 形状
        self.grads['b'] = np.sum(grads, axis=(0, 2, 3))[:, np.newaxis, np.newaxis]
        
        # 2. 初始化权重梯度与输入梯度
        self.grads['W'] = np.zeros_like(self.W)
        dX_pad = np.zeros_like(self.X_pad)
        
        # 遍历空间位置，将局部梯度累加
        for h in range(H_out):
            for w in range(W_out):
                h_start = h * self.stride
                h_end = h_start + k
                w_start = w * self.stride
                w_end = w_start + k
                
                # 当前位置接收到的上层梯度切片: [B, C_out]
                grad_slice = grads[:, :, h, w]
                X_slice = self.X_pad[:, :, h_start:h_end, w_start:w_end]
                
                # 计算 dW 的局部累加
                # grad_slice: [B, C_out, 1, 1, 1], X_slice: [B, 1, C_in, k, k] -> 对 Batch(0) 求和得到 [C_out, C_in, k, k]
                self.grads['W'] += np.sum(grad_slice[:, :, np.newaxis, np.newaxis, np.newaxis] * X_slice[:, np.newaxis, :, :, :], axis=0)
                
                # 计算 dX_pad 的局部累加
                # grad_slice: [B, C_out, 1, 1, 1], self.W: [1, C_out, C_in, k, k] -> 对 C_out(1) 求和得到 [B, C_in, k, k]
                dX_pad[:, :, h_start:h_end, w_start:w_end] += np.sum(grad_slice[:, :, np.newaxis, np.newaxis, np.newaxis] * self.W[np.newaxis, :, :, :, :], axis=1)
        
        # 如果开启了权重衰减，加上 L2 正则化梯度
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.W
            
        # 3. 剥离 Padding 部分，恢复出原始输入尺寸的 dX
        if self.padding > 0:
            dX = dX_pad[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dX = dX_pad
            
        return dX
    
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
    
class Flatten(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.optimizable = False
        self.orig_shape = None

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        output:  [batch, channels * H * W]
        """
        self.orig_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grad):
        """
        input grad: [batch, channels * H * W]
        output:     [batch, channels, H, W]
        """
        return grad.reshape(self.orig_shape)
    
class Dropout(Layer):
    """
    Dropout 正则化层。
    """
    def __init__(self, drop_rate=0.5) -> None:
        super().__init__()
        self.drop_rate = drop_rate
        self.mask = None
        self.optimizable = False
        self.mode = 'train'  # 默认为训练模式，可选 'train' 或 'test'

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        前向传播
        X: 任意形状的输入特征
        """
        if self.mode == 'train' and self.drop_rate > 0:
            keep_prob = 1.0 - self.drop_rate
            self.mask = (np.random.rand(*X.shape) >= self.drop_rate) / keep_prob
            return X * self.mask
        else:
            # 测试模式下，Dropout 不起作用，直接透传
            return X

    def backward(self, grads):
        """
        反向传播
        grads: 传回当前层的梯度，形状与前向传播的 X 一致
        """
        if self.mode == 'train' and self.drop_rate > 0:
            return grads * self.mask
        else:
            return grads

    def train(self):
        """切换到训练模式"""
        self.mode = 'train'

    def eval(self):
        """切换到测试/评估模式"""
        self.mode = 'test'
    
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
