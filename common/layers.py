"""
# File       :  layers.py
# Time       :  2021/11/13 10:56 上午
# Author     : Qi
# Description:
"""
import numpy as np
from common.functions import sigmoid, softmax, cross_entropy_error

# 矩阵乘法类， 包含正向传播和反向传播
# 正向传播 (mini_batch X I) X (I X O) -> (mini_batch X O)
# 反向传播 到下一层的dx (mini_batch X O) X (O X I) -> (mini_batch X I)
# 反向传播 本层的dw (I X mini_batch) X (mini_batch X O) -> (I X O)
class MatMul:
    def __init__(self, W):
        # 权重参数
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        # 为什么保存输入x？ 因为计算 dw 时候需要
        self.x = None

    def forward(self, x):
        W, = self.params
        out = np.dot(x, W)
        self.x = x
        return out

    def backward(self, dout):
        W, = self.params
        dx = np.dot(dout, W.T)
        dw = np.dot(self.x.T, dout)
        # 这里[0]放dw、[1]放db，但是此时还没有偏置。
        # [...] 固定内存地址，覆盖数组
        self.grads[0][...] = dw
        return dx

# 激活函数 Sigmoid 层
class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * self.out * (1. - self.out)
        return dx


class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.loss = None
        self.y = None  # sigmoid的输出
        self.t = None  # 监督标签

    def forward(self, x, t):
        self.t = t
        self.y = 1 / (1 + np.exp(-x))

        self.loss = cross_entropy_error(np.c_[1 - self.y, self.y], self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = (self.y - self.t) * dout / batch_size
        return dx


# 全连接层类， 包含正向传播和反向传播
# 正向传播 (mini_batch X I) X (I X O) + b -> (mini_batch X O)
# 反向传播 到下一层的dx (mini_batch X O) X (O X I) -> (mini_batch X I)
# 反向传播 本层的dw (I X mini_batch) X (mini_batch X O) -> (I X O)
# 反向传播 本层的db sum(dout, axis=0) -> (O) 行求和，压缩为一行 O 列
class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        self.x = x
        return out

    def backward(self, dout):
        W, b = self.params
        dx = np.dot(dout, W.T)
        dw = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dw
        self.grads[1][...] = db
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None  # softmax的输出
        self.t = None  # 监督标签

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        # 在监督标签为one-hot向量的情况下，转换为正确解标签的索引
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size

        return dx

# 嵌入层，用于从权重参数中抽取层
class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        self.idx = idx
        W, = self.params
        out = W[idx]
        return out

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0
       #  for i, word_id in enumerate(self.idx):
       #     dW[word_id] += dout[i]
        np.add.at(dW, self.idx, dout)

        return None
