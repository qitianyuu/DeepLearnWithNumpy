"""
# File       :  forward_net.py
# Time       :  2021/11/13 10:22 上午
# Author     : Qi
# Description:
"""
import numpy as np

# 激活函数 sigmoid
class Sigmoid:
    def __init__(self):
        # 因为没有任何参数
        self.params = []

    def forward(self, x):
        return 1 / (1 + np.exp(-x))

class Affine:
    def __init__(self, W, b):
        # 参数为权重和偏置
        self.params = [W, b]

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        return out


# 组合两个层，实现正向推理

class TwoLayerNet:
    # 初始化 (权重初始化、创建层并初始化，参数初始化)
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        # 初始化权重, randn标准正态分布
        # (mini_batch X I) X (I X H) X (H X O) -> (mini_batch X O)
        W1 = np.random.randn(I, H)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)

        # 三层，两层仿射中间夹一层激活
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]

        # 权重
        self.params = []
        for layer in self.layers:
            self.params += layer.params

    # 前向传播
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x