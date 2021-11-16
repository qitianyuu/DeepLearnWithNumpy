"""
# File       :  cbow.py
# Time       :  2021/11/16 10:53 上午
# Author     : Qi
# Description:
"""
import sys
sys.path.append('..')
import numpy as np
from common.layers import Embedding
from c04.negative_sample_layer import NegativeSamplingLoss

class CBOW:
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        V, H = vocab_size, hidden_size

        #  初始化权重
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(V, H).astype('f')

        # 生成层
        self.in_layers = []
        # 两倍的窗口大小，前后都要算
        for i in range(2 * window_size):
            layer = Embedding(W_in)
            self.in_layers.append(layer)
        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)

        # 整理所有权重和参数列表
        layers = self.in_layers + [self.ns_loss]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        self.word_vecs = W_in


    def forward(self, contexts, target):
        h = 0
        # 循环 window_size * 2 次，求得上下文的所有输入层的结果的和
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:, i])

        h *= 1/len(self.in_layers)

        loss = self.ns_loss.forward(h, target)
        return loss

    def backward(self, dout=1):
        dout = self.ns_loss.backward(dout)
        dout *= 1/len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)

        return None
