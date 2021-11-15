"""
# File       :  simple_skip_gram.py
# Time       :  2021/11/15 3:05 下午
# Author     : Qi
# Description:
"""

import sys
sys.path.append('..')
import numpy as np
from common.layers import MatMul, SoftmaxWithLoss

class SimpleSkipGram:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size

        # 权重初始化
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(H, V).astype('f')

        # 层
        self.in_layer = MatMul(W_in)
        self.out_laye = MatMul(W_out)
        self.loss_layer0 = SoftmaxWithLoss()
        self.loss_layer1 = SoftmaxWithLoss()

        # 权重参数保存到list
        layers = [self.in_layer, self.out_laye]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        self.word_vecs = W_in

    def forward(self, contexts, target):
        h = self.in_layer.forward(target)
        s = self.out_laye.forward(h)
        l1 = self.loss_layer0.forward(s, contexts[:, 0])
        l2 = self.loss_layer1.forward(s, contexts[:, 1])
        loss = l1 + l2
        return loss

    def backward(self, dout=1):
        dl1 = self.loss_layer0.backward(dout)
        dl2 = self.loss_layer1.backward(dout)
        ds = dl1 + dl2
        dh = self.out_laye.backward(ds)
        self.in_layer.backward(dh)
        return None
