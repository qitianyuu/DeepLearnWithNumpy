"""
# File       :  simple_cbow.py
# Time       :  2021/11/15 11:39 上午
# Author     : Qi
# Description: CBOW 的简单实现
"""
import sys
sys.path.append('..')
import numpy as np
from common.layers import MatMul, SoftmaxWithLoss
class SimplyCBOW:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size

        # 初始化权重
        W_in = 0.01 * np.random.randn(V,H).astype('f')
        W_out = 0.01 * np.random.randn(H,V).astype('f')

        # 层，其中输入权重共享
        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()

        # 权重和参数保存到list
        layers = [self.in_layer0, self.in_layer1, self.out_layer, self.loss_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # 分布式表示
        self.word_vecs = W_in

    def forward(self, contexts, target):
        # contexts -> (mini_batch, 2 * window_size, len(word)) (6,2,7)
        # h0       -> (mini_batch, len(word))                  (6,7) 上面第二维的第一个
        # h1       -> (mini_batch, len(word))                  (6,7) 上面第二维的第二个
        h0 = self.in_layer0.forward(contexts[:, 0])
        h1 = self.in_layer1.forward(contexts[:, 1])
        h = 0.5 * (h0 + h1)
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)
        return loss

    def backward(self, dout=1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)

        da *= 0.5
        self.in_layer0.backward(da)
        self.in_layer1.backward(da)
        return None