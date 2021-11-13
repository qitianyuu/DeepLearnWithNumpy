"""
# File       :  optimizer.py
# Time       :  2021/11/13 12:46 下午
# Author     : Qi
# Description:
"""

import numpy as np

# 随机梯度下降
class SGD:
    def __init__(self, lr = 0.01):
        self.lr = lr

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]
