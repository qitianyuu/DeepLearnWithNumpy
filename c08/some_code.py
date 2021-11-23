"""
# File       :  some_code.py
# Time       :  2021/11/23 1:05 下午
# Author     : Qi
# Description:
"""
import numpy as np
from common.functions import softmax

N, T, H = 10, 5, 4
hs = np.random.randn(N, T, H)
# a = np.random.randn(N, T)
# ar = a.reshape(N, T, 1)
#
# t = hs * ar
# print(t)
#
# c = np.sum(t, axis=1)
# print(c)
h = np.random.randn(N, H)
hr = h.reshape(N, 1, H).repeat(T, axis=1)
t = hs * hr
print(t.shape)

s = np.sum(t, axis=2)
print(s.shape)

a = softmax(s)
print(a.shape)