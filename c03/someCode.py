"""
# File       :  someCode.py
# Time       :  2021/11/15 10:19 上午
# Author     : Qi
# Description:
"""

import numpy as np

c = np.array([[1,0,0,0,0,0,0]])
W = np.random.randn(7,3)
h = np.dot(c, W)
print(h)