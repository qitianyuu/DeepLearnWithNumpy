"""
# File       :  chow_predict.py
# Time       :  2021/11/15 10:23 上午
# Author     : Qi
# Description:
"""

import sys
sys.path.append('..')
import numpy as np
from common.layers import MatMul
from common.util import preprocess, create_contexts_target, convert_one_hot

"""
# 推理过程
# 文本数据 onehot
c0 = np.array([[1, 0, 0, 0, 0, 0, 0]])
c1 = np.array([[0, 0, 1, 0, 0, 0, 0]])

# 权重初始值
W_in = np.random.randn(7, 3)
W_out = np.random.randn(3, 7)

# 层
in_layer0 = MatMul(W_in)
in_layer1 = MatMul(W_in)
out_layer = MatMul(W_out)

# 正向传播
h0 = in_layer0.forward(c0)
h1 = in_layer1.forward(c1)
h = 0.5 * (h0 + h1)
s = out_layer.forward(h)

print(s)
"""

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
contexts, target = create_contexts_target(corpus, 1)

vocab_size = len(word_to_id)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)
print(target)