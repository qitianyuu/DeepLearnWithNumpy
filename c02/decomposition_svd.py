"""
# File       :  decomposition_svd.py
# Time       :  2021/11/14 1:42 下午
# Author     : Qi
# Description:
"""
import matplotlib.pyplot as plt
import numpy as np

from common.util import preprocess, create_co_matrix, ppmi

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
co_matrix = create_co_matrix(corpus, len(set(corpus)))
W = ppmi(co_matrix)

# SVD
U, S, V = np.linalg.svd(W)

# print('\nC[0] -> ', co_matrix[0])
# print('\nW[0] -> ', W[0])
# print('\nU[0] -> ', U[0])
for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]))

plt.scatter(U[:,0], U[:,1], alpha=0.5)
plt.show()