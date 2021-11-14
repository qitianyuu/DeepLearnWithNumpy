"""
# File       :  count_method_big.py
# Time       :  2021/11/14 2:03 下午
# Author     : Qi
# Description:
"""
import dataset.ptb as ptb
import numpy as np
from common.util import create_co_matrix, ppmi, most_similat
from sklearn.utils.extmath import randomized_svd
import pickle

window_size = 2
wordvec_size = 100

corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(id_to_word)
print('counting co_matrix...')
C = create_co_matrix(corpus, vocab_size, window_size)
print('counting PPMI...')
W = ppmi(C, True)

print('calculating SVD...')
U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5, random_state=None)

word_vecs = U[:, :wordvec_size]
filename = 'wordVecs.data'
f = open(filename, 'wb')
pickle.dump(word_vecs, f)
f.close()


querys = ['you', 'year', 'car', 'plane', 'god', 'dream']

for query in querys:
    most_similat(query, word_to_id, id_to_word, word_vecs, top=5)