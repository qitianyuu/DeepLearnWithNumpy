"""
# File       :  eval.py
# Time       :  2021/11/16 2:15 下午
# Author     : Qi
# Description:
"""
import sys
sys.path.append('..')
from common.util import most_similar, analogy
import pickle

pkl_file = 'cbow_params.pkl'

with open(pkl_file, 'rb') as f:
    params = pickle.load(f)
    word_vets = params['word_vecs']
    word_to_id = params['word_to_id']
    id_to_word = params['id_to_word']

# querys = ['you', 'year', 'dream', 'car', 'toyota']
#
# for query in querys:
#     most_similar(query, word_to_id, id_to_word, word_vets, 5)
analogy('king', 'man', 'queen', word_to_id, id_to_word, word_vets)
analogy('take', 'took', 'go', word_to_id, id_to_word, word_vets)
analogy('car', 'cars', 'child',  word_to_id, id_to_word, word_vets)