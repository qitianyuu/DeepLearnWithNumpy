"""
# File       :  someCode.py
# Time       :  2021/11/14 10:20 上午
# Author     : Qi
# Description:
"""
import numpy as np

text = 'You say goodbye and I say hello.'


from common.util import preprocess, create_co_matrix, cos_similarity, most_similar, ppmi

corpus, word_to_id, id_to_word = preprocess(text)

# print('\nwords 2 id -> ', word_to_id)
# print('\nid 2 words -> ', id_to_word)
# print('\ncorpus -> ', corpus)


"""
# 预处理部分，完整的方法包装在 common/util.py 文件中
# 小写
text = text.lower()
# 处理末尾的句号
text = text.replace('.', ' .')
print('text -> ', text)
# 按照空格进行分割
words = text.split(' ')
print('\nwords list -> ', words)

word_to_id = {}
id_to_word = {}

for word in words:
    if word not in word_to_id:
        new_id = len(word_to_id)
        word_to_id[word] = new_id
        id_to_word[new_id] = word

print('\nwords 2 id -> ', word_to_id)
print('\nid 2 words -> ', id_to_word)

# corpus = [word_to_id[w] for w in words]
# corpus = np.array(corpus)

print('\ncorpus -> ', corpus)
"""
co_matrix = create_co_matrix(corpus, 7, 1)
# 求you 和 i 的余弦相似度
most_similar('you', word_to_id, id_to_word, co_matrix)
#
M = ppmi(co_matrix)
# 设置3位有效数位
np.set_printoptions(precision=3)
print('\nM -> \n', M)


