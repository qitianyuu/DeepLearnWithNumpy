"""
# File       :  generate_text.py
# Time       :  2021/11/22 3:45 下午
# Author     : Qi
# Description:
"""
import sys
sys.path.append('..')
from rnnlm_gen import RnnlmGen
from dataset import ptb

corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
corpus_size = len(corpus)

model = RnnlmGen()
model.load_params('../c06/Rnnlm.pkl')

start_word = 'you'
start_id = word_to_id[start_word]
skip_word = ['N', '<unk>', '$']
skip_ids = [word_to_id[w] for w in skip_word]

word_ids = model.generate(start_id, skip_ids)
text = ' '.join([id_to_word[i] for i in word_ids])
text = text.replace(' <eos>', '.\n')
print(text)