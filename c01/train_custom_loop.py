"""
# File       :  train_custom_loop.py
# Time       :  2021/11/13 1:07 下午
# Author     : Qi
# Description:
"""

import sys

from common.optimizer import SGD
from c01.two_layer_net import TwoLayerNet
from dataset import spiral
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('..')

# 超参数
max_eopch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0

# 载入数据，定义优化器
x, t = spiral.load_data()
model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
optimizer = SGD(lr=learning_rate)

#
data_size = len(x)
max_iters = data_size//batch_size
total_loss = 0
loss_count = 0
loss_list = []

for epoch in range(max_eopch):
    # permutation(x) 返回 [0, x-1]的随机序列
    # 打乱数据
    idx = np.random.permutation(data_size)
    x = x[idx]
    t = t[idx]

    for iter in range(max_iters):
        batch_x = x[iter * batch_size:(iter+1) * batch_size]
        batch_t = t[iter * batch_size:(iter+1) * batch_size]

        # 计算梯度
        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)

        total_loss += loss
        loss_count += 1
        if (iter+1) % 10 == 0:
            avg_loss = total_loss / loss_count
            print('| epoch %3d |  iter %d / %d | loss %.2f'%(epoch+1, iter+1, max_iters, avg_loss))
            loss_list.append(avg_loss)
            total_loss, loss_count = 0, 0


plt.plot(np.arange(len(loss_list)), loss_list, label='train')
plt.xlabel('iterations (x10)')
plt.ylabel('loss')
plt.show()

# 绘制决策边界
h = 0.001
x_min, x_max = x[:, 0].min() - .1, x[:, 0].max() + .1
y_min, y_max = x[:, 1].min() - .1, x[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X = np.c_[xx.ravel(), yy.ravel()]
score = model.predict(X)
predict_cls = np.argmax(score, axis=1)
Z = predict_cls.reshape(xx.shape)
plt.contourf(xx, yy, Z)
plt.axis('off')

# 绘制数据点
x, t = spiral.load_data()
N = 100
CLS_NUM = 3
markers = ['o', 'x', '^']
for i in range(CLS_NUM):
    plt.scatter(x[i*N:(i+1)*N, 0], x[i*N:(i+1)*N, 1], s=40, marker=markers[i])
plt.show()