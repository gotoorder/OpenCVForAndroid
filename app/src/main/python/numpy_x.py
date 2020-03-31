# -*- coding: utf-8
import numpy as np

def new_list(*args):
    return list(args)

def numpy_softmax(list):
    x_list = []

    # 遍历JAVA的list
    for i in range(list.size()):
        x_sub_list = []
        for j in range(list.get(i).size()):
            x_sub_list.append(list.get(i).get(j))
        x_list.append(x_sub_list)

    x = np.array(x_list)
    return softmax(x)

def softmax(x, axis=1):
    row_max = x.max(axis=axis)

    row_max=row_max.reshape(-1, 1)
    x = x - row_max

    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    print(type(s))
    return s.tolist()