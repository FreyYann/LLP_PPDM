import operator
from src.pre_process import o_config
import numpy as np


def builtTree(args, train_x):
    # train_x=train_x.astype(np.float16)
    x = train_x.copy()

    idx_ne = [x[0] for x in args.frequency[0]]
    idx_po = [x[0] for x in args.frequency[1]]

    root_ne = node()
    root_po = node()
    idxx = list(range(x.shape[1]))
    result = []
    result = build(root_ne, x, idxx, idx_ne, result)
    for row in result:
        x[row, :] = x[row, :].mean(axis=0)

    result = []
    result = build(root_po, x, idxx, idx_po, result)
    for row in result:
        x[row, :] = x[row, :].mean(axis=0)
        if o_config.is_laplace:
            for r in row:
                sensitivity = 1 / len(row)
                epsilon = o_config.epsilon
                noise = np.random.laplace(0, sensitivity / epsilon, x.shape[1])
                x[r, :] = x[r, :] + noise

    return x


def build(cur_node, x, record, idx, result):
    rec = record.copy()
    xx = x[:, rec]

    freq = xx[idx].sum(0)
    freq = dict(zip(rec, freq))
    pivot = max(freq.items(), key=operator.itemgetter(1))[0]

    rec.remove(pivot)

    if freq[pivot] <= o_config.anony_k:
        cur_node.val = idx
        result.append(idx.copy())
        return result

    if len(idx) < 2 * o_config.anony_k:
        cur_node.val = idx
        result.append(idx.copy())
        return result

    left = []
    right = []  # put 1

    for i in idx:
        if x[i][pivot]:
            right.append(i)
        else:
            left.append(i)

    if len(right) == o_config.anony_k:
        if len(left) >= o_config.anony_k:
            cur_node.right = node()
            cur_node.right.val = right
            result.append(right.copy())
            cur_node.left = node()
            result = build(cur_node.left, x, rec, left, result)
        else:
            cur_node.left = None
            cur_node.right = node()
            cur_node.right.val = left + right
            result.append(left.copy() + right.copy())

        return result

    if len(left) == o_config.anony_k:
        if len(right) >= o_config.anony_k:
            cur_node.left = node()
            cur_node.left.val = left
            result.append(left.copy())
            cur_node.right = node()
            result = build(cur_node.right, x, rec, right, result)
        else:
            cur_node.left = node()
            cur_node.left.val = left + right
            result.append(left.copy() + right.copy())
        return result

    if len(left) < o_config.anony_k:
        cur_node.left = None
        if len(right) > o_config.anony_k:
            cur_node.right = node()
            result = build(cur_node.right, x, rec, left + right, result)
        else:
            cur_node.right = node()
            cur_node.right.val = left + right
            result.append(left.copy() + right.copy())
        return result

    if len(right) < o_config.anony_k:
        cur_node.right = None
        cur_node.left = node()
        cur_node.left.val = left + right
        result.append(left.copy() + right.copy())
        return result

    cur_node.left = node()
    cur_node.right = node()
    result = build(cur_node.left, x, rec, left, result)
    result = build(cur_node.right, x, rec, right, result)

    return result


class node:
    def __init__(self):
        self.val = []
        self.left = None
        self.right = None
