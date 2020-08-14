import sys
import numpy as np
import sys

import numpy as np


def txt2dict(directory):
    f = open(directory, "r", encoding="utf8")
    vecs = []
    while True:
        line = f.readline()
        if not line:
            break
        line = line.split()
        word = line.pop(0)
        vec = [float(v) for v in line]
        vecs.append(vec)
    f.close()
    return vecs

def get_stat(vecs):
    # all_vec = []
    all_vec_length = []
    for vec in vecs:
        # all_vec += vec
        vec_length = np.sqrt(np.sum([e ** 2 for e in vec]))
        all_vec_length.append(vec_length)

    min_elm = np.min(vecs)
    max_elm = np.max(vecs)
    mean_elm = np.mean(vecs)
    std_elm = np.std(vecs)

    min_length = np.min(all_vec_length)
    max_length = np.max(all_vec_length)
    mean_length = np.mean(all_vec_length)
    std_length = np.std(all_vec_length)

    stats = {'min_element': min_elm, 'max_element': max_elm, 'mean_element': mean_elm, 'std_element': std_elm,
             'min_length': min_length, 'max_length': max_length, 'mean_length': mean_length, 'std_length': std_length}
    return stats


if __name__ == '__main__':
    vectors = txt2dict(sys.argv[1])
    # print(vectors)
    stats = get_stat(vectors)
    for k, e in stats.items():
        print(k)
        print("{:.4f}".format(e))