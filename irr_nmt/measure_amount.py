import math
import sys

import numpy
import pandas as pd
vocab = {}


def count_sents(directory):
    f = open(directory, "r", encoding="utf8")
    count = 0

    while True:
        line = f.readline()
        if not line:
            break
        count += 1
    f.close()
    return count


def count_tokens(directory):
    f = open(directory, "r")#, encoding="utf8")
    count = 0

    while True:
        line = f.readline()
        if not line:
            break
        words = line.split()
        count += len(words)
        for w in words:
            if w in vocab.keys():
                vocab[w] += 1
            else:
                vocab[w] = 1
    f.close()
    return count


cnt_sen = count_sents(sys.argv[1])
# cnt_tok = count_tokens(sys.argv[1])
# f = open(".".join(sys.argv[1].split('.')[:-1]) + ".csv",'w', encoding='UTF-8')
# f = open(sys.argv[1] + ".csv", 'w')
# print('{:15}{:15}'.format('Word', 'Count'))
# f.write('Word, Count\n')
# df1 = pd.DataFrame({'Word': list(vocab.keys()), 'Count': list(vocab.values())})
# for i, w in enumerate(sorted(vocab)):
#     # print('{:15}{:15}'.format(w, vocab[w]))
#     # f.write(w + ',' + str(vocab[w]) + '\n')
#     df1.loc[i+1] = [w, vocab[w]]

# f.close()
# df1.to_csv(sys.argv[1] + '.csv')
# print("{} file has {} sentences, {} tokens.".format(sys.argv[1], cnt_sen, cnt_tok))
# mean = numpy.mean(list(vocab.values()))
# std = numpy.std(list(vocab.values()))
# print("Mean: {}, Std: {}.".format(mean, std))

print("{} file has {} sentences.".format(sys.argv[1], cnt_sen))