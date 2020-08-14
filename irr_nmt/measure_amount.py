import sys


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
    f = open(directory, "r", encoding="utf8")
    count = 0

    while True:
        line = f.readline()
        if not line:
            break
        words = line.split()
        count += len(words)
    f.close()
    return count


cnt_sen = count_sents(sys.argv[1])
cnt_tok = count_tokens(sys.argv[1])
print("{} file has {} sentences, {} tokens.".format(sys.argv[1], cnt_sen, cnt_tok))
