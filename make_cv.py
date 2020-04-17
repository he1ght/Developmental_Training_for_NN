import random
import sys
import tqdm


def calculate_f(directory, no_blank=False):
    f = open(directory, "r", encoding="utf-8")
    count = 0

    while True:
        line = f.readline()
        if not line: break
        if no_blank and line is '\n': continue
        if line is '\n':
            raise Exception("blank!")
        count += 1
    f.close()
    return count


def pruning_data(src_input, src_output, tgt_input, tgt_output, limit_len):
    with open(src_input, "r", encoding="utf-8") as f:
        total_sent = f.readlines()
        print(len(total_sent))
    indexs = []
    for i, ct in enumerate(total_sent):
        if len(ct.split(" ")) > limit_len or not ct:
            total_sent[i] = ""
            indexs.append(i)
    print(len(indexs))
    with open(src_output, "w", encoding="utf-8") as f:
        f.write("".join(total_sent))
    with open(tgt_input, "r", encoding="utf-8") as f:
        total_sent = f.readlines()
        print(len(total_sent))

    for i in indexs:
        total_sent[i] = ""
    with open(tgt_output, "w", encoding="utf-8") as f:
        f.write("".join(total_sent))


def remove_blank(input, output):
    f = open(input, "r", encoding="utf-8")
    of = open(output, "w", encoding="utf-8")
    count = 0
    while True:
        line = f.readline()
        if not line: break
        if line is '\n': continue
        count += 1
        of.write(line)
    f.close()
    of.close()
    print(count)


def cut_data_size(valid_rate, src_input, src_output, tgt_input, tgt_output):
    valid_rate = float(valid_rate)
    with open(src_input, "r", encoding="utf-8") as f:
        total_sent = f.readlines()
    valid_size = int(len(total_sent) * valid_rate)
    train_size = len(total_sent) - valid_size

    src_train_output = src_output + "_train.txt"
    src_valid_output = src_output + "_valid.txt"
    tgt_train_output = tgt_output + "_train.txt"
    tgt_valid_output = tgt_output + "_valid.txt"

    print("get src {} sentences".format(len(total_sent)))
    rand_idx_array = random.sample(range(len(total_sent)), valid_size)
    extra_idx_array = list(set(range(len(total_sent))).difference(set(rand_idx_array)))

    with open(src_train_output, "w", encoding="utf8") as of:
        for count, i in enumerate(tqdm.tqdm(extra_idx_array)):
            of.write(total_sent[i])
    with open(src_valid_output, "w", encoding="utf8") as of:
        for count, i in enumerate(tqdm.tqdm(rand_idx_array)):
            of.write(total_sent[i])
    with open(tgt_input, "r", encoding="utf-8") as f:
        total_sent = f.readlines()
    print("get tgt {} sentences".format(len(total_sent)))
    with open(tgt_train_output, "w", encoding="utf-8") as of:
        for count, i in enumerate(tqdm.tqdm(extra_idx_array)):
            of.write(total_sent[i])
    with open(tgt_valid_output, "w", encoding="utf-8") as of:
        for count, i in enumerate(tqdm.tqdm(rand_idx_array)):
            of.write(total_sent[i])


if __name__ == "__main__":
    cut_data_size(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])