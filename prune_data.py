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
    blk_cnt = 0
    while True:
        line = f.readline()
        if not line: break
        count += 1
        if line is '\n':
            blk_cnt += 1
            continue
        of.write(line)
    f.close()
    of.close()
    print("Total line: {}. {} blank lines detected.".format(count, blk_cnt))


def remove_blank_sync(input_s, output_s, input_t, output_t):
    f_s = open(input_s, "r", encoding="utf-8")
    f_t = open(input_t, "r", encoding="utf-8")

    blank_idx = []
    idx = 0
    while True:
        line = f_s.readline()
        if not line: break
        idx += 1
        if line is '\n':
            blank_idx.append(idx)
            continue
    idx = 0
    while True:
        line = f_t.readline()
        if not line: break
        idx += 1
        if line is '\n':
            blank_idx.append(idx)
            continue

    f_s.close()
    f_t.close()

    f_s = open(input_s, "r", encoding="utf-8")
    f_t = open(input_t, "r", encoding="utf-8")
    of_s = open(output_s, "w", encoding="utf-8")
    of_t = open(output_t, "w", encoding="utf-8")

    blank_idx = list(set(blank_idx))
    print("Total line: {}. {} blank lines detected.".format(idx, len(blank_idx)))

    idx = 0
    while True:
        s = f_s.readline()
        t = f_t.readline()
        if not s or not t: break
        idx += 1
        if idx in blank_idx:
            # print(idx)
            pass
        else:
            of_s.write(s)
            of_t.write(t)

    f_s.close()
    f_t.close()
    of_s.close()
    of_t.close()


def cut_data_size(limit_size, src_input, src_output, tgt_input, tgt_output):
    limit_size = int(limit_size)
    with open(src_input, "r", encoding="utf-8") as f:
        total_sent = f.readlines()
    print("get src {} sentences".format(len(total_sent)))
    rand_idx_array = random.sample(range(len(total_sent)), limit_size)
    with open(src_output, "w", encoding="utf8") as of:
        for count, i in enumerate(tqdm.tqdm(rand_idx_array)):
            of.write(total_sent[i])
    with open(tgt_input, "r", encoding="utf-8") as f:
        total_sent = f.readlines()
    print("get tgt {} sentences".format(len(total_sent)))
    with open(tgt_output, "w", encoding="utf-8") as of:
        for count, i in enumerate(tqdm.tqdm(rand_idx_array)):
            of.write(total_sent[i])
    print("{}, {} files has {} examples.".format(src_output, tgt_output, limit_size))


if __name__ == "__main__":
    cut_data_size(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
