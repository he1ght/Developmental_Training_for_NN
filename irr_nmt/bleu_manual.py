def bleu_cnt(ans, tst):
    # print ans, tst

    gram_table = [{}, {}, {}, {}]
    for i, e in enumerate(ans):
        if e in gram_table[0]:
            gram_table[0][e] += 1
        else:
            gram_table[0][e] = 1

        if len(ans) > i + 1:
            e = ' '.join(ans[i:i + 2])
            if e in gram_table[1]:
                gram_table[1][e] += 1
            else:
                gram_table[1][e] = 1
        if len(ans) > i + 2:
            e = ' '.join(ans[i:i + 3])
            if e in gram_table[2]:
                gram_table[2][e] += 1
            else:
                gram_table[2][e] = 1
        if len(ans) > i + 3:
            e = ' '.join(ans[i:i + 4])
            if e in gram_table[3]:
                gram_table[3][e] += 1
            else:
                gram_table[3][e] = 1

    pred_cnt = [{}, {}, {}, {}]
    cnt = [0] * 4
    for i, e in enumerate(tst):
        if e in gram_table[0]:
            if e in pred_cnt[0]:
                pred_cnt[0][e] = min(pred_cnt[0][e] + 1, gram_table[0][e])
            else:
                pred_cnt[0][e] = 1
        if len(tst) > i + 1:
            ng = ' '.join(tst[i:i + 2])
            if ng in gram_table[1]:
                if ng in pred_cnt[1]:
                    pred_cnt[1][ng] = min(pred_cnt[1][ng] + 1, gram_table[1][ng])
                else:
                    pred_cnt[1][ng] = 1
        if len(tst) > i + 2:
            ng = ' '.join(tst[i:i + 3])
            if ng in gram_table[2]:
                if ng in pred_cnt[2]:
                    pred_cnt[2][ng] = min(pred_cnt[2][ng] + 1, gram_table[2][ng])
                else:
                    pred_cnt[2][ng] = 1
        if len(tst) > i + 3:
            ng = ' '.join(tst[i:i + 4])
            if ng in gram_table[3]:
                if ng in pred_cnt[3]:
                    pred_cnt[3][ng] = min(pred_cnt[3][ng] + 1, gram_table[3][ng])
                else:
                    pred_cnt[3][ng] = 1
    for i, tot in enumerate(pred_cnt):
        cnt[i] = sum(tot.values())
    # print 'HERE', tst, ans,pred_cnt, gram_table
    # print cnt
    return cnt


def correct_bleu(ans_list, tst_list):
    # individual score =
    #
    # precision : add all correct prediction count of each sentence
    # denomitor : add all possible ngrams of each sentences
    # chencherry method 3 smoothing for 0 precision count case
    #
    # cumulative score
    # 1,2,3,4 grams' geometric mean (p1,p2, p3, p4)^1/4

    pred_cnt = [0] * 4
    ngram_cnt = [0] * 4
    for ans, tst in zip(ans_list, tst_list):
        n = bleu_cnt(ans, tst)
        pred_cnt[0] += n[0]
        pred_cnt[1] += n[1]
        pred_cnt[2] += n[2]
        pred_cnt[3] += n[3]

        ngram_cnt[0] += max(len(tst), 0)
        ngram_cnt[1] += max(len(tst) - 1, 0)
        ngram_cnt[2] += max(len(tst) - 2, 0)
        ngram_cnt[3] += max(len(tst) - 3, 0)

    # print n, pred_cnt

    ind_bleu = [0] * 4

    k = 1
    cum_bleu = 1.0
    for i, e in enumerate(pred_cnt):
        if e == 0:
            if ngram_cnt[i] == 0:
                ind_bleu[i] = 1.0
            else:
                ind_bleu[i] = 1.0 / (ngram_cnt[i] * 2 ** k)
                k += 1
        else:
            ind_bleu[i] = pred_cnt[i] / (ngram_cnt[i] * 1.0)
        ind_bleu[i] = round(ind_bleu[i], 4)
        cum_bleu *= ind_bleu[i]

    cum_bleu = cum_bleu ** (1.0 / 4)
    # print pred_cnt, ngram_cnt
    # print "HERE", ind_bleu, cum_bleu

    return ind_bleu, cum_bleu


if __name__ == '__main__':
    # print(correct_bleu([['hi'], ['it is a tree']], [['hi'], ["it's a tree"]]))
    # a = input("1 >> ")
    # b = input("2 >> ")
    import sys
    with open(sys.argv[1], 'r', encoding='utf8') as af:
        pred_sents = af.readlines()
        for s in pred_sents:
            s.replace("\n", "")
        # print(a)
    print("A read done. Line: {}".format(len(pred_sents)))

    with open(sys.argv[2], 'r', encoding='utf8') as bf:
        true_sents = bf.readlines()
        for s in true_sents:
            s.replace("\n", "")
    print("B read done. Line: {}".format(len(true_sents)))
    print(correct_bleu(pred_sents, true_sents)[1] * 100)
    # print(correct_bleu(b, a)[1]*100)
    pass
