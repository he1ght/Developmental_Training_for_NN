import sys
import nltk
from nltk.translate.bleu_score import SmoothingFunction


def read_list_of_words(directory, ref=False):
    list_of_words = []
    f = open(directory, "r", encoding="utf8")
    while True:
        line = f.readline()
        if not line:
            break
        words = line.split()
        if ref:
            list_of_words.append([words])
        else:
            list_of_words.append(words)
    f.close()
    return list_of_words

def measure_bleu(ref, pred):
    chencherry = SmoothingFunction()
    bleu_score = nltk.translate.bleu_score.corpus_bleu(ref, pred, smoothing_function=chencherry.method4)
    return bleu_score


def measure_prec(ref, pred):
    prec_score = 0
    total = 0
    correct = 0
    for sent_refs, sent_pred in zip(ref, pred):
        sent_ref = sent_refs[0]
        for index, word in enumerate(sent_ref):
            total += 1
            try:
                if sent_pred[index] == word:
                    correct += 1
            except IndexError:
                pass
    try:
        prec_score = correct / total
    except ZeroDivisionError:
        prec_score = 0
    return prec_score

if __name__ == '__main__':
    list_of_references = read_list_of_words(sys.argv[1], ref=True)
    hypothesis = read_list_of_words(sys.argv[2])
    # list_of_references = [references]
    # list_of_hypothesis = hypothesis
    bleu_score = measure_bleu(list_of_references, hypothesis)
    prec_score = measure_prec(list_of_references, hypothesis)
    round_bleu_score = round(bleu_score, 4) * 100
    round_prec_score = round(prec_score, 4) * 100
    print("Prec.: {}".format(round_prec_score), end=' | ')
    print("BLEU : {}".format(round_bleu_score))