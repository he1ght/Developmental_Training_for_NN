from konlpy .tag import Okt
# okt=Okt()
# print(okt.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요!"))
import tqdm
import configargparse


def config_opts(parser):
    parser.add('-config', '--config', required=False,
               is_config_file_arg=True, help='config file path')
    parser.add('-save_config', '--save_config', required=False,
               is_write_out_config_file_arg=True,
               help='config file save path')


def tokenizer_opts(parser):
    group = parser.add_argument_group('Corpus')
    group.add('--input', '-input',
              type=str, required=True,
              help='')
    group.add('--output', '-output',
              type=str, required=True,
              help='')


def _get_parser():
    parser = configargparse.ArgumentParser()

    config_opts(parser)
    tokenizer_opts(parser)
    return parser


def main():
    parser = _get_parser()
    okt = Okt()
    opt = parser.parse_args()

    with open(opt.input, 'r') as file_in:
        with open(opt.output, 'w') as file_out:
            sents = file_in.readlines()
            tokens = []
            # cnt = 0
            for sent in tqdm.tqdm(sents):
                t = okt.morphs(sent)
                t.pop(-1)
                tok = " ".join(t)
                # tokens.append(" ".join(t))
            # for tok in tokens:
                file_out.write(tok)
                file_out.write('\n')


if __name__ == "__main__":
    main()
