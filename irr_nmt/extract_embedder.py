from __future__ import print_function
from __future__ import unicode_literals

import onmt.decoders.ensemble
import onmt.model_builder
from onmt.utils.parse import ArgumentParser


def config_opts(parser):
    parser.add('-config', '--config', required=False,
               is_config_file_arg=True, help='config file path')
    parser.add('-save_config', '--save_config', required=False,
               is_write_out_config_file_arg=True,
               help='config file save path')


def extractor_opts(parser):
    """ Translation / inference options """
    group = parser.add_argument_group('Model')
    group.add('--model', '-model', dest='models', metavar='MODEL',
              nargs='+', type=str, default=[], required=True,
              help="Path to model .pt file(s). "
                   "Multiple models can be specified, "
                   "for ensemble decoding.")
    group.add('--fp32', '-fp32', action='store_true',
              help="Force the model to be in FP32 "
                   "because FP16 is very slow on GTX1080(ti).")
    group.add('--avg_raw_probs', '-avg_raw_probs', action='store_true',
              help="If this is set, during ensembling scores from "
                   "different models will be combined by averaging their "
                   "raw probabilities and then taking the log. Otherwise, "
                   "the log probabilities will be averaged directly. "
                   "Necessary for models whose output layers can assign "
                   "zero probability.")

    group = parser.add_argument_group('Data')
    group.add('--data_type', '-data_type', default="text",
              help="Type of the source input. Options: [text|img].")

    # group.add('--src', '-src', required=True,
    #           help="Source sequence to decode (one line per "
    #                "sequence)")
    group.add('--src_dir', '-src_dir', default="",
              help='Source directory for image or audio files')
    group.add('--tgt', '-tgt',
              help='True target sequence (optional)')
    group.add('--shard_size', '-shard_size', type=int, default=10000,
              help="Divide src and tgt (if applicable) into "
                   "smaller multiple src and tgt files, then "
                   "build shards, each shard will have "
                   "opt.shard_size samples except last shard. "
                   "shard_size=0 means no segmentation "
                   "shard_size>0 means segment dataset into multiple shards, "
                   "each shard has shard_size samples")
    group.add('--output', '-output', default='pred.txt',
              help="Path to output the predictions (each line will "
                   "be the decoded sequence")
    group.add('--report_align', '-report_align', action='store_true',
              help="Report alignment for each translation.")
    group.add('--report_time', '-report_time', action='store_true',
              help="Report some translation time metrics")

    # Options most relevant to summarization.
    group.add('--dynamic_dict', '-dynamic_dict', action='store_true',
              help="Create dynamic dictionaries")
    group.add('--share_vocab', '-share_vocab', action='store_true',
              help="Share source and target vocabulary")

    group = parser.add_argument_group('Random Sampling')
    group.add('--random_sampling_topk', '-random_sampling_topk',
              default=1, type=int,
              help="Set this to -1 to do random sampling from full "
                   "distribution. Set this to value k>1 to do random "
                   "sampling restricted to the k most likely next tokens. "
                   "Set this to 1 to use argmax or for doing beam "
                   "search.")
    group.add('--random_sampling_temp', '-random_sampling_temp',
              default=1., type=float,
              help="If doing random sampling, divide the logits by "
                   "this before computing softmax during decoding.")
    group.add('--seed', '-seed', type=int, default=829,
              help="Random seed")

    group = parser.add_argument_group('Beam')
    group.add('--beam_size', '-beam_size', type=int, default=5,
              help='Beam size')
    group.add('--min_length', '-min_length', type=int, default=0,
              help='Minimum prediction length')
    group.add('--max_length', '-max_length', type=int, default=100,
              help='Maximum prediction length.')
    # group.add('--max_sent_length', '-max_sent_length', action=DeprecateAction,
    #           help="Deprecated, use `-max_length` instead")

    # Alpha and Beta values for Google Length + Coverage penalty
    # Described here: https://arxiv.org/pdf/1609.08144.pdf, Section 7
    group.add('--stepwise_penalty', '-stepwise_penalty', action='store_true',
              help="Apply penalty at every decoding step. "
                   "Helpful for summary penalty.")
    group.add('--length_penalty', '-length_penalty', default='none',
              choices=['none', 'wu', 'avg'],
              help="Length Penalty to use.")
    group.add('--ratio', '-ratio', type=float, default=-0.,
              help="Ratio based beam stop condition")
    group.add('--coverage_penalty', '-coverage_penalty', default='none',
              choices=['none', 'wu', 'summary'],
              help="Coverage Penalty to use.")
    group.add('--alpha', '-alpha', type=float, default=0.,
              help="Google NMT length penalty parameter "
                   "(higher = longer generation)")
    group.add('--beta', '-beta', type=float, default=-0.,
              help="Coverage penalty parameter")
    group.add('--block_ngram_repeat', '-block_ngram_repeat',
              type=int, default=0,
              help='Block repetition of ngrams during decoding.')
    group.add('--ignore_when_blocking', '-ignore_when_blocking',
              nargs='+', type=str, default=[],
              help="Ignore these strings when blocking repeats. "
                   "You want to block sentence delimiters.")
    group.add('--replace_unk', '-replace_unk', action="store_true",
              help="Replace the generated UNK tokens with the "
                   "source token that had highest attention weight. If "
                   "phrase_table is provided, it will look up the "
                   "identified source token and give the corresponding "
                   "target token. If it is not provided (or the identified "
                   "source token does not exist in the table), then it "
                   "will copy the source token.")
    group.add('--phrase_table', '-phrase_table', type=str, default="",
              help="If phrase_table is provided (with replace_unk), it will "
                   "look up the identified source token and give the "
                   "corresponding target token. If it is not provided "
                   "(or the identified source token does not exist in "
                   "the table), then it will copy the source token.")
    group = parser.add_argument_group('Logging')
    group.add('--verbose', '-verbose', action="store_true",
              help='Print scores and predictions for each sentence')
    group.add('--log_file', '-log_file', type=str, default="",
              help="Output logs to a file under this path.")
    group.add('--attn_debug', '-attn_debug', action="store_true",
              help='Print best attn for each word')
    group.add('--align_debug', '-align_debug', action="store_true",
              help='Print best align for each word')
    group.add('--dump_beam', '-dump_beam', type=str, default="",
              help='File to dump beam information to.')
    group.add('--n_best', '-n_best', type=int, default=1,
              help="If verbose is set, will output the n_best "
                   "decoded sentences")

    group = parser.add_argument_group('Efficiency')
    group.add('--batch_size', '-batch_size', type=int, default=30,
              help='Batch size')
    group.add('--batch_type', '-batch_type', default='sents',
              choices=["sents", "tokens"],
              help="Batch grouping for batch_size. Standard "
                   "is sents. Tokens will do dynamic batching")
    group.add('--gpu', '-gpu', type=int, default=-1,
              help="Device to run on")

    # Options most relevant to speech.
    group = parser.add_argument_group('Speech')
    group.add('--sample_rate', '-sample_rate', type=int, default=16000,
              help="Sample rate.")
    group.add('--window_size', '-window_size', type=float, default=.02,
              help='Window size for spectrogram in seconds')
    group.add('--window_stride', '-window_stride', type=float, default=.01,
              help='Window stride for spectrogram in seconds')
    group.add('--window', '-window', default='hamming',
              help='Window type for spectrogram generation')

    # Option most relevant to image input
    group.add('--image_channel_size', '-image_channel_size',
              type=int, default=3, choices=[3, 1],
              help="Using grayscale image can training "
                   "model faster and smaller")


def _get_parser():
    parser = ArgumentParser(description='extract_embedder.py')

    config_opts(parser)
    extractor_opts(parser)
    return parser


if __name__ == "__main__":
    parser = _get_parser()

    opt = parser.parse_args()
    opt.gpu = -1

    load_test_model = onmt.decoders.ensemble.load_test_model \
        if len(opt.models) > 1 else onmt.model_builder.load_test_model
    fields, model, model_opt = load_test_model(opt)

    # Encoder
    enc_emb_cover = model.encoder.embeddings
    enc_field = fields['src'].fields[0][1]
    enc_emb = enc_emb_cover.make_embedding[0][0]

    with open('encoder_emb.txt', 'w') as f:
        for n_word in range(len(enc_field.vocab.itos)):
            line = ""
            line += enc_field.vocab.itos[n_word]
            tt = 0
            for n_vec in range(len(enc_emb.weight[n_word])):
                line += " " + str(enc_emb.weight[n_word][n_vec].item())
            line += "\n"
            f.write(line)

    # Decoder
    dec_emb_cover = model.decoder.embeddings
    dec_field = fields['tgt'].fields[0][1]
    dec_emb = dec_emb_cover.make_embedding[0][0]

    with open('decoder_emb.txt', 'w') as f:
        for n_word in range(len(dec_field.vocab.itos)):
            line = ""
            line += dec_field.vocab.itos[n_word]
            for n_vec in range(len(dec_emb.weight[n_word])):
                line += " " + str(dec_emb.weight[n_word][n_vec].item())
            line += "\n"
            f.write(line)