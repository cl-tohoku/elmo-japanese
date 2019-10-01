import argparse
from data import LMDataset, BidirectionalLMDataset
from elmo import test
from utils import load_options_latest_checkpoint, load_vocab


def main(args):
    options, ckpt_file = load_options_latest_checkpoint(args.save_dir)

    # load the vocab
    if 'char_cnn' in options:
        max_word_length = options['char_cnn']['max_characters_per_token']
    else:
        max_word_length = None
    vocab = load_vocab(word_file=args.word_file,
                       char_file=args.char_file,
                       max_word_length=max_word_length)

    test_prefix = args.test_prefix

    kwargs = {'test': True, 'shuffle_on_load': False}

    if options.get('bidirectional'):
        data = BidirectionalLMDataset(test_prefix, vocab, **kwargs)
    else:
        data = LMDataset(test_prefix, vocab, **kwargs)

    test(options, ckpt_file, data, batch_size=args.batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute test perplexity')
    parser.add_argument('--save_dir',
                        help='Location of checkpoint files')
    parser.add_argument('--word_file',
                        help='Vocabulary file')
    parser.add_argument('--char_file',
                        help='Vocabulary file')
    parser.add_argument('--test_prefix',
                        help='Prefix for test files')
    parser.add_argument('--batch_size',
                        type=int,
                        default=256,
                        help='Batch size')
    main(parser.parse_args())
