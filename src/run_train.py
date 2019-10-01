import sys
import argparse
import os
import json
from training import train
from data import BidirectionalLMDataset
from utils import load_options_latest_checkpoint, load_vocab


def main(args):
    sys.stdout.write("\nELMo Training START\n")
    sys.stdout.flush()

    os.makedirs(args.save_dir, exist_ok=True)

    if args.restart:
        options, ckpt_file = load_options_latest_checkpoint(args.save_dir)
    else:
        options = json.load(open(args.option_file))
        ckpt_file = None

    output_file = os.path.join(args.save_dir, args.output_file)

    vocab = load_vocab(word_file=args.word_file,
                       char_file=args.char_file,
                       max_word_length=options["char_cnn"]["max_characters_per_token"])

    train_data = BidirectionalLMDataset(file_path=args.train_prefix,
                                        vocab=vocab,
                                        test=False,
                                        shuffle_on_load=True)

    if args.valid_prefix:
        valid_data = BidirectionalLMDataset(file_path=args.valid_prefix,
                                            vocab=vocab,
                                            test=True,
                                            shuffle_on_load=False)
    else:
        valid_data = None

    if options["n_tokens_vocab"] == 0:
        options["n_tokens_vocab"] = vocab.size
    if options["n_train_tokens"] == 0:
        options["n_train_tokens"] = train_data.data_forward.n_tokens
    if options["char_cnn"]["n_characters"] == 0:
        options["char_cnn"]["n_characters"] = vocab.char_size

    train(options=options,
          train_data=train_data,
          valid_data=valid_data,
          n_gpus=args.n_gpus,
          tf_save_dir=args.save_dir,
          restart_ckpt_file=ckpt_file,
          output_file=output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--option_file',
                        default=None,
                        help='Option (configuration) file')
    parser.add_argument('--save_dir',
                        default='checkpoint',
                        help='Location of checkpoint files')
    parser.add_argument('--word_file',
                        default=None,
                        help='Word vocabulary file')
    parser.add_argument('--char_file',
                        default=None,
                        help='Character vocabulary file')
    parser.add_argument('--train_prefix',
                        default=None,
                        help='Prefix for train files')
    parser.add_argument('--valid_prefix',
                        default=None,
                        help='Prefix for valid files')
    parser.add_argument('--n_gpus',
                        default=4,
                        type=int,
                        help='the number of GPUs used for training')
    parser.add_argument('--restart',
                        action='store_true',
                        default=False)
    parser.add_argument('--output_file',
                        default='weight.hdf5',
                        help='Output hdf5 file with weights')

    main(parser.parse_args())
