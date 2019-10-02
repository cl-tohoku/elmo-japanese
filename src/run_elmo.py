import argparse
from dump import dump_embeddings_from_dynamic_bilm, dump_weights


def main(args):
    if args.mode == 'dump_emb':
        dump_embeddings_from_dynamic_bilm(args.option_file,
                                          args.weight_file,
                                          args.word_file,
                                          args.char_file,
                                          args.data_file,
                                          args.output_file,
                                          args.sent_vec,
                                          args.sent_vec_type,
                                          args.cell_reset)
    else:
        dump_weights(args.save_dir, args.weight_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute elmo embeddings')
    parser.add_argument('--mode',
                        default='dump_emb',
                        help='dump_emb/dump_weights')
    parser.add_argument('--option_file',
                        help='Option (configuration) file')
    parser.add_argument('--weight_file',
                        default='weight.hdf5',
                        help='Output hdf5 file with weights')
    parser.add_argument('--save_dir',
                        help='Location of checkpoint files')
    parser.add_argument('--word_file',
                        help='Vocabulary file')
    parser.add_argument('--char_file',
                        help='Vocabulary file')
    parser.add_argument('--data_file',
                        help='Text file')
    parser.add_argument('--output_file',
                        default='elmo.hdf5',
                        help='Output file name')
    parser.add_argument('--sent_vec',
                        action='store_true',
                        default=False,
                        help='output sent vec')
    parser.add_argument('--sent_vec_type',
                        default='last',
                        help='last/mean')
    parser.add_argument('--cell_reset',
                        action='store_true',
                        default=False,
                        help='reset lstm cells or not')
    main(parser.parse_args())
