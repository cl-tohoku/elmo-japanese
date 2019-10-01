import os
import json

import tensorflow as tf

from data import Vocabulary, CharsVocabulary


def print_variable_summary():
    import pprint
    variables = sorted([[v.name, v.get_shape()]
                        for v in tf.global_variables()])
    pprint.pprint(variables)


def get_feed_dict_from_X(X, start, end, model, char_inputs, bidirectional):
    feed_dict = {}
    if not char_inputs:
        token_ids = X['token_ids'][start:end]
        feed_dict[model.token_ids] = token_ids
    else:
        # character inputs
        char_ids = X['tokens_characters'][start:end]
        feed_dict[model.tokens_characters] = char_ids

    if bidirectional:
        if not char_inputs:
            feed_dict[model.token_ids_reverse] = \
                X['token_ids_reverse'][start:end]
        else:
            feed_dict[model.tokens_characters_reverse] = \
                X['tokens_characters_reverse'][start:end]

    # now the targets with weights
    next_id_placeholders = [[model.next_token_id, '']]
    if bidirectional:
        next_id_placeholders.append([model.next_token_id_reverse, '_reverse'])

    for id_placeholder, suffix in next_id_placeholders:
        name = 'next_token_id' + suffix
        feed_dict[id_placeholder] = X[name][start:end]

    return feed_dict


def load_options_latest_checkpoint(save_dir):
    options_file = os.path.join(save_dir, 'options.json')
    ckpt_file = tf.train.latest_checkpoint(save_dir)

    with open(options_file, 'r') as fin:
        options = json.load(fin)

    return options, ckpt_file


def load_vocab(word_file, char_file=None, max_word_length=None):
    if max_word_length:
        return CharsVocabulary(word_file,
                               char_file,
                               max_word_length,
                               validate_file=True)
    else:
        return Vocabulary(word_file,
                          validate_file=True)
