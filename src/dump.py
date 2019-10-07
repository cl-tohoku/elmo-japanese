import re
import json
import h5py
import numpy as np
import tensorflow as tf
from data import Batcher
from model import DynamicLanguageModel, StaticLanguageModel
from utils import load_options_latest_checkpoint


def dump_embeddings_from_dynamic_bilm(option_file,
                                      weight_file,
                                      word_file,
                                      char_file,
                                      data_file,
                                      output_file,
                                      sent_vec=False,
                                      sent_vec_type='last',
                                      cell_reset=False):
    """
    Get elmo embeddings
    """

    with open(option_file, 'r') as fin:
        options = json.load(fin)

    # add one so that 0 is the mask value
    options['char_cnn']['n_characters'] += 1

    max_word_length = options['char_cnn']['max_characters_per_token']
    batcher = Batcher(word_file, char_file, max_word_length)

    # 1D: batch_size, 2D: time_steps, 3D: max_characters_per_token
    ids_placeholder = tf.placeholder('int32', shape=(None, None, max_word_length))
    model = DynamicLanguageModel(options, weight_file, cell_reset=cell_reset)
    ops = model(ids_placeholder)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())

        print('Computing ELMo...')
        sentence_id = 0
        with open(data_file, 'r') as fin, h5py.File(output_file, 'w') as fout:
            for line in fin:
                if (sentence_id + 1) % 100 == 0:
                    print("%d" % (sentence_id + 1), flush=True, end=" ")

                sentence = line.rstrip().split()
                char_ids = batcher.batch_sentences([sentence])

                embeddings = sess.run(
                    ops['lm_embeddings'],
                    feed_dict={ids_placeholder: char_ids}
                )

                # 1D: 3(ELMo layers), 2D: n_words, 3D: vector dim
                embeddings = embeddings[0, :, :, :]
                if sent_vec:
                    embeddings = np.mean(embeddings, axis=1)
                    if sent_vec_type == 'last':
                        embeddings = embeddings[-1]
                    else:
                        embeddings = np.mean(embeddings, axis=0)
                else:
                    # 1D: n_words, 2D: 3(ELMo layers), 3D: vector dim
                    embeddings = np.transpose(embeddings, (1, 0, 2))

                fout.create_dataset(
                    name=str(sentence_id),
                    data=embeddings
                )
                sentence_id += 1
        print('Finished')


def dump_weights(tf_save_dir, output_file):
    """
    Dump the trained weights from a model to a HDF5 file.
    """

    def _get_out_name(tf_name):
        out_name = re.sub(':0$', '', tf_name)
        out_name = out_name.lstrip('lm/')
        out_name = re.sub('/rnn/', '/RNN/', out_name)
        out_name = re.sub('/multi_rnn_cell/', '/MultiRNNCell/', out_name)
        out_name = re.sub('/cell_', '/Cell', out_name)
        out_name = re.sub('/lstm_cell/', '/LSTMCell/', out_name)
        if '/RNN/' in out_name:
            if 'projection' in out_name:
                out_name = re.sub('projection/kernel', 'W_P_0', out_name)
            else:
                out_name = re.sub('/kernel', '/W_0', out_name)
                out_name = re.sub('/bias', '/B', out_name)
        return out_name

    options, ckpt_file = load_options_latest_checkpoint(tf_save_dir)

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        with tf.variable_scope('lm'):
            model = StaticLanguageModel(options, False)
            # we use the "Saver" class to load the variables
            loader = tf.train.Saver()
            loader.restore(sess, ckpt_file)

        with h5py.File(output_file, 'w') as fout:
            for v in tf.trainable_variables():
                if v.name.find('softmax') >= 0:
                    # don't dump these
                    continue
                outname = _get_out_name(v.name)
                print("Saving variable {0} with name {1}".format(
                    v.name, outname)
                )
                shape = v.get_shape().as_list()
                dset = fout.create_dataset(outname, shape, dtype='float32')
                values = sess.run([v])[0]
                dset[...] = values


def dump_weights_during_training(tf_save_dir, output_file):
    """
    Dump the trained weights from a model to a HDF5 file.
    """

    def _get_out_name(tf_name):
        _out_name = re.sub(':0$', '', tf_name)
        _out_name = _out_name.lstrip('lm/')
        _out_name = re.sub('/rnn/', '/RNN/', _out_name)
        _out_name = re.sub('/multi_rnn_cell/', '/MultiRNNCell/', _out_name)
        _out_name = re.sub('/cell_', '/Cell', _out_name)
        _out_name = re.sub('/lstm_cell/', '/LSTMCell/', _out_name)
        if '/RNN/' in _out_name:
            if 'projection' in _out_name:
                _out_name = re.sub('projection/kernel', 'W_P_0', _out_name)
            else:
                _out_name = re.sub('/kernel', '/W_0', _out_name)
                _out_name = re.sub('/bias', '/B', _out_name)
        return _out_name

    ckpt_file = tf.train.latest_checkpoint(tf_save_dir)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        with tf.variable_scope('lm'):
            # we use the "Saver" class to load the variables
            loader = tf.train.Saver()
            loader.restore(sess, ckpt_file)

        with h5py.File(output_file, 'w') as fout:
            for v in tf.trainable_variables():
                if v.name.find('softmax') >= 0:
                    # don't dump these
                    continue
                out_name = _get_out_name(v.name)
                print("Saving variable {0} with name {1}".format(
                    v.name, out_name)
                )
                shape = v.get_shape().as_list()
                dset = fout.create_dataset(out_name, shape, dtype='float32')
                values = sess.run([v])[0]
                dset[...] = values
