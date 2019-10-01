import sys
import time
import re

import json
import h5py
import numpy as np
import tensorflow as tf

from model import DynamicLanguageModel, MultiOutputStaticLanguageModel
from data import Batcher, BidirectionalLMDataset, LMDataset
from utils import get_feed_dict_from_X, load_options_latest_checkpoint, load_vocab


def dump_embeddings_from_dynamic_bilm(option_file,
                                      weight_file,
                                      word_file,
                                      char_file,
                                      data_file,
                                      output_file,
                                      sent_vec=False,
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
    ids_placeholder = tf.placeholder('int32',
                                     shape=(None, None, max_word_length)
                                     )
    model = DynamicLanguageModel(options, weight_file, cell_reset=cell_reset)
    ops = model(ids_placeholder)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())

        sentence_id = 0
        with open(data_file, 'r') as fin, h5py.File(output_file, 'w') as fout:
            for line in fin:
                sentence = line.strip().split()
                char_ids = batcher.batch_sentences([sentence])

                embeddings = sess.run(
                    ops['lm_embeddings'],
                    feed_dict={ids_placeholder: char_ids}
                )

                embeddings = np.transpose(embeddings[0, :, :, :], (1, 0, 2))
                if sent_vec:
                    embeddings = np.mean(np.asarray(embeddings), axis=1)
                    embeddings = np.mean(np.asarray(embeddings), axis=0)

                fout.create_dataset(
                    name=str(sentence_id),
                    data=embeddings
                )
                sentence_id += 1


def dump_embeddings_from_static_bilm(save_dir,
                                     word_file,
                                     char_file,
                                     data_file,
                                     output_file,
                                     sent_vec=False):
    """
    Get elmo embeddings
    """
    options, ckpt_file = load_options_latest_checkpoint(save_dir)

    # load the vocab
    char_inputs = 'char_cnn' in options
    if char_inputs:
        max_chars = options['char_cnn']['max_characters_per_token']
    else:
        max_chars = None
    vocab = load_vocab(word_file=word_file,
                       char_file=char_file,
                       max_word_length=max_chars)
    kwargs = {
        'test': True,
        'shuffle_on_load': False,
    }

    if options.get('bidirectional'):
        data = BidirectionalLMDataset(data_file, vocab, **kwargs)
    else:
        data = LMDataset(data_file, vocab, **kwargs)

    bidirectional = options.get('bidirectional', False)

    batch_size = 1
    unroll_steps = 20

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        with tf.device('/gpu:0'), tf.variable_scope('lm'):
            test_options = dict(options)
            # NOTE: the number of tokens we skip in the last incomplete
            # batch is bounded above batch_size * unroll_steps
            test_options['batch_size'] = batch_size
            test_options['unroll_steps'] = unroll_steps
            model = MultiOutputStaticLanguageModel(test_options, False)
            # we use the "Saver" class to load the variables
            loader = tf.train.Saver()
            loader.restore(sess, ckpt_file)

        # model.total_loss is the op to compute the loss
        # perplexity is exp(loss)
        init_state_tensors = model.init_lstm_state
        final_state_tensors = model.final_lstm_state
        if not char_inputs:
            feed_dict = {
                model.token_ids:
                    np.zeros([batch_size, unroll_steps], dtype=np.int64)
            }
            if bidirectional:
                feed_dict.update({
                    model.token_ids_reverse:
                        np.zeros([batch_size, unroll_steps], dtype=np.int64)
                })
        else:
            feed_dict = {
                model.tokens_characters:
                    np.zeros([batch_size, unroll_steps, max_chars],
                             dtype=np.int32)
            }
            if bidirectional:
                feed_dict.update({
                    model.tokens_characters_reverse:
                        np.zeros([batch_size, unroll_steps, max_chars],
                                 dtype=np.int32)
                })

        init_state_values = sess.run(init_state_tensors, feed_dict=feed_dict)

        t1 = time.time()
        batch_losses = []
        total_loss = 0.0

        sentence_id = 0
        prev_elmo = []
        outfh = h5py.File(output_file, 'w')
        for batch_no, batch in enumerate(data.iter_batches(batch_size, unroll_steps), start=1):
            # slice the input in the batch for the feed_dict

            # X: {'token_ids': (batch, unroll_steps),
            #     'tokens_characters': (batch, unroll_steps, max_characters_per_token),
            #     'next_token_id': (batch, unroll_steps)}
            X: dict = batch

            ##############
            # Set inputs #
            ##############
            feed_dict = {t: v for t, v in zip(init_state_tensors, init_state_values)}
            feed_dict.update(
                get_feed_dict_from_X(X=X,
                                     start=0,
                                     end=X['token_ids'].shape[0],
                                     model=model,
                                     char_inputs=char_inputs,
                                     bidirectional=bidirectional)
            )

            #################
            # Run the model #
            #################
            ret = sess.run(
                [model.total_loss,
                 final_state_tensors,
                 model.elmo_embs,
                 model.elmo_lstm],
                feed_dict=feed_dict
            )

            ###################
            # Get the results #
            ###################
            # elmo_embs: (2(forward/backward), batch_size, unroll_steps, dim)
            # elmo_lstm: (2(forward/backward), unroll_steps, 2(lstm layers), batch_size, dim)
            loss, init_state_values, elmo_embs, elmo_lstm = ret

            ####################
            # Make & save elmo #
            ####################
            new_elmo = _make_elmo(elmo_embs, elmo_lstm, unroll_steps)
            elmo_list, prev_elmo = _concat_elmo(prev_elmo,
                                                new_elmo,
                                                X,
                                                sent_vec)
            for elmo in elmo_list:
                outfh.create_dataset(name=str(sentence_id),
                                     data=elmo)
                sentence_id += 1

            batch_losses.append(loss)
            total_loss += loss
            avg_perplexity = np.exp(total_loss / batch_no)

            if batch_no % 100 == 0:
                sys.stdout.write("batch=%s, avg_perplexity=%s, time=%s\n" %
                                 (batch_no, avg_perplexity, time.time() - t1))
                sys.stdout.flush()

    avg_loss = np.mean(batch_losses)
    sys.stdout.write("FINISHED!  AVERAGE PERPLEXITY = %s, time=%s\n" % (np.exp(avg_loss),
                                                                        time.time() - t1))
    sys.stdout.flush()


def _make_elmo(elmo_embs, elmo_lstm, unroll_steps):
    elmo_embs = np.asarray(elmo_embs)
    # (unroll_steps, 2, dim)
    elmo_embs = np.transpose(elmo_embs, (1, 2, 0, 3))[0]
    # (unroll_steps, 1, 2 * dim)
    elmo_embs = elmo_embs.reshape((unroll_steps, 1, -1))

    elmo_lstm = np.asarray(elmo_lstm)
    # (unroll_steps, lstm_layers, 2, dim)
    elmo_lstm = np.transpose(elmo_lstm, (3, 1, 2, 0, 4))[0]
    # (unroll_steps, lstm_layers, 2 * dim)
    elmo_lstm = elmo_lstm.reshape((unroll_steps, 2, -1))

    # (unroll_steps, 3, 2 * dim)
    elmo = np.concatenate((elmo_embs, elmo_lstm), axis=1)
    return elmo


def _concat_elmo(prev_elmo, new_elmo, X, sent_vec=False):
    token_ids = X['token_ids'][0]
    elmo_list = []
    tmp_elmo = []
    assert len(new_elmo) == len(token_ids)
    for e, token_id in zip(new_elmo, token_ids):
        # token id = 0 = </S>
        # token id = 1 = <S>
        if token_id <= 1:
            sent_elmo = prev_elmo + tmp_elmo
            if sent_elmo:
                if sent_vec:
                    sent_elmo = np.mean(np.asarray(sent_elmo), axis=1)
                    sent_elmo = np.mean(np.asarray(sent_elmo), axis=0)
                    elmo_list.append(sent_elmo)
                else:
                    elmo_list.append(np.asarray(sent_elmo))
            prev_elmo = []
            tmp_elmo = []
        else:
            tmp_elmo.append(e)
    if tmp_elmo:
        prev_elmo = prev_elmo + tmp_elmo
    return elmo_list, prev_elmo


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
