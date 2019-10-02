import argparse
import sys
import time
import numpy as np
import tensorflow as tf
from data import LMDataset, BidirectionalLMDataset
from model import StaticLanguageModel
from utils import get_feed_dict_from_X, load_options_latest_checkpoint, load_vocab


def test(options, ckpt_file, data, batch_size=256):
    """
    Get the test set perplexity
    """

    bidirectional = options.get('bidirectional', False)
    char_inputs = 'char_cnn' in options
    if char_inputs:
        max_chars = options['char_cnn']['max_characters_per_token']

    unroll_steps = 20

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        with tf.device('/gpu:0'), tf.variable_scope('lm'):
            test_options = dict(options)
            # NOTE: the number of tokens we skip in the last incomplete
            # batch is bounded above batch_size * unroll_steps
            test_options['batch_size'] = batch_size
            test_options['unroll_steps'] = unroll_steps
            model = StaticLanguageModel(test_options, False)
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
        for batch_no, batch in enumerate(data.iter_batches(batch_size, unroll_steps), start=1):
            # slice the input in the batch for the feed_dict
            X = batch
            feed_dict = {t: v for t, v in zip(init_state_tensors, init_state_values)}
            feed_dict.update(
                get_feed_dict_from_X(X=X,
                                     start=0,
                                     end=X['token_ids'].shape[0],
                                     model=model,
                                     char_inputs=char_inputs,
                                     bidirectional=bidirectional)
            )

            ret = sess.run(
                [model.total_loss, final_state_tensors],
                feed_dict=feed_dict
            )

            loss, init_state_values = ret
            batch_losses.append(loss)
            batch_perplexity = np.exp(loss)
            total_loss += loss
            avg_perplexity = np.exp(total_loss / batch_no)

            if batch_no % 100 == 0:
                sys.stdout.write("batch=%s, batch_perplexity=%s, avg_perplexity=%s, time=%s\n" %
                                 (batch_no, batch_perplexity, avg_perplexity, time.time() - t1))
                sys.stdout.flush()

    avg_loss = np.mean(batch_losses)
    sys.stdout.write("FINISHED!  AVERAGE PERPLEXITY = %s\n" % np.exp(avg_loss))
    sys.stdout.flush()

    return np.exp(avg_loss)


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
