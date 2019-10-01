import sys
import os
import time
import json

import tensorflow as tf
import numpy as np

from dump import dump_weights_during_training
from model import StaticLanguageModel
from utils import print_variable_summary, get_feed_dict_from_X

DTYPE = 'float32'
DTYPE_INT = 'int64'

tf.logging.set_verbosity(tf.logging.INFO)


def average_gradients(tower_grads, batch_size, options):
    # calculate average gradient for each shared variable across all GPUs
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        # We need to average the gradients across each GPU.

        g0, v0 = grad_and_vars[0]

        if g0 is None:
            # no gradient for this variable, skip it
            average_grads.append((g0, v0))
            continue

        if isinstance(g0, tf.IndexedSlices):
            # If the gradient is type IndexedSlices then this is a sparse
            #   gradient with attributes indices and values.
            # To average, need to concat them individually then create
            #   a new IndexedSlices object.
            indices = []
            values = []
            for g, v in grad_and_vars:
                indices.append(g.indices)
                values.append(g.values)
            all_indices = tf.concat(indices, 0)
            avg_values = tf.concat(values, 0) / len(grad_and_vars)
            # deduplicate across indices
            av, ai = _deduplicate_indexed_slices(avg_values, all_indices)
            grad = tf.IndexedSlices(av, ai, dense_shape=g0.dense_shape)

        else:
            # a normal tensor can just do a simple average
            grads = []
            for g, v in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)
                # Append on a 'tower' dimension which we will average over 
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

        # the Variables are redundant because they are shared
        # across towers. So.. just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)

        average_grads.append(grad_and_var)

    assert len(average_grads) == len(list(zip(*tower_grads)))

    return average_grads


def summary_gradient_updates(grads, opt, lr):
    """
    get summary ops for the magnitude of gradient updates
    """

    # strategy:
    # make a dict of variable name -> [variable, grad, adagrad slot]
    vars_grads = {}
    for v in tf.trainable_variables():
        vars_grads[v.name] = [v, None, None]
    for g, v in grads:
        vars_grads[v.name][1] = g
        vars_grads[v.name][2] = opt.get_slot(v, 'accumulator')

    # now make summaries
    ret = []
    for vname, (v, g, a) in vars_grads.items():

        if g is None:
            continue

        if isinstance(g, tf.IndexedSlices):
            # a sparse gradient - only take norm of params that are updated
            values = tf.gather(v, g.indices)
            updates = lr * g.values
            if a is not None:
                updates /= tf.sqrt(tf.gather(a, g.indices))
        else:
            values = v
            updates = lr * g
            if a is not None:
                updates /= tf.sqrt(a)

        values_norm = tf.sqrt(tf.reduce_sum(v * v)) + 1.0e-7
        updates_norm = tf.sqrt(tf.reduce_sum(updates * updates))
        ret.append(
            tf.summary.scalar('UPDATE/' + vname, updates_norm / values_norm))

    return ret


def _deduplicate_indexed_slices(values, indices):
    """
    Sums `values` associated with any non-unique `indices`.
    Args:
      values: A `Tensor` with rank >= 1.
      indices: A one-dimensional integer `Tensor`, indexing into the first
      dimension of `values` (as in an IndexedSlices object).
    Returns:
      A tuple of (`summed_values`, `unique_indices`) where `unique_indices` is a
      de-duplicated version of `indices` and `summed_values` contains the sum of
      `values` slices associated with each unique index.
    """
    unique_indices, new_index_positions = tf.unique(indices)
    summed_values = tf.unsorted_segment_sum(
        values, new_index_positions,
        tf.shape(unique_indices)[0])
    return summed_values, unique_indices


def clip_by_global_norm_summary(t_list, clip_norm, norm_name, variables):
    # wrapper around tf.clip_by_global_norm that also does summary ops of norms

    # compute norms
    # use global_norm with one element to handle IndexedSlices vs dense
    norms = [tf.global_norm([t]) for t in t_list]

    # summary ops before clipping
    summary_ops = []
    for ns, v in zip(norms, variables):
        name = 'norm_pre_clip/' + v.name
        summary_ops.append(tf.summary.scalar(name, ns))

    # clip 
    clipped_t_list, tf_norm = tf.clip_by_global_norm(t_list, clip_norm)

    # summary ops after clipping
    norms_post = [tf.global_norm([t]) for t in clipped_t_list]
    for ns, v in zip(norms_post, variables):
        name = 'norm_post_clip/' + v.name
        summary_ops.append(tf.summary.scalar(name, ns))

    summary_ops.append(tf.summary.scalar(norm_name, tf_norm))

    return clipped_t_list, tf_norm, summary_ops


def clip_grads(grads, options, do_summaries, global_step):
    # grads = [(grad1, var1), (grad2, var2), ...]
    def _clip_norms(grad_and_vars, val, name):
        # grad_and_vars is a list of (g, v) pairs
        grad_tensors = [g for g, v in grad_and_vars]
        vv = [v for g, v in grad_and_vars]
        scaled_val = val
        if do_summaries:
            clipped_tensors, g_norm, so = clip_by_global_norm_summary(
                grad_tensors, scaled_val, name, vv)
        else:
            so = []
            clipped_tensors, g_norm = tf.clip_by_global_norm(
                grad_tensors, scaled_val)

        ret = []
        for t, (g, v) in zip(clipped_tensors, grad_and_vars):
            ret.append((t, v))

        return ret, so

    all_clip_norm_val = options['all_clip_norm_val']
    ret, summary_ops = _clip_norms(grads, all_clip_norm_val, 'norm_grad')

    assert len(ret) == len(grads)

    return ret, summary_ops


def train(options,
          train_data,
          valid_data,
          n_gpus,
          tf_save_dir,
          restart_ckpt_file=None,
          output_file=None):

    ####################
    # Save the options #
    ####################
    if restart_ckpt_file is None:
        with open(os.path.join(tf_save_dir, 'options.json'), 'w') as fout:
            fout.write(json.dumps(options))

    #################
    # Set variables #
    #################
    with tf.device('/cpu:0'):
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        # set up the optimizer
        lr = options.get('learning_rate', 0.2)
        opt = tf.train.AdagradOptimizer(learning_rate=lr,
                                        initial_accumulator_value=1.0)

        # calculate the gradients on each GPU
        tower_grads = []
        models = []
        train_perplexity = tf.get_variable(
            'train_perplexity', [],
            initializer=tf.constant_initializer(0.0), trainable=False)
        norm_summaries = []
        for k in range(n_gpus):
            with tf.device('/gpu:%d' % k):
                with tf.variable_scope('lm', reuse=k > 0):
                    # calculate the loss for one model replica and get
                    #   lstm states
                    model = StaticLanguageModel(options, True)
                    loss = model.total_loss
                    models.append(model)
                    # get gradients
                    grads = opt.compute_gradients(
                        loss * options['unroll_steps'],
                        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE,
                    )
                    tower_grads.append(grads)
                    # keep track of loss across all GPUs
                    train_perplexity += loss

        print_variable_summary()

        # calculate the mean of each gradient across all GPUs
        grads = average_gradients(tower_grads, options['batch_size'], options)
        grads, norm_summary_ops = clip_grads(grads, options, True, global_step)
        norm_summaries.extend(norm_summary_ops)

        # log the training perplexity
        train_perplexity = tf.exp(train_perplexity / n_gpus)
        perplexity_summary = tf.summary.scalar(
            'train_perplexity', train_perplexity)

        # some histogram summaries.  all models use the same parameters
        # so only need to summarize one
        histogram_summaries = [
            tf.summary.histogram('token_embedding', models[0].embedding)
        ]

        # apply the gradients to create the training operation
        train_op = opt.apply_gradients(grads, global_step=global_step)

        # histograms of variables
        for v in tf.global_variables():
            histogram_summaries.append(tf.summary.histogram(v.name, v))

        # get the gradient updates -- these aren't histograms, but we'll
        # only update them when histograms are computed
        histogram_summaries.extend(
            summary_gradient_updates(grads, opt, lr))

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
        summary_op = tf.summary.merge(
            [perplexity_summary] + norm_summaries
        )
        hist_summary_op = tf.summary.merge(histogram_summaries)

        ##################
        # Initialization #
        ##################
        init = tf.initialize_all_variables()

    ########################
    # do the training loop #
    ########################
    bidirectional = options.get('bidirectional', False)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # Build a computational graph
        sess.run(init)

        ############################
        # Load the checkpoint data #
        ############################
        if restart_ckpt_file is not None:
            # variables_except_softmax = [v for v in tf.global_variables() if "softmax" not in v.name]
            # loader = tf.train.Saver(variables_except_softmax)
            loader = tf.train.Saver()
            loader.restore(sess, restart_ckpt_file)

        # For each batch:
        # Get a batch of data from the generator. The generator will
        # yield batches of size batch_size * n_gpus that are sliced
        # and fed for each required placeholder.
        #
        # We also need to be careful with the LSTM states.  We will
        # collect the final LSTM states after each batch, then feed
        # them back in as the initial state for the next batch
        batch_size = options['batch_size']
        unroll_steps = options['unroll_steps']
        n_train_tokens = options.get('n_train_tokens', 768648884)
        n_tokens_per_batch = batch_size * unroll_steps * n_gpus
        n_batches_per_epoch = int(n_train_tokens / n_tokens_per_batch)
        n_batches_total = options['n_epochs'] * n_batches_per_epoch
        sys.stdout.write("\n\nTraining for %s epochs and %s batches\n\n" % (
            options['n_epochs'], n_batches_total)
        )
        sys.stdout.flush()

        ###############################
        # Get the initial lstm states #
        ###############################
        init_state_tensors = []
        final_state_tensors = []
        for model in models:
            init_state_tensors.extend(model.init_lstm_state)
            final_state_tensors.extend(model.final_lstm_state)

        char_inputs = 'char_cnn' in options
        if char_inputs:
            max_chars = options['char_cnn']['max_characters_per_token']
        else:
            max_chars = None

        ##################
        # Make feed dict #
        ##################
        if not char_inputs:
            feed_dict = {
                model.token_ids:
                    np.zeros([batch_size, unroll_steps], dtype=np.int64)
                for model in models
            }
        else:
            feed_dict = {
                model.tokens_characters:
                    np.zeros([batch_size, unroll_steps, max_chars],
                             dtype=np.int32)
                for model in models
            }

        if bidirectional:
            if not char_inputs:
                feed_dict.update({
                    model.token_ids_reverse:
                        np.zeros([batch_size, unroll_steps], dtype=np.int64)
                    for model in models
                })
            else:
                feed_dict.update({
                    model.tokens_characters_reverse:
                        np.zeros([batch_size, unroll_steps, max_chars],
                                 dtype=np.int32)
                    for model in models
                })

        ##############
        # Initialize #
        ##############
        init_state_values = sess.run(init_state_tensors, feed_dict=feed_dict)

        ######################
        # Main training loop #
        ######################
        t1 = time.time()
        best_perplexity = 1000000.0
        data_gen = train_data.iter_batches(batch_size=batch_size * n_gpus,
                                           num_steps=unroll_steps)

        for batch_no, batch in enumerate(data_gen, start=1):
            # slice the input in the batch for the feed_dict
            X = batch
            feed_dict = {t: v for t, v in zip(init_state_tensors, init_state_values)}
            for k in range(n_gpus):
                model = models[k]
                start = k * batch_size
                end = (k + 1) * batch_size

                feed_dict.update(
                    get_feed_dict_from_X(X, start, end, model, char_inputs, bidirectional)
                )

            # This runs the train_op, summaries and the "final_state_tensors"
            #   which just returns the tensors, passing in the initial
            #   state tensors, token ids and next token ids
            if batch_no % 1250 != 0:
                ret = sess.run(
                    [train_op, summary_op, train_perplexity] +
                    final_state_tensors,
                    feed_dict=feed_dict
                )

                # first three entries of ret are: train_op, summary_op, train_perplexity.
                # last entries are the final states
                #   -- set them to init_state_values for next batch
                init_state_values = ret[3:]
            else:
                # also run the histogram summaries
                ret = sess.run(
                    [train_op, summary_op, train_perplexity, hist_summary_op] +
                    final_state_tensors,
                    feed_dict=feed_dict
                )
                init_state_values = ret[4:]

            if (batch_no % 1250 == 0) or (batch_no == n_batches_total):
                sys.stdout.write("Batch %s, train_perplexity=%s\n" % (batch_no, ret[2]))
                sys.stdout.write("Total time: %s\n" % (time.time() - t1))
                sys.stdout.flush()

                if valid_data:
                    valid_perplexity = validate(sess=sess,
                                                model=models[0],
                                                options=options,
                                                data=valid_data,
                                                batch_size=batch_size)
                    if valid_perplexity < best_perplexity:
                        best_perplexity = valid_perplexity
                        checkpoint_path = os.path.join(tf_save_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=global_step)
                else:
                    checkpoint_path = os.path.join(tf_save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=global_step)

            if batch_no == n_batches_total:
                # done training!
                break

    dump_weights_during_training(tf_save_dir, output_file)


def validate(sess, model, options, data, batch_size=256):
    """
    Get the test set perplexity!
    """
    sys.stdout.write("\n\n####################\n")
    sys.stdout.write("VALIDATING THE MODEL\n")
    sys.stdout.flush()

    bidirectional = options.get('bidirectional', False)
    unroll_steps = options.get('unroll_steps')

    char_inputs = 'char_cnn' in options
    if char_inputs:
        max_chars = options['char_cnn']['max_characters_per_token']

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

    batch_losses = []

    data.reload_all_shards()
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

    avg_loss = np.mean(batch_losses)
    sys.stdout.write("\nAVERAGE PERPLEXITY = %s\n" % np.exp(avg_loss))
    sys.stdout.write("####################\n\n")
    sys.stdout.flush()

    return np.exp(avg_loss)

