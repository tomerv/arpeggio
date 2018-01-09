import numpy as np
import json
import os
import sys
from datetime import datetime
import time
from collections import defaultdict
sys.path.append('/home/tvromen/research')
from Common.Utils import IdAssigner, print_flush
from Common.RatingsData import RatingsData, remove_extreme_users, constant_user_length
import tensorflow as tf
import matplotlib.pyplot as plt
from occf import calc_scores

class Flags(object):
    def __init__(self):
        # Model Hyperparameters
        self.embedding_dim = 10
        self.alpha = 0.1
        self.reg_lambda = 0.000001

        # Training parameters
        self.training_stop_after = None
        self.batch_size = 1024
        self.num_epochs = 10
        # self.num_epochs = 2000
        self.summary_every = 100
        self.evaluate_every = 1000
        self.checkpoint_every = 2000
        self.num_checkpoints = 3

        # Misc Parameters
        self.allow_soft_placement = True
        self.log_device_placement = False

FLAGS = Flags()


np.random.seed(1234)


###########
# Dataset #
###########

if len(sys.argv) != 2:
    print('Usage: python3 ours2.py <dataset_name>')
    exit(1)

dataset_name = sys.argv[1]

ratings = RatingsData.from_files(dataset_name+'.train.txt', dataset_name+'.val.txt')
print_flush('Num users: {}'.format(ratings.num_users))
print_flush('Num items: {}'.format(ratings.num_items))

print_flush("Train/Val/Test split: {}/{}/{}".format(
    len(ratings.train), len(ratings.val), len(ratings.test)
))

users_train = set(ratings.train['user_id'])
users_val   = set(ratings.val['user_id'])
print_flush('# users in train set: {}'.format(len(users_train)))
print_flush('# users in val set: {}'.format(len(users_val)))
print_flush('# users in val set not in train set: {}'.format(len(users_val - users_train)))


#############
# The model #
#############

def get_dynamic_tensor_shape(x):
    """
    Calculate the tensor shape. Use a plain number where possible and a tensor elsewhere.
    x is a tensor of some shape.
    returns a list with the dimensions of x.
    """
    shape_tensor = tf.shape(x)
    shape = list(x.get_shape())
    for i in range(len(shape)):
        shape[i] = shape[i].value
        if shape[i] is None:
            # use tensor to represent the dimension
            shape[i] = shape_tensor[i]
    return shape

def embedding_lookup_layer(x, vocab_size, embedding_dim, variable_scope, reuse=False, zero_init=False):
    """
    Lookup embedding
    x is tensor of shape (d_1, d_2, ..., d_n) and type int32
    result is tensor of shape (d_1, d_2, ..., d_n, embedding_dim) of n+1 dimensions and type DT_FLOAT
    """
    initializer = tf.contrib.layers.xavier_initializer() if not zero_init else tf.zeros_initializer()
    with tf.variable_scope(variable_scope, reuse=reuse):
        W = tf.get_variable(
            'W',
            shape=[vocab_size, embedding_dim],
            initializer=initializer,
            regularizer=tf.contrib.layers.l2_regularizer(1.)
        )
    x_embedded = tf.nn.embedding_lookup(W, x)
    return x_embedded

def bias_lookup_layer(x, vocab_size, variable_scope, reuse=False):
    """
    Lookup bias
    x is tensor of shape (d_1, d_2, ..., d_n) and type int32
    result is tensor of same shape as x and type DT_FLOAD
    """
    with tf.variable_scope(variable_scope, reuse=reuse):
        b = tf.get_variable(
            'b',
            shape=[vocab_size, 1],
            initializer=tf.zeros_initializer() # ,
            # TODO: does bias need regularization?
            # regularizer=tf.contrib.layers.l2_regularizer(1.)
        )
    x_bias = tf.squeeze(tf.nn.embedding_lookup(b, x), -1)
    return x_bias

def fc_layer(x, output_size, variable_scope, reuse=False):
    """
    Fully-connected layer
    x has shape (batch_size, d_2)
    result has shape (batch_size, output_size)
    """
    shape = get_dynamic_tensor_shape(x)
    assert len(shape) == 2
    ## TODO: regularization
    with tf.variable_scope(variable_scope, reuse=reuse):
        W = tf.get_variable(
            "W",
            shape=[shape[1], output_size],
            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(
            "b",
            shape=[output_size],
            initializer=tf.contrib.layers.xavier_initializer())
    result = tf.nn.xw_plus_b(x, W, b, name="fc")
    return result

class PredictionModel(object):
    """
    A neural network for predicting per-user item ratings.
    The input to the network is the user_id and item_id.
    """
    def __init__(self, num_users, num_items, num_ratings, embedding_dim, mu, alpha, reg_lambda):

        assert num_users >= 1
        self.num_users = num_users
        assert num_items >= 1
        self.num_items = num_items
        assert num_ratings >= 1
        self.num_ratings = num_ratings
        assert embedding_dim >= 1
        self.embedding_dim = embedding_dim
        assert reg_lambda >= 0

        # Placeholders for input, output and dropout
        self.input_user_ids = tf.placeholder(tf.int32, [None], name="input_user_ids")
        self.input_per_user_count    = tf.placeholder(tf.int32,   [None], name="input_per_user_count")
        self.input_per_user_item_ids = tf.placeholder(tf.int32,   [None, None], name="input_per_user_item_ids")
        self.input_per_user_ratings  = tf.placeholder(tf.float32, [None, None], name="input_per_user_ratings")
        self.input_per_user_neg_ids  = tf.placeholder(tf.int32,   [None, None], name="input_per_user_neg_ids")
    
        num_users = tf.shape(self.input_user_ids)[0]
        batch_size = tf.reduce_sum(self.input_per_user_count)
        asrt1 = tf.assert_equal(num_users, tf.shape(self.input_per_user_count)[0])
        asrt2 = tf.assert_equal(num_users, tf.shape(self.input_per_user_item_ids)[0])
        asrt3 = tf.assert_equal(num_users, tf.shape(self.input_per_user_ratings)[0])
        asrt4 = tf.assert_equal(num_users, tf.shape(self.input_per_user_neg_ids)[0])

        # pu = per_user

        pu_mask = tf.sequence_mask(self.input_per_user_count, dtype=tf.float32)

        # embedding lookup layer
        with tf.device('/cpu:0'), tf.name_scope('embedding_lookup'), tf.control_dependencies([asrt1, asrt2, asrt3, asrt4]):
            # get dimension of user_ids to match the per_user_* stuff
            user_em = embedding_lookup_layer(self.input_user_ids, self.num_users, self.embedding_dim, 'user_embedding')
            user_bias = bias_lookup_layer(self.input_user_ids, self.num_users, 'user_embedding')
            expanded_user_em = tf.expand_dims(user_em, 1)
            expanded_user_bias = tf.expand_dims(user_bias, 1)
            pu_item_em = embedding_lookup_layer(self.input_per_user_item_ids, self.num_items, self.embedding_dim, 'item_embedding')
            pu_neg_em  = embedding_lookup_layer(self.input_per_user_neg_ids,  self.num_items, self.embedding_dim, 'item_embedding', reuse=True)
            pu_item_bias = bias_lookup_layer(self.input_per_user_item_ids, self.num_items, 'item_embedding')
            pu_neg_bias  = bias_lookup_layer(self.input_per_user_neg_ids,  self.num_items, 'item_embedding', reuse=True)

        with tf.name_scope('bpr'):
            pu_em_delta   = pu_item_em - pu_neg_em
            pu_bias_delta = pu_item_bias - pu_neg_bias
            pu_prediction_delta = tf.reduce_sum(expanded_user_em * pu_em_delta, axis=-1) + pu_bias_delta
            # self.bpr_loss = pu_mask * tf.log(tf.sigmoid(-pu_prediction_delta) + 0.01)
            self.bpr_loss = pu_mask * tf.sigmoid(-pu_prediction_delta)

        # PMF (a.k.a. "SVD for recommender systems) part
        with tf.name_scope('rating'):
            user_rating_em = tf.tanh(fc_layer(user_em, self.embedding_dim, 'ranking_to_rating'))
            expanded_user_rating_em = tf.expand_dims(user_rating_em, 1)
            shape = get_dynamic_tensor_shape(pu_item_em)
            all_item_em = tf.reshape(pu_item_em, (shape[0]*shape[1], shape[2]))
            all_item_rating_em = tf.tanh(fc_layer(all_item_em, self.embedding_dim, 'ranking_to_rating', reuse=True))
            pu_item_rating_em0 = tf.reshape(all_item_rating_em, (shape[0], shape[1], self.embedding_dim))
            pu_item_rating_em1 = embedding_lookup_layer(self.input_per_user_item_ids, self.num_items, self.embedding_dim,  'item_rating_embedding', zero_init=True)
            pu_item_rating_em = pu_item_rating_em0 + pu_item_rating_em1
            pu_item_rating_bias = bias_lookup_layer(self.input_per_user_item_ids, self.num_items, 'item_rating_embedding')
            self.rating_prediction = tf.reduce_sum(expanded_user_rating_em * pu_item_rating_em, axis=-1) + mu + expanded_user_bias + pu_item_rating_bias
            self.rating_loss = tf.square(self.input_per_user_ratings - self.rating_prediction)

        # regularization
        with tf.name_scope('regularization'):
            self.reg_loss = (reg_lambda) / 2 * sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        # loss
        with tf.name_scope('loss'):
            self.loss = ((1-alpha)*tf.reduce_mean(self.bpr_loss)) + (alpha*tf.reduce_mean(self.rating_loss)) + self.reg_loss 

    def get_embedding_mats(self):
        with tf.device('/cpu:0'), tf.name_scope('embedding_lookup'):
            user_ids = np.arange(self.num_users)
            item_ids = np.arange(self.num_items)
            U = embedding_lookup_layer(user_ids, self.num_users, self.embedding_dim, 'user_embedding', reuse=True)
            V = embedding_lookup_layer(item_ids, self.num_items, self.embedding_dim, 'item_embedding', reuse=True)
            Vb = bias_lookup_layer(item_ids, self.num_items, 'item_embedding', reuse=True)
            return (U, V, Vb)


# Training
# ==================================================

def train(
    model, sess, starter_learning_rate, learning_rate_decay_every, learning_rate_decay_by, stop_after
):
    # Define Training procedure
    global_step = tf.Variable(0, name="global_step", trainable=False)
    #optimizer = tf.train.AdamOptimizer(1e-3)
    learning_rate = tf.train.exponential_decay(
        starter_learning_rate, global_step, learning_rate_decay_every,
        learning_rate_decay_by, staircase=True)
    # optimizer = tf.train.AdamOptimizer(learning_rate)
    optimizer = tf.train.AdagradOptimizer(learning_rate)

    grads_and_vars = optimizer.compute_gradients(model.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # Keep track of gradient values and sparsity (optional)
    grad_summaries = []
    #for g, v in grads_and_vars:
    for g,v in []:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    #grad_summaries_merged = tf.summary.merge(grad_summaries)

    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print_flush("Writing to {}\n".format(out_dir))

    # Summaries for loss
    loss_summary = tf.summary.scalar("loss", model.loss)
    learning_rate_summary = tf.summary.scalar("learning_rate", learning_rate)

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary, learning_rate_summary])#, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Val summaries
    val_summary_op = tf.summary.merge([loss_summary, learning_rate_summary])
    val_summary_dir = os.path.join(out_dir, "summaries", "val")
    val_summary_writer = tf.summary.FileWriter(val_summary_dir, sess.graph)

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    def train_step(user_ids, per_user_count, per_user_item_ids, per_user_ratings):
        """
        A single training step 
        """
        per_user_neg_ids = ratings.get_batch_neg(user_ids, per_user_item_ids.shape[1])
        feed_dict = {
            model.input_user_ids:          user_ids,
            model.input_per_user_count:    per_user_count,
            model.input_per_user_item_ids: per_user_item_ids,
            model.input_per_user_neg_ids:  per_user_neg_ids,
            model.input_per_user_ratings:  per_user_ratings,
        }
        sess.run(train_op, feed_dict)
        step, loss, rate = sess.run(
            [global_step, model.loss, learning_rate],
            feed_dict)
        if step % FLAGS.summary_every == 0:
            summaries = sess.run(train_summary_op, feed_dict)
            train_summary_writer.add_summary(summaries, step)
        time_str = datetime.now().isoformat()
        if step % FLAGS.summary_every == 0:
            print_flush("{}: step {}, loss {:g}, rate {:g}".format(
                time_str, step, loss, rate)
            )
        return loss

    def val_step(user_ids, per_user_count, per_user_item_ids, per_user_ratings, writer=None):
        """
        Evaluates model on a val set
        """
        per_user_neg_ids = ratings.get_batch_neg(user_ids, per_user_item_ids.shape[1])
        feed_dict = {
            model.input_user_ids:          user_ids,
            model.input_per_user_count:    per_user_count,
            model.input_per_user_item_ids: per_user_item_ids,
            model.input_per_user_neg_ids:  per_user_neg_ids,
            model.input_per_user_ratings:  per_user_ratings,
        }
        step, summaries, loss = sess.run(
            [global_step, val_summary_op, model.loss],
            feed_dict)
        time_str = datetime.now().isoformat()
        print_flush("{}: step {}, loss {:g}".format(
            time_str, step, loss))
        if writer:
            writer.add_summary(summaries, step)
        return loss

    # Generate batches
    batches = ratings.train_batch_iter(FLAGS.batch_size, FLAGS.num_epochs)
    last_val_loss = 0
    # Training loop. For each batch...
    for (user_ids, per_user_count, per_user_item_ids, per_user_ratings) in batches:
        last_train_loss = train_step(user_ids, per_user_count, per_user_item_ids, per_user_ratings)
        current_step = tf.train.global_step(sess, global_step)
        if stop_after and current_step > stop_after:
            print_flush('Stopping after {} training steps'.format(stop_after))
            break
        if current_step % FLAGS.evaluate_every == 0:
            print_flush("\nEvaluation:")
            (val_user_ids, val_per_user_count, val_per_user_item_ids, val_per_user_ratings) = ratings.get_batch(ratings.val[:FLAGS.batch_size])
            last_val_loss = val_step(val_user_ids, val_per_user_count, val_per_user_item_ids, val_per_user_ratings, writer=val_summary_writer)
            U, V, Vb = sess.run(model.get_embedding_mats())
            numtest = 1000
            testids = np.random.permutation(list(set(ratings.val['user_id'])))[:numtest]
            predictions = np.matmul(U[testids], np.transpose(V)) + np.transpose(Vb)
            ndcg,mrr,precision = calc_scores.calc_scores(ratings.val, testids, predictions, 10)
            print_flush(' Scores for val set: NDCG@10={:.4f}, MRR@10={:.4f}, P@10={:.4f}'.format(ndcg, mrr, precision))
            print_flush("")
        if current_step % FLAGS.checkpoint_every == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print_flush("Saved model checkpoint to {}\n".format(path))
            pass
    return (last_train_loss, last_val_loss)


def runall():
    res = defaultdict(list)
    with open('results.txt', 'a') as f:
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=FLAGS.allow_soft_placement,
                log_device_placement=FLAGS.log_device_placement)
            session_conf.gpu_options.allow_growth=True
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                model = PredictionModel(
                    num_users=ratings.num_users,
                    num_items=ratings.num_items,
                    num_ratings=len(ratings.train),
                    embedding_dim=FLAGS.embedding_dim,
                    mu=np.mean(ratings.train['rating']),
                    alpha=FLAGS.alpha,
                    reg_lambda=FLAGS.reg_lambda,
                )
                for i in range(1):
                    last_loss = train(model, sess, 1e0, 40000, 0.5, FLAGS.training_stop_after)
                    f.write('loss: {}\n'.format(last_loss))
                    f.flush()
                    res['loss'].append(last_loss)
                    U, V, Vb = sess.run(model.get_embedding_mats())
                    np.savetxt('results-'+dataset_name+'/ours2.u.txt', U, delimiter=',')
                    np.savetxt('results-'+dataset_name+'/ours2.v.txt', V, delimiter=',')
                    np.savetxt('results-'+dataset_name+'/ours2.vb.txt', Vb, delimiter=',')
                    numtest = 1000
                    testids = np.random.permutation(list(set(ratings.val['user_id'])))[:numtest]
                    predictions = np.matmul(U[testids], np.transpose(V)) + np.transpose(Vb)
                    ndcg,mrr,precision = calc_scores.calc_scores(ratings.val, testids, predictions, 10)
                    f.write(repr((ndcg,mrr,precision)) + '\n')
                    f.write('\n')
                    f.flush()
                    res['ndcg_at_10'].append(ndcg)
                    res['mrr_at_10'].append(mrr)
                    res['precision_at_10'].append(precision)
        print_flush(res)
    return res

res = runall()

print_flush(res)

