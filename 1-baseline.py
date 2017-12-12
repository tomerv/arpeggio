import numpy as np
import json
import os
import sys
from datetime import datetime
import time
from collections import defaultdict
sys.path.append('/home/tvromen/research')
from Common.Utils import IdAssigner, print_flush
from Common.RatingsData import RatingsData
import tensorflow as tf
import matplotlib.pyplot as plt

class Flags(object):
    def __init__(self):
        # Data loading params
        self.ratings_file = '/home/tvromen/research/yelp_dataset/review.json' # Data source for the ratings
        self.max_lines = None

        # Model Hyperparameters
        self.embedding_dim = 10
        self.reg_lambda = 0.1

        # Training parameters
        self.training_stop_after = None
        self.batch_size = 1024
        self.num_epochs = 200
        self.summary_every = 100
        self.evaluate_every = 1000
        self.checkpoint_every = 2000
        self.num_checkpoints = 3

        # Misc Parameters
        self.allow_soft_placement = True
        self.log_device_placement = False

FLAGS = Flags()


def load_yelp(path, max_lines=None, verbose=True):
    if verbose:
        print_flush('Loading Yelp ratings...')
    user2id = IdAssigner()
    item2id = IdAssigner()
    with open(path) as f:
        if verbose:
            print_flush('Scanning file...')
        for num_lines,_ in enumerate(f, 1):
            if num_lines == max_lines:
                break
        if verbose:
            print_flush('Will load {} ratings'.format(num_lines))
        all_data = np.zeros(
            num_lines,
            dtype=[('user_id', np.int32), ('item_id', np.int32), ('rating', np.float32), ('timestamp', np.int64)]
        )
        f.seek(0)
        for i,line in enumerate(f):
            if i == num_lines:
                break
            if verbose and ((i+1) % 100000) == 0:
                print_flush('Loaded {} ratings...'.format(i+1))
            data = json.loads(line)
            user_id = user2id.get_id(data['user_id'])
            item_id = item2id.get_id(data['business_id'])
            rating = (data['stars'] - 1) / 4.0
            # too slow:
            # timestamp = datetime.strptime(data['date'], '%Y-%m-%d').toordinal()
            year, month, day = map(int, data['date'].split('-'))
            timestamp = datetime(year=year, month=month, day=day).toordinal()
            all_data[i] = (user_id, item_id, rating, timestamp)
    if verbose:
        print_flush('Loaded {} ratings'.format(len(all_data)))
        print_flush('Num users: {}'.format(user2id.get_next_id()))
        print_flush('Num items: {}'.format(item2id.get_next_id()))
        ratings = all_data['rating']
        print_flush('Min/mean/max rating: {}/{:.3}/{}'.format(
            np.min(ratings), np.mean(ratings), np.max(ratings)
        ))
    return all_data

np.random.seed(1234)

FLAGS.max_lines = 500000  #TODO
yelp_data = load_yelp(FLAGS.ratings_file, FLAGS.max_lines)

def remove_extreme_users(data):
    counts = np.bincount(data['user_id'])
    q = [50.0, 90.0, 99.0, 99.9, 99.99, 99.999]
    percentiles = np.percentile(counts, q, interpolation='nearest')
    print_flush('Percentiles for number of ratings per user:')
    for a,b in zip(q, percentiles):
        print_flush('  {}: {}'.format(a, b))
    # plt.hist(counts, bins=np.logspace(0., np.log10(np.max(counts)) , 20), normed=1, cumulative=True)
    # plt.gca().set_xscale("log")
    # plt.show()
    max_items_per_user = percentiles[3] + 1  # 99.9%
    print_flush('Removing all users with just 1 rating, or more than {} ratings'.format(max_items_per_user))
    too_much   = set(np.flatnonzero(counts > max_items_per_user))
    too_little = set(np.flatnonzero(counts < 2))
    bad_user = too_much | too_little
    print_flush('Removing {} users'.format(len(bad_user)))
    data = np.array([x for x in data if x['user_id'] not in bad_user])
    print_flush('Left with {} ratings'.format(len(data)))
    return data

yelp_data = remove_extreme_users(yelp_data)

ratings = RatingsData(yelp_data)

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

# print_flush(ratings.train[:10])
# for i,b in enumerate(ratings.train_batch_iter(10, 1)):
#     if i == 0:
#         print_flush(b)
#     break
# exit()


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

def embedding_lookup_layer(x, vocab_size, embedding_dim, variable_scope, reuse=False):
    """
    Lookup embedding
    x is tensor of shape (d_1, d_2, ..., d_n) and type int32
    result is tensor of shape (d_1, d_2, ..., d_n, embedding_dim) of n+1 dimensions and type DT_FLOAT
    """
    with tf.variable_scope(variable_scope, reuse=reuse):
        W = tf.get_variable(
            'W',
            shape=[vocab_size, embedding_dim],
            initializer=tf.contrib.layers.xavier_initializer(),
            regularizer=tf.contrib.layers.l2_regularizer(1.)
        )
    x_embedded = tf.nn.embedding_lookup(W, x)
    return x_embedded

class PredictionModel(object):
    """
    A neural network for predicting per-user item ratings.
    The input to the network is the user_id and item_id.
    """
    def __init__(self, num_users, num_items, num_ratings, embedding_dim, alpha, reg_lambda):

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
        self.input_user_id = tf.placeholder(tf.int32, [None], name="input_user_id")
        self.input_item_id = tf.placeholder(tf.int32, [None], name="input_item_id")
        self.input_rating = tf.placeholder(tf.float32, [None], name="input_rating")
        self.input_topone = tf.placeholder(tf.float32, [None], name="input_topone")
        self.input_per_user_items = tf.placeholder(tf.int32, [None, None], name="input_per_user_items")
        self.input_per_user_item_count = tf.placeholder(tf.int32, [None], name="input_per_user_item_count")
    
        batch_size = tf.shape(self.input_user_id)[0]
        asrt1 = tf.assert_equal(batch_size, tf.shape(self.input_item_id)[0])
        asrt2 = tf.assert_equal(batch_size, tf.shape(self.input_rating)[0])
        asrt3 = tf.assert_equal(batch_size, tf.shape(self.input_topone)[0])
        asrt4 = tf.assert_equal(batch_size, tf.shape(self.input_per_user_items)[0])
        asrt5 = tf.assert_equal(batch_size, tf.shape(self.input_per_user_item_count)[0])

        # embedding lookup layer
        with tf.device('/cpu:0'), tf.name_scope('embedding_lookup'), tf.control_dependencies([asrt1, asrt2, asrt3, asrt4, asrt5]):
            user_embedding = embedding_lookup_layer(self.input_user_id, self.num_users, self.embedding_dim, 'user_embedding')
            item_embedding = embedding_lookup_layer(self.input_item_id, self.num_items, self.embedding_dim, 'item_embedding')
            item_embedding2 = embedding_lookup_layer(self.input_per_user_items, self.num_items, self.embedding_dim, 'item_embedding', reuse=True)

        # PMF part
        with tf.name_scope('pmf'):
            self.pmf_prediction = tf.sigmoid(tf.reduce_sum(user_embedding * item_embedding, axis=1))
            self.pmf_loss_batch = 0.5 * tf.reduce_sum(tf.squared_difference(self.input_rating, self.pmf_prediction))
            self.pmf_loss = self.pmf_loss_batch / tf.cast(batch_size, tf.float32) * self.num_ratings

        # ListRank part
        with tf.name_scope('listrank'):
            mask = tf.sequence_mask(self.input_per_user_item_count, dtype=tf.float32)
            # calculate topone for the prediction
            t = tf.expand_dims(user_embedding, 1) * item_embedding2
            prediction_topone_denom = tf.reduce_sum(mask * tf.exp(tf.sigmoid(tf.reduce_sum(t, axis=-1))))
            prediction_topone = tf.exp(tf.sigmoid(tf.reduce_sum(user_embedding * item_embedding, axis=-1))) / prediction_topone_denom
            self.listrank_loss_batch = tf.reduce_sum(-self.input_topone * tf.log(prediction_topone))
            self.listrank_loss = self.listrank_loss_batch / tf.cast(batch_size, tf.float32) * self.num_ratings

        # regularization
        with tf.name_scope('regularization'):
            # batch_lambda = reg_lambda * tf.cast(batch_size, tf.float32) / tf.cast(self.num_users, tf.float32)
            # self.reg_loss = batch_lambda / 2 * (tf.reduce_sum(tf.square(user_embedding)) + tf.reduce_sum(tf.square(item_embedding)))
            self.reg_loss = reg_lambda / 2 * (tf.reduce_sum(tf.square(user_embedding)) + tf.reduce_sum(tf.square(item_embedding)))

        # loss
        with tf.name_scope('loss'):
            self.loss = alpha * self.pmf_loss + (1-alpha) * self.listrank_loss + self.reg_loss

    def get_ranking_predictions(self, user_ids):
        with tf.device('/cpu:0'), tf.name_scope('embedding_lookup'):
            item_ids = np.arange(self.num_items)
            user_embeddings = embedding_lookup_layer(user_ids, self.num_users, self.embedding_dim, 'user_embedding', reuse=True)
            item_embeddings = embedding_lookup_layer(item_ids, self.num_items, self.embedding_dim, 'item_embedding', reuse=True)
            predicitions = tf.matmul(user_embeddings, item_embeddings, transpose_b=True)
            return predicitions


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

    def train_step(batch_user_id, batch_item_id, batch_rating, batch_topone, batch_pui, batch_puic):
        """
        A single training step 
        """
        feed_dict = {
            model.input_user_id: batch_user_id,
            model.input_item_id: batch_item_id,
            model.input_rating: batch_rating,
            model.input_topone: batch_topone,
            model.input_per_user_items: batch_pui,
            model.input_per_user_item_count: batch_puic,
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

    def val_step(batch_user_id, batch_item_id, batch_rating, batch_topone, batch_pui, batch_puic, writer=None):
        """
        Evaluates model on a val set
        """
        feed_dict = {
            model.input_user_id: batch_user_id,
            model.input_item_id: batch_item_id,
            model.input_rating: batch_rating,
            model.input_topone: batch_topone,
            model.input_per_user_items: batch_pui,
            model.input_per_user_item_count: batch_puic,
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
    for (batch_user_id, batch_item_id, batch_rating, batch_topone, batch_per_user_items, batch_per_user_item_count) in batches:
        last_train_loss = train_step(batch_user_id, batch_item_id, batch_rating, batch_topone, batch_per_user_items, batch_per_user_item_count)
        current_step = tf.train.global_step(sess, global_step)
        if stop_after and current_step > stop_after:
            print_flush('Stopping after {} training steps'.format(stop_after))
            break
        if current_step % FLAGS.evaluate_every == 0:
            print_flush("\nEvaluation:")
            (val_user_id, val_item_id, val_rating, val_topone, val_per_user_items, val_per_user_item_count) = ratings.get_batch(ratings.val[:1024])
            last_val_loss = val_step(val_user_id, val_item_id, val_rating, val_topone, val_per_user_items, val_per_user_item_count, writer=val_summary_writer)
            print_flush("")
        if current_step % FLAGS.checkpoint_every == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print_flush("Saved model checkpoint to {}\n".format(path))
            pass
    return (last_train_loss, last_val_loss)


def calc_precision(sess, model, k):
    ranks = []
    precision_at_k = 0
    mrr_at_k = 0.
    user_ids = list(set(ratings.val['user_id']))
    scores = sess.run(model.get_ranking_predictions(user_ids))
    n = len(ratings.val)
    print_flush('Calculating precision on {} items'.format(n))
    for i in range(n):
        user_id, item_id, rating = ratings.val[i]
        if i % 1000 == 0:
            print_flush('{}...'.format(i))
        s = scores[i][item_id] # the score for the correct item
        #print_flush(s)
        train_items = ratings.train[ratings.train['user_id'] == user_id]['item_id']
        # not_watched = (scores[i] == scores[i]) # all True
        not_watched = np.ones_like(scores[i], dtype=np.bool)
        not_watched[train_items] = False
        higher_scores = (scores[i] > s)    
        rank = np.sum(higher_scores & not_watched) + 1
        ranks.append(rank)
        #print_flush('for user_id {} the rank is {}'.format(user_id, rank))
        if rank <= k:
            precision_at_k += 1
            mrr_at_k += 1. / rank
    precision_at_k /= n
    mrr_at_k /= n
    print_flush('Precision@{} is {}'.format(k, precision_at_k))
    print_flush('MRR@{} is {}'.format(k, mrr_at_k))
    # plt.hist(ranks, bins=np.logspace(0., np.log10(ratings.id_assigner.get_next_id()) , 20), normed=1)
    # plt.gca().set_xscale("log")
    # plt.show()
    return precision_at_k, mrr_at_k


def runall():
    res = defaultdict(lambda : defaultdict(list))
    with open('results.txt', 'a') as f:
        # for alpha in [0.0, 0.1, 0.2, 0.3, 0.4]:
        for alpha in [0.0]:
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
                        alpha=alpha,
                        reg_lambda=FLAGS.reg_lambda,
                    )
                    for i in range(1):
                        f.write('alpha: {}\n'.format(alpha))
                        last_loss = train(model, sess, 1e0, 40000, 0.5, FLAGS.training_stop_after)
                        f.write('loss: {}\n'.format(last_loss))
                        f.flush()
                        res[alpha]['loss:'].append(last_loss)
                        precision_at_10, mrr_at_10 = calc_precision(sess, model, 10)
                        f.write(repr((precision_at_10, mrr_at_10)) + '\n')
                        f.write('\n')
                        f.flush()
                        res[alpha]['precision_at_10'].append(precision_at_10)
                        res[alpha]['mrr_at_10'].append(mrr_at_10)
            print_flush(res)
    return res

res = runall()

print_flush(res)

