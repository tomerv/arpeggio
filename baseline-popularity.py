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


np.random.seed(1234)


###########
# Dataset #
###########

if len(sys.argv) != 2:
    print('Usage: python3 baseline-popularity.py <dataset_name>')
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

k = 10
def range_print(n):
    print('n = {}'.format(n))
    for i in range(n):
        if i % 2000 == 0:
            print('  i = {}'.format(i))
        yield i

# Calculate popularity of each item
# This is too slow:
# popularity = np.array([np.sum(ratings.train_sparse[:,item_id] > 0) for item_id in range_print(ratings.num_items)])
# A faster way:
counts = defaultdict(int)
for rating in ratings.train:
    counts[rating['item_id']] += 1
popularity = np.array([counts[i] for i in range(ratings.num_items)])


numtest = 1000
testids = np.random.permutation(list(set(ratings.val['user_id'])))[:numtest]
predictions = np.repeat(popularity[None], numtest, axis=0)
ndcg, mrr, precision = calc_scores.calc_scores(ratings.train, testids, predictions, k)
print_flush('Results on TRAINING set: NDCG@{}={}, MRR@{}={}, P@{}={}'.format(k, ndcg, k, mrr, k, precision))
ndcg, mrr, precision = calc_scores.calc_scores(ratings.val, testids, predictions, k)
print_flush('Results on VALIDATION set: NDCG@{}={}, MRR@{}={}, P@{}={}'.format(k, ndcg, k, mrr, k, precision))

U = np.ones((ratings.num_users, 1))
V = popularity[:,None]
np.savetxt('results-'+dataset_name+'/popularity.u.txt', U, delimiter=',')
np.savetxt('results-'+dataset_name+'/popularity.v.txt', V, delimiter=',')

