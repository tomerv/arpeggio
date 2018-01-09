import sys
sys.path.append('/home/tvromen/research')
from Common.Utils import print_flush
from Common import RatingsData
import numpy as np
import scipy.io


def calc_scores(true_ratings, user_ids, predictions, k, save_path=None, verbose=True):
    if verbose:
        print_flush('Calculating scores on {} users'.format(len(user_ids)))
    per_user_top_rankings = calc_top(user_ids, predictions, k, verbose)
    if save_path is not None:
        try:
            with open(save_path, 'w') as f:
                f.write(repr(per_user_top_rankings))
        except Exception as e:
            print_flush(e)
    num_items = predictions.shape[1]
    return calc_scores_(true_ratings, num_items, k, per_user_top_rankings, verbose)


def calc_scores_(true_ratings, num_items, k, per_user_top_rankings, verbose):
    def calc_dcg(top_k_ratings):
        assert len(top_k_ratings) == k
        return np.sum(((2 ** top_k_ratings) - 1) / np.log2(2 + np.arange(k)))
    ndcg = 0
    mrr = 0
    precision = 0
    for i,(user_id,top_predicted_item_ids) in enumerate(per_user_top_rankings.items()):
        if verbose and ((i+1) % 1000) == 0:
            print_flush('  {}...'.format(i+1))
        user_ratings = true_ratings[true_ratings['user_id'] == user_id]
        assert len(user_ratings) != 0
        user_all_ratings = np.zeros([num_items], dtype=np.float32)
        user_all_ratings[user_ratings['item_id']] = user_ratings['rating']
        top_predicted_ratings = user_all_ratings[top_predicted_item_ids]
        dcg = calc_dcg(top_predicted_ratings)
        # calculate IDCG (ideal DCG)
        top_k_ratings = np.sort(user_ratings['rating'])[::-1][:k]
        pad = [0]*(k-len(top_k_ratings))
        top_k_ratings = np.concatenate((top_k_ratings, pad))
        idcg = calc_dcg(top_k_ratings)
        if idcg == 0:
            print(user_id)
            print(user_ratings)
        ndcg += dcg / idcg
        # calculate MRR & precision
        match_rank = np.nonzero(top_predicted_ratings)[0]
        mrr += 1./(match_rank[0]+1) if len(match_rank) > 0 else 0
        precision += 1. if len(match_rank) > 0 else 0
    num_users = len(per_user_top_rankings)
    ndcg      = ndcg      / num_users
    mrr       = mrr       / num_users
    precision = precision / num_users
    return (ndcg, mrr, precision)

def calc_top(user_ids, predictions, k, verbose):
    per_user_top_rankings = dict()
    for i,user_id in enumerate(user_ids):
        if verbose and ((i+1) % 1000) == 0:
            print_flush('  {}...'.format(i+1))
        # calculate the score for the prediction
        user_predictions = predictions[i]
        # too slow:
        # top_predicted_item_ids = np.array(np.argsort(user_predictions)[::-1][:k])
        # a more efficient way:
        top_k = np.argpartition(-user_predictions, k)[:k]
        top_predicted_item_ids = top_k[np.argsort(user_predictions[top_k])[::-1]]
        # if i < 5:
        #     print_flush(top_predicted_item_ids)
        per_user_top_rankings[user_id] = top_predicted_item_ids
    return per_user_top_rankings


#### Test for calc_scores
# Using the example from the Wikipedia articles on NDCG and MRR
# https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Example
# https://en.wikipedia.org/wiki/Mean_reciprocal_rank#Example
####

def test_calc_scores():

    true_ratings = np.array(
        [
            (1, 1, np.log2(3+1)),
            (1, 2, np.log2(2+1)),
            (1, 3, np.log2(3+1)),
            (1, 4, np.log2(0+1)),
            (1, 5, np.log2(1+1)),
            (1, 6, np.log2(2+1)),
            (1, 7, np.log2(3+1)),
            (1, 8, np.log2(2+1)),
        ], dtype = ([('user_id', '<i4'), ('item_id', '<i4'), ('rating', '<f4')])
    )
    true_ratings = np.random.permutation(true_ratings)
    predictions = np.array(
        [ [0, 100, 99, 98, 97, 96, 95, 0, 0, 0, 0] ]
    )
    ndcg,mrr,precision = calc_scores(true_ratings, [1], predictions, 6, verbose=False)
    assert np.abs(ndcg - 0.785) < 1e-4, ndcg
    assert np.abs(mrr - 1.0) < 1e-6, mrr
    assert precision == 1.0, precision

    true_ratings = np.array(
        [
            (1, 1, 0),
            (1, 2, 0),
            (1, 3, 1),
            (2, 1, 0),
            (2, 2, 1),
            (2, 3, 0),
            (3, 1, 1),
            (3, 2, 0),
            (3, 3, 0),
        ], dtype = ([('user_id', '<i4'), ('item_id', '<i4'), ('rating', '<f4')])
    )
    true_ratings = np.random.permutation(true_ratings)
    predictions = np.array(
        [ [0, 99, 98, 97, 0],
          [0, 99, 98, 97, 0],
          [0, 99, 98, 97, 0] ]
    )
    ndcg,mrr,precision = calc_scores(true_ratings, [1,2,3], predictions, 3, verbose=False)
    assert np.abs(mrr - 11./18.) < 1e-6, mrr
    assert np.abs(ndcg - 0.7103) < 1e-4, ndcg
    assert precision == 1.0, precision

test_calc_scores()


def main():

    if len(sys.argv) != 3:
        print_flush('Usage: python3 calc_scores.py <dataset_name> <model_name>')
        exit(1)

    dataset = sys.argv[1]
    model = sys.argv[2]
    # dataset = 'yelp-take1'
    # model = 'popularity'

    ratings = RatingsData.RatingsData.from_files(dataset+'.train.txt', dataset+'.val.txt')

    def loadmat(path):
        with open(path) as f:
            lines = f.readlines()
            vals = [list(map(float, line.split(','))) for line in lines]
            lens = list(map(len, vals))
            assert np.min(lens) == np.max(lens)
            return np.array(vals)

    U = loadmat('results-'+dataset+'/'+model+'.u.txt')
    assert U.shape[0] == ratings.num_users, (U.shape, ratings.num_users)
    V = loadmat('results-'+dataset+'/'+model+'.v.txt')
    assert V.shape[0] == ratings.num_items, (V.shape, ratings.num_items)
    # verify that the embedding dim is the same
    assert U.shape[1] == V.shape[1], (U.shape, V.shape)
    Vb = None
    try:
        Vb = loadmat('results-'+dataset+'/'+model+'.vb.txt')
        assert Vb.shape == (ratings.num_items, 1), Vb.shape
    except FileNotFoundError as e:
        print_flush('Skipping item bias: {}'.format(e))

    save_path = 'results-'+dataset+'/'+model+'.top_rankings.txt'
    calc_wrapper(ratings, U, V, Vb, save_path)


def calc_wrapper(ratings, U, V, Vb, save_path=None):

    k = 10

    batch_size = 5000
    num_batches = 4
    per_user_top_rankings = dict()
    user_ids = np.random.permutation(ratings.val['user_id'])
    for i in range(num_batches):
        print_flush('Batch {}'.format(i+1))
        batch_ids = user_ids[i*batch_size:(i+1)*batch_size]
        if len(batch_ids) == 0:
            break
        if Vb is None:
            predictions = np.matmul(U[batch_ids], np.transpose(V))
        else:
            predictions = np.matmul(U[batch_ids], np.transpose(V)) + np.transpose(Vb)
        print_flush('Removing items from training set...')
        for rating in ratings.train:
            pos = np.where(batch_ids==rating['user_id'])
            if len(pos) == 0:
                continue
            assert len(pos) == 1
            predictions[pos,rating['item_id']] = -10000000
        print_flush('Calculating top items...'.format(i))
        batch_top_rankings = calc_top(batch_ids, predictions, k, verbose=True)
        per_user_top_rankings.update(batch_top_rankings)
        del predictions
    if save_path is not None:
        try:
            with open(save_path, 'w') as f:
                f.write(repr(per_user_top_rankings))
        except Exception as e:
            print_flush(e)
    print_flush('Calculating scores...')
    ndcg, mrr, precision = calc_scores_(ratings.train, ratings.num_items, k, per_user_top_rankings, verbose=True)
    print_flush('Results on TRAINING set: NDCG@{}={}, MRR@{}={}, P@{}={}'.format(k, ndcg, k, mrr, k, precision))
    ndcg, mrr, precision = calc_scores_(ratings.val, ratings.num_items, k, per_user_top_rankings, verbose=True)
    print_flush('Results on VALIDATION set: NDCG@{}={}, MRR@{}={}, P@{}={}'.format(k, ndcg, k, mrr, k, precision))


if __name__ == '__main__':
    main()

