import sys
sys.path.append('/home/tvromen/research')
from Common.Utils import IdAssigner, print_flush
from Common import RatingsData
import numpy as np
import scipy.io


def calc_scores(true_ratings, predictions, k, save_path=None, verbose=True):
    user_ids = set(true_ratings['user_id'])
    if verbose:
        print_flush('Calculating scores on {} users and {} ratings'.format(len(user_ids), len(true_ratings)))
    per_user_top_rankings = calc_top(user_ids, predictions, k, verbose)
    if save_path is not None:
        try:
            with open(save_path, 'w') as f:
                f.write(repr(per_user_top_rankings))
        except Exception as e:
            print(e)
    num_items = predictions.shape[1]
    return calc_scores_(true_ratings, num_items, k, per_user_top_rankings, verbose)


def calc_scores_(true_ratings, num_items, k, per_user_top_rankings, verbose):
    def calc_dcg(top_k_ratings):
        assert len(top_k_ratings) == k
        return np.sum(((2 ** top_k_ratings) - 1) / np.log2(2 + np.arange(k)))
    ndcg = 0
    mrr = 0
    for i,(user_id,top_predicted_item_ids) in enumerate(per_user_top_rankings.items()):
        if verbose and ((user_id+1) % 1000) == 0:
            print_flush('  {}...'.format(user_id+1))
        user_ratings = true_ratings[true_ratings['user_id'] == user_id]
        user_all_ratings = np.zeros([num_items], dtype=np.float32)
        user_all_ratings[user_ratings['item_id']] = user_ratings['rating']
        top_predicted_ratings = user_all_ratings[top_predicted_item_ids]
        dcg = calc_dcg(top_predicted_ratings)
        # calculate IDCG (ideal DCG)
        top_k_ratings = np.sort(user_ratings['rating'])[::-1][:k]
        pad = [0]*(k-len(top_k_ratings))
        top_k_ratings = np.concatenate((top_k_ratings, pad))
        idcg = calc_dcg(top_k_ratings)
        ndcg += dcg / idcg
        # calculate MRR
        match_rank = np.nonzero(top_predicted_ratings)[0]
        mrr += 1./(match_rank[0]+1) if len(match_rank) > 0 else 0
    ndcg = ndcg / len(per_user_top_rankings)
    mrr = mrr / len(per_user_top_rankings)
    return (ndcg, mrr)

def calc_top(user_ids, predictions, k, verbose):
    per_user_top_rankings = dict()
    for user_id in range(len(predictions)):
        if user_id not in user_ids:
            continue
        if verbose and ((user_id+1) % 1000) == 0:
            print_flush('  {}...'.format(user_id+1))
        # calculate the score for the prediction
        user_predictions = predictions[user_id]
        top_predicted_item_ids = np.argsort(user_predictions)[::-1][:k]
        if user_id < 5:
            print(top_predicted_item_ids)
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
        [ [0,   0,  0,  0,  0,  0,  0, 0, 0, 0, 0],
          [0, 100, 99, 98, 97, 96, 95, 0, 0, 0, 0] ]
    )
    ndcg,mrr = calc_scores(true_ratings, predictions, 6, verbose=False)
    assert np.abs(ndcg - 0.785) < 1e-4, ndcg
    assert np.abs(mrr - 1.0) < 1e-6, mrr

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
        [ [0,  0,  0,  0, 0],
          [0, 99, 98, 97, 0],
          [0, 99, 98, 97, 0],
          [0, 99, 98, 97, 0] ]
    )
    ndcg,mrr = calc_scores(true_ratings, predictions, 3, verbose=False)
    assert np.abs(mrr - 11./18.) < 1e-6, mrr
    assert np.abs(ndcg - 0.7103) < 1e-4, ndcg

test_calc_scores()


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print('Usage: python3 calc_scores.py <dataset_name> <model_name>')
        exit(1)

    name = sys.argv[1]
    model = sys.argv[2]

    ratings = RatingsData.RatingsData.from_files(name+'.train.txt', name+'.val.txt')

    def loadmat(path):
        with open(path) as f:
            lines = f.readlines()
            vals = [list(map(float, line.split(','))) for line in lines]
            lens = list(map(len, vals))
            assert np.min(lens) == np.max(lens)
            return np.array(vals)

    U = loadmat('results/'+name+'.'+model+'.u.txt')
    assert U.shape[0] == ratings.num_users, (U.shape, ratings.num_users)
    V = loadmat('results/'+name+'.'+model+'.v.txt')
    assert V.shape[0] == ratings.num_items, (V.shape, ratings.num_items)
    # verify that the embedding dim is the same
    assert U.shape[1] == V.shape[1], (U.shape, V.shape)
    Vb = None
    try:
        Vb = loadmat('results/'+name+'.'+model+'.vb.txt')
        assert Vb.shape == (ratings.num_items, 1), Vb.shape
    except FileNotFoundError as e:
        print('Skipping item bias: {}'.format(e))

    if Vb is None:
        predictions = np.matmul(U, np.transpose(V))
    else:
        predictions = np.matmul(U, np.transpose(V)) + np.transpose(Vb)

    k = 10

    ndcg, mrr = calc_scores(ratings.train, predictions, k)
    print('Results on TRAINING set: NDCG@{}={}, MRR@{}={}'.format(k, ndcg, k, mrr))

    save_path = 'results/'+name+'.'+model+'.top_rankings.txt'
    ndcg, mrr = calc_scores(ratings.val, predictions, k, save_path)
    print('Results on VALIDATION set: NDCG@{}={}, MRR@{}={}'.format(k, ndcg, k, mrr))

