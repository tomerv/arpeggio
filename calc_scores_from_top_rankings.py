import sys
sys.path.append('/home/tvromen/research')
from Common.Utils import print_flush
from Common import RatingsData
import numpy as np
import scipy.io
import ast



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


def main():

    if len(sys.argv) != 3:
        print_flush('Usage: python3 calc_scores_from_top_rankings.py <dataset_name> <model_name>')
        exit(1)

    dataset = sys.argv[1]
    model = sys.argv[2]
    # dataset = 'yelp-take1'
    # model = 'popularity'

    ratings = RatingsData.RatingsData.from_files(dataset+'.train.txt', dataset+'.val.txt')

    save_path = 'results-'+dataset+'/'+model+'.top_rankings.txt'
    print('Loading top rankings from {}'.format(save_path))
    with open(save_path) as f:
        print('Reading file...')
        r = f.read().replace('array', 'np.array')
    print('Processing file...')
    per_user_top_rankings = eval(r)
    k = len(per_user_top_rankings[list(per_user_top_rankings.keys())[0]])

    print_flush('Calculating scores...')
    ndcg, mrr, precision = calc_scores_(ratings.val, ratings.num_items, k, per_user_top_rankings, verbose=True)
    print_flush('Results on VALIDATION set: NDCG@{}={}, MRR@{}={}, P@{}={}'.format(k, ndcg, k, mrr, k, precision))


if __name__ == '__main__':
    main()

