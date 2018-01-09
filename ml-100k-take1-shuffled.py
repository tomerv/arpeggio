import sys
sys.path.append('/home/tvromen/research')
from Common.Utils import IdAssigner, print_flush
from Common import RatingsData
import numpy as np
import scipy.io


def load_ml_100k(path, verbose=True):
    if verbose:
        print_flush('Loading MovieLens 100k ratings...')
    with open(path) as f:
        if verbose:
            print_flush('Scanning file...')
        for num_lines,_ in enumerate(f, 1):
            pass
        if verbose:
            print_flush('Will load {} ratings'.format(num_lines))
        f.seek(0)
        all_data = np.zeros(
            num_lines,
            dtype=[('user_id', np.int32), ('item_id', np.int32), ('rating', np.float32), ('timestamp', np.int64)]
        )
        for i,line in enumerate(f):
            user_id, item_id, rating, timestamp = map(int, line.split())
            user_id -= 1 # IDs start at 0, and we don't want users that don't have any ratings
            # rating = (rating - 1) / 4.0
            rating = rating / 5.0
            all_data[i] = (user_id, item_id, rating, timestamp)
    if verbose:
        print_flush('Loaded {} ratings'.format(len(all_data)))
        print_flush('Num users: {}'.format(np.max(all_data['user_id']+1)))
        print_flush('Num items: {}'.format(np.max(all_data['item_id']+1)))
        ratings = all_data['rating']
        print_flush('Min/mean/max rating: {}/{:.3}/{}'.format(
            np.min(ratings), np.mean(ratings), np.max(ratings)
        ))
    return all_data

data = load_ml_100k('/home/tvromen/research/datasets/ml-100k/u.data')
# shuffle order
data['timestamp'] = np.random.randint(1000000, size=len(data))
ratings = RatingsData.RatingsData.from_data(data, p_val=1, p_test=0, take_last=1)

ratings.output_as_text(ratings.train, 'ml-100k-take1-shuffled.train.txt')
ratings.output_as_text(ratings.val, 'ml-100k-take1-shuffled.val.txt')

