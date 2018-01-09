import numpy as np
import scipy.sparse
from collections import defaultdict
if __name__ == '__main__':
    from Utils import print_flush
else:
    import sys
    import os
    PACKAGE_PARENT = '..'
    SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
    sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
    from Common.Utils import print_flush


class RatingsData:
    """
    Class to handle a dataset of ratings
    """

    def __init__(self, num_users, num_items, train, val, test):
        self.num_users = num_users
        self.num_items = num_items
        correct_type = np.dtype([('user_id', '<i4'), ('item_id', '<i4'), ('rating', '<f4')])
        for x in [train, val, test]:
            assert x.dtype == correct_type, x.dtype
            if len(x) > 0:
                assert np.min(x['user_id']) >= 0
                assert np.max(x['user_id']) < num_users
                assert np.min(x['item_id']) >= 0
                assert np.max(x['item_id']) < num_items
        self.train = train
        self.val   = val
        self.test  = test
        self.train_sparse = scipy.sparse.coo_matrix(
            (train['rating'], (train['user_id'], train['item_id'])),
            shape=(num_users, num_items)
        ).tolil()
        assert not np.any(np.array(np.sum(self.train_sparse, axis=1))[0] == 0), 'A user does not have any ratings'

    @classmethod
    def from_data(cls, data, p_val=0.1, p_test=0.1, give_first=None, take_last=None):
        correct_type = np.dtype([('user_id', '<i4'), ('item_id', '<i4'), ('rating', '<f4'), ('timestamp', '<i8')])
        assert data.dtype == correct_type, data.dtype
        assert p_val >= 0
        assert p_test >= 0
        assert p_val+p_test <= 1
        assert not (give_first and take_last)
        if not give_first and not take_last:
            take_last = 1
        num_users = np.max(data['user_id']) + 1
        num_items = np.max(data['item_id']) + 1
        # we allow users/items without any appearances, so the following assert is incorrect
        # assert len(set(data['user_id'])) == num_users, (len(set(data['user_id'])), num_users)
        # assert len(set(data['item_id'])) == num_items, (len(set(data['item_id'])), num_items)
        sorted_data = np.sort(data, order=['user_id', 'timestamp'])
        sorted_data = sorted_data[['user_id', 'item_id', 'rating']]
        # print some stats
        counts = np.bincount(sorted_data['user_id'])
        nonzero_counts = counts[counts > 0]
        print_flush('Items per user min/mean/max: {}/{:.2f}/{}'.format(
            np.min(nonzero_counts), np.mean(nonzero_counts), np.max(nonzero_counts)
        ))
        # split training/validation:
        # take either a constant number for the training set, or a constant number for the val/test sets,
        # depending on the given options
        # calculate mask for last items of each user
        user_ids = sorted_data['user_id']
        if take_last:
            shifted_user_ids = np.append(user_ids[take_last:], [-1]*take_last)
            is_last = (user_ids != shifted_user_ids)
            assert np.sum(is_last) == len(set(data['user_id'])) * take_last, 'take_last ({}) is bigger than the smallest user'.format(take_last)
        else:
            shifted_user_ids = np.append([-1]*give_first, user_ids[:-give_first])
            is_last = (user_ids == shifted_user_ids)
            assert np.sum(~is_last) == len(set(data['user_id'])) * give_first, 'give_first ({}) is bigger than the smallest user'.format(give_first)
        # p_val go to val, p_test to test
        r = np.random.rand(*is_last.shape)
        is_val   =  is_last & ((0.0 <= r) & (r < p_val))
        is_test  =  is_last & ((p_val <= r) & (r < p_val+p_test))
        is_train = ~is_last | ((p_val+p_test <= r) & (r < 1.0))
        train = sorted_data[is_train]
        val   = sorted_data[is_val]
        test  = sorted_data[is_test]
        assert len(train) + len(val) + len(test) == len(sorted_data)
        return cls(num_users, num_items, train, val, test)

    @classmethod
    def from_files(cls, trainpath, valpath, testpath=None):
        dtype = np.dtype([('user_id', '<i4'), ('item_id', '<i4'), ('rating', '<f4')])
        num_users = None
        num_items = None
        def readmat(path):
            with open(path) as f:
                nonlocal num_users, num_items
                m, n = map(int, f.readline().split())
                if num_users is None:
                    num_users = m
                if num_items is None:
                    num_items = n
                assert num_users == m
                assert num_items == n
                def processline(line):
                    i,j,v = line.split()
                    return (int(i), int(j), float(v))
                data = np.array([processline(line) for line in f], dtype=dtype)
                return data
        def emptymat(m, n):
            return np.array([], dtype=dtype)
        train = readmat(trainpath)
        val   = readmat(valpath)
        test  = readmat(testpath) if testpath is not None else emptymat(num_users, num_items)
        return cls(num_users, num_items, train, val, test)


    def train_batch_iter(self, min_batch_size, num_epochs):
        """
        Generates batches for the training data.
        min_batch_size (and not batch_size), since we want all of each user's datapoints in a single batch
        """
        assert min_batch_size >= 2
        data = self.train
        n = len(data)
        user_ids = data['user_id']
        print_flush('About {} steps per epoch'.format(len(data) // min_batch_size))
        for epoch in range(num_epochs):
            print_flush('Starting epoch {} out of {}'.format(epoch+1, num_epochs))
            next_start = 0
            while next_start < n:
                start = next_start
                end = min(start + min_batch_size, n)
                while end < n:
                    if user_ids[end-1] != user_ids[end]:
                        # seeing a new user
                        break
                    end += 1
                batch = data[start:end]
                next_start = end
                yield self.get_batch(batch)

    def get_batch(self, batch):
        # group the data by user
        ratings_per_user = defaultdict(list)
        for x in batch:
            ratings_per_user[x['user_id']].append((x['item_id'], x['rating']))
        max_ratings_per_user = np.max(list(map(len, ratings_per_user.values())))
        num_users = len(ratings_per_user.keys())
        user_ids          = np.zeros([num_users], dtype=np.int32)
        per_user_count    = np.zeros([num_users], dtype=np.int32)
        per_user_item_ids = np.zeros([num_users, max_ratings_per_user], dtype=np.int32)
        per_user_ratings  = np.zeros([num_users, max_ratings_per_user], dtype=np.float32)
        # per_user_topone   = np.zeros([num_users, max_ratings_per_user], dtype=np.float32)
        for i,(user_id,ratings) in enumerate(ratings_per_user.items()):
            user_ids[i] = user_id
            per_user_count[i] = len(ratings)
            for j,(item_id,rating) in enumerate(ratings):
                per_user_item_ids[i,j] = item_id
                per_user_ratings[i,j]  = rating
            # per_user_topone[i,:] = np.exp(per_user_ratings[i.:]) / np.sum(np.exp(per_user_ratings[i,:]))
        return (user_ids, per_user_count, per_user_item_ids, per_user_ratings)

    def get_batch_neg(self, user_ids, count):
        """
        Generates a negative samples (unrated items) for each user.
        Note: Uses the training set to decide, so it might return items that are actually rated in the val/test sets.
        Inputs:
            user_ids: array of length n
            count: integer
        Output: array of shape (len(user_ids), count)
        """
        res = np.zeros((len(user_ids), count), dtype=np.int32)
        for i,user_id in enumerate(user_ids):
            user_watched = np.array(self.train_sparse[user_id].todense())[0] != 0
            p = np.array((1 - user_watched), dtype=np.float32)
            p = p / np.sum(p)
            res[i] = np.random.choice(self.num_items, size=[count], p=p)
        return res

    def output_as_text(self, data, filename):
        with open(filename, 'w') as f:
            f.write('{} {}\n'.format(self.num_users, self.num_items))
            for rating in data:
                f.write('{} {} {}\n'.format(rating['user_id'], rating['item_id'], rating['rating']))


def remove_extreme_users(data, min_ratings, max_ratings):
    print_flush('Removing all users with less than {} or more than {} ratings'.format(min_ratings, max_ratings))
    counts = np.bincount(data['user_id'])
    too_much   = set(np.flatnonzero(counts > max_ratings))
    too_little = set(np.flatnonzero(counts < min_ratings))
    zero       = set(np.flatnonzero(counts == 0))
    bad_user = (too_much | too_little) - zero
    print_flush('Removing {} users'.format(len(bad_user)))
    data = np.array([x for x in data if x['user_id'] not in bad_user])
    print_flush('Left with {} ratings'.format(len(data)))
    return data

def constant_user_length(data, n):
    print_flush('Creating constant user length = {}'.format(n))
    data = remove_extreme_users(data, n, 1000)
    # sort by user_id, but within each user have a random order
    data = np.random.permutation(data)
    data = data[data['user_id'].argsort()]
    user_ids = data['user_id']
    # keep n last ratings from each user
    # due to the random permutation, it's n random ratings from each user
    take_last = n
    shifted_user_ids = np.append(user_ids[take_last:], [-1]*take_last)
    is_last = (user_ids != shifted_user_ids)
    data = data[is_last]
    print_flush('Left with {} ratings'.format(len(data)))
    return data

def remove_top_percentile(data):
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
    return remove_extreme_users(data, 2, max_items_per_user)

