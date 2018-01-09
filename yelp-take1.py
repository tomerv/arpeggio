import sys
sys.path.append('/home/tvromen/research')
from Common.Utils import IdAssigner, print_flush
from Common import RatingsData
import numpy as np
import json
from datetime import datetime


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
        unique = set()
        f.seek(0)
        for i,line in enumerate(f):
            if i == num_lines:
                break
            if verbose and ((i+1) % 100000) == 0:
                print_flush('  Loaded {} ratings...'.format(i+1))
            data = json.loads(line)
            user_id = user2id.get_id(data['user_id'])
            item_id = item2id.get_id(data['business_id'])
            # rating = (data['stars'] - 1) / 4.0
            rating = data['stars'] / 5.0
            # too slow:
            # timestamp = datetime.strptime(data['date'], '%Y-%m-%d').toordinal()
            year, month, day = map(int, data['date'].split('-'))
            timestamp = datetime(year=year, month=month, day=day).toordinal()
            all_data[i] = (user_id, item_id, rating, timestamp)
            t = (user_id, item_id)
            if t in unique:
                print_flush('Multiple ratings for user {} on item {}'.format(data['user_id'], data['business_id']))
            unique.add(t)
    if verbose:
        print_flush('Loaded {} ratings'.format(len(all_data)))
        print_flush('Num users: {}'.format(user2id.get_next_id()))
        print_flush('Num items: {}'.format(item2id.get_next_id()))
        ratings = all_data['rating']
        print_flush('Min/mean/max rating: {}/{:.3}/{}'.format(
            np.min(ratings), np.mean(ratings), np.max(ratings)
        ))
    return all_data

max_lines = None
# max_lines = 500000
yelp_data = load_yelp('/home/tvromen/research/datasets/yelp/review.json', max_lines)
yelp_data = RatingsData.remove_top_percentile(yelp_data)

# Need to remap user IDs to be sequential
print_flush('Reassigning user IDs')
user2id = IdAssigner()
for i in range(len(yelp_data)):
    yelp_data[i]['user_id'] = user2id.get_id(yelp_data[i]['user_id'])

np.random.seed(1234)
ratings = RatingsData.RatingsData.from_data(yelp_data)

ratings.output_as_text(ratings.train, 'yelp-take1.train.txt')
ratings.output_as_text(ratings.val,   'yelp-take1.val.txt')
ratings.output_as_text(ratings.test,  'yelp-take1.test.txt')

