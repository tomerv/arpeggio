#TODO: add imports, etc.

def load_epinions_data(path, verbose=True):
    if verbose:
        print_flush('Loading Epinions trust data...')
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
            user_id1, user_id2, _ = map(int, line.split())
            all_data[i] = (user_id1, user_id2, 1, 0)
    if verbose:
        print_flush('Loaded {} ratings'.format(len(all_data)))
        print_flush('Num users: {}'.format(np.max(all_data['user_id']+1)))
        print_flush('Num items: {}'.format(np.max(all_data['item_id']+1)))
        ratings = all_data['rating']
        print_flush('Min/mean/max rating: {}/{:.3}/{}'.format(
            np.min(ratings), np.mean(ratings), np.max(ratings)
        ))
    return all_data

# epinions_data = load_epinions_data('/home/tvromen/research/datasets/epinions/trust_data.txt')
# data = remove_extreme_users(epinions_data, 25, 1000)
# ratings = RatingsData.from_data(data, p_val=1, p_test=0, give_first=5)
