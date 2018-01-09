if __name__ == '__main__':
    from Utils import print_flush
    from WordEmbedding import WordEmbedding
else:
    import sys
    import os
    PACKAGE_PARENT = '..'
    SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
    sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
    from Utils import print_flush
import numpy as np

class Subtitles:
    """
    Class that is in charge of subtitles
    Load subtitles for each movie, and encode words using word2id
    """
    def __init__(self, data_file, word2id, num_scenes, words_per_scene, max_movies=None, verbose=True):
        if verbose:
            print_flush('Loading subtitles...')
        self.subs = dict()
        with open(data_file) as f:
            for i,line in enumerate(f):
                if verbose and i % 500 == 0:
                    print_flush('{}...'.format(i))
                if i == max_movies:
                    break
                line = line.replace("'", " ' ") # TODO
                words = line.strip().lower().split()
                movie_id = int(words[0])
                text = ' '.join(words[1:])
                encoded_scenes = np.zeros([num_scenes, words_per_scene], dtype=np.uint32)
                # padding has id 0, so we don't need to do anything for it
                scenes = text.split('</scene>')
                unk_id = word2id['<UNK>']
                def id_getter(word):
                    return (word2id[word] if word in word2id else unk_id)
                for j,scene in enumerate(scenes):
                    if j == num_scenes:
                        break
                    words = scene.split()[:words_per_scene]
                    encoded_scenes[j,:len(words)] = list(map(id_getter, words))
                self.subs[movie_id] = encoded_scenes
        if verbose:
            print_flush('Done')


if __name__ == '__main__':
    path = '/home/tvromen/research/GoogleNews-vectors-negative300.bin.gz'
    word2vec = WordEmbedding(path, max_vocab_size=300000, verbose=True)
    path = '/home/tvromen/research/subtitles2/movielens-subtitles-scenes-150s.txt'
    subs = Subtitles(path, word2vec.vocab.forward, num_scenes=48, words_per_scene=512, max_movies=1000, verbose=True)
    print(subs.subs[1])

