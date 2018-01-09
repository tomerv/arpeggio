if __name__ == '__main__':
    from Utils import print_flush, IdAssigner
else:
    import sys
    import os
    PACKAGE_PARENT = '..'
    SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
    sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
    from Utils import print_flush, IdAssigner
import gzip
import numpy as np
import struct


class WordEmbedding:
    def __init__(self, path, max_vocab_size=None, verbose=True):
        if verbose:
            print_flush('Loading word embedding...')
        self.vocab = IdAssigner()
        with gzip.open(path, 'rb') as f:
            word_count, dim = map(int, f.readline().split())
            # +2 for the "padding" and "unknown" words
            word_count += 2
            if max_vocab_size:
                word_count = min(word_count, max_vocab_size)
            self.dimension = dim
            self.embedding = np.zeros([word_count, dim], dtype=np.float32)
            # First word is "padding"
            pad_id = self.vocab.get_id('<PAD>')
            assert pad_id == 0
            # default in self.embedding is already zeros
            # self.embedding[pad] = np.zeros([dim], dtype=np.float32)
            # Second word is "out of vocabulary"
            unk_id = self.vocab.get_id('<UNK>')
            assert unk_id == 1
            # default in self.embedding is already zeros
            # self.embedding[unk_id] = np.zeros([dim], dtype=np.float32)
            if verbose:
                print_flush('Loading {} words with embedding dimension {}'.format(word_count, dim))
            format_string = 'f'*dim
            sz = struct.calcsize(format_string)
            i = self.vocab.get_next_id()
            while True:
                encoded = b''
                c = f.read(1)
                while c != ' '.encode():
                    encoded += c
                    c = f.read(1)
                try:
                    word = encoded.decode('utf-8')
                except UnicodeDecodeError as e:
                    if verbose:
                        print_flush('Error decoding this sequence: {} (skipping).'.format(encoded))
                if '_' in word:
                    # only load single words, not phrases
                    f.seek(sz, 1)
                    continue
                word_lower = word.lower()
                if word_lower in self.vocab.forward:
                    # already saw this word
                    f.seek(sz, 1)
                    continue
                word_id = self.vocab.get_id(word_lower)
                val = np.array(struct.unpack(format_string, f.read(sz)), dtype=np.float32)
                self.embedding[word_id] = val
                if verbose and i % 10000 == 0:
                    print_flush('{}... ({})'.format(i, word_lower))
                i += 1
                if i == max_vocab_size:
                    break
            #assert f.read() == ''
        if verbose:
            print_flush('Done. Loaded {} words in total'.format(i))

if __name__ == '__main__':
    path = '/home/tvromen/research/GoogleNews-vectors-negative300.bin.gz'
    word2vec = WordEmbedding(path, max_vocab_size=100000, verbose=True)


