import sys
import h5py


def view(in_file):
    with h5py.File(in_file, 'r') as infh:
        keys = list(infh.keys())
        print('Sents: %d' % len(keys))
        for key in keys:
            n_words, n_layers, dim = infh[key].value.shape
            print('Words:{:>4} Layers:{:>2} Dim:{:>4}'.format(n_words,
                                                              n_layers,
                                                              dim)
                  )


if __name__ == '__main__':
    view(sys.argv[1])
