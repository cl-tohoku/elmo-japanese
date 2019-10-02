import argparse
import h5py
import numpy as np


def load_data(file_path):
    with open(file_path) as f:
        return [line.rstrip().split('\t') for line in f]


def load_elmo(file_path):
    with h5py.File(file_path, 'r') as infh:
        return [infh[str(key)].value for key in range(len(list(infh.keys())))]


def main(args):
    dataset = load_data(args.data)
    sent_vecs = load_elmo(args.elmo)

    sent_id = 0
    for sent_src, label_src in dataset:
        scores = []
        vec = sent_vecs[sent_id]
        l2_norm = np.sqrt(np.sum(vec ** 2))
        for index, e in enumerate(sent_vecs):
            cos = np.sum(vec * e) / (l2_norm * np.sqrt(np.sum(e ** 2)))
            scores.append(cos)
        rank = np.argsort(scores)[::-1]

        print(label_src)
        # print(sent_src)
        for i, r in enumerate(rank[1:11]):
            sent, label = dataset[r]
            print('%d\t%f\t%s' % (i + 1, scores[r], label))
            # print('%d\t%f\t%s\t%s' % (i + 1, scores[r], label, sent))
        print()
        sent_id += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute nearest neighbors')
    parser.add_argument('--data', type=str, help='text file')
    parser.add_argument('--elmo', type=str, help='elmo file')
    main(parser.parse_args())
