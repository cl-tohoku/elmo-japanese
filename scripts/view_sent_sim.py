import argparse
import h5py
import numpy as np


def main(args):
    with h5py.File(args.elmo, 'r') as infh:
        embs = [infh[str(key)].value for key in range(len(list(infh.keys())))]

    sent_index = 0
    sents = [line.rstrip() for line in open(args.data)]
    for sent in sents:
        print(sent)
        scores = []
        emb = embs[sent_index]
        l2_norm = np.sqrt(np.sum(emb ** 2))
        for index, e in enumerate(embs):
            cos = np.sum(emb * e) / (l2_norm * np.sqrt(np.sum(e ** 2)))
            scores.append(cos)
        rank = np.argsort(scores)[::-1]
        for i, r in enumerate(rank[1:]):
            print('%d\t%f\t%s' % (i + 1, scores[r], sents[r]))
        print()
        sent_index += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute test perplexity')
    parser.add_argument('--data', type=str, help='text file')
    parser.add_argument('--elmo', type=str, help='elmo file')

    main(parser.parse_args())
