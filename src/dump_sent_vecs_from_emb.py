import argparse
import h5py
import numpy as np

UNK = '<UNK>'


def load_emb(path):
    word_list = []
    emb = []
    with open(path) as f:
        for line in f:
            line = line.rstrip().split()
            word_list.append(line[0])
            emb.append(line[1:])
    emb = np.asarray(emb, dtype=np.float32)

    if UNK not in word_list:
        word_list = [UNK] + word_list
        unk_vector = np.mean(emb, axis=0)
        emb = np.vstack((unk_vector, emb))

    word_dict = {w: i for i, w in enumerate(word_list)}
    return word_dict, emb


def str_to_id(sent, word_dict):
    return map(lambda w: word_dict[w] if w in word_dict else word_dict[UNK], sent)


def main(args):
    print('Loading embeddings...')
    word_dict, emb = load_emb(args.emb)

    print('Computing sentence vectors...')
    sentence_id = 0
    with open(args.data, 'r') as fin, h5py.File(args.output_file, 'w') as fout:
        for line in fin:
            if (sentence_id + 1) % 100 == 0:
                print("%d" % (sentence_id + 1), flush=True, end=" ")

            sentence = line.rstrip().split()
            word_ids = str_to_id(sentence, word_dict)
            embeddings = np.mean([emb[w] for w in word_ids], axis=0)

            fout.create_dataset(
                name=str(sentence_id),
                data=embeddings
            )
            sentence_id += 1
    print('Finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute nearest neighbors')
    parser.add_argument('--data', type=str, help='text file')
    parser.add_argument('--emb', type=str, help='word2vec file')
    parser.add_argument('--output_file', type=str, default='w2v.hdf5', help='text file')
    main(parser.parse_args())


