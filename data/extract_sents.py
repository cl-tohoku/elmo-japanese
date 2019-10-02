import argparse
import os
import glob


def get_file_paths_in_dir(dir_path):
    file_paths = glob.glob(dir_path + '/*')
    file_paths.sort()
    # Remove LICENCE.txt
    file_paths = file_paths[1:]
    return file_paths


def extract_sentences(file_paths):
    sents = []
    for file_path in file_paths:
        doc = []
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    doc.append(line)
        # Remove (0) URL, (1) Date, and (2) Title
        doc = doc[3:]
        sents.append("".join(doc))
    return sents


def save_sentences(label_sents_pairs, file_name='dataset'):
    f_sent = open(file_name + '.sent.txt', 'w')
    f_label = open(file_name + '.label.txt', 'w')
    for label, sents in label_sents_pairs:
        for sent in sents:
            sent.replace('\t', '　')
            f_sent.write('%s\n' % sent)
            f_label.write('%s\n' % label)
    f_sent.close()
    f_label.close()


def main(args):
    label_sents_pairs = []
    for dir_name in ['dokujo-tsushin', 'it-life-hack', 'kaden-channel',
                     'livedoor-homme', 'movie-enter', 'peachy',
                     'smax', 'sports-watch', 'topic-news']:
        dir_path = os.path.join(args.dir_path, dir_name)
        file_paths = get_file_paths_in_dir(dir_path)
        sents = extract_sentences(file_paths)[:args.data_size]
        label_sents_pairs.append((dir_name, sents))
    save_sentences(label_sents_pairs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract sentences')
    parser.add_argument('--dir_path', type=str, help='path to directory')
    parser.add_argument('--data_size', type=int, default=100000000, help='data size')
    main(parser.parse_args())
