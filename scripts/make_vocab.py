from collections import defaultdict


def main(argv):
    vocab = defaultdict(int)
    with open(argv.input_fn) as in_f:
        for line in in_f:
            for w in line.rstrip().split():
                vocab[w] += 1

    out_f = open(argv.output_fn, "w")
    out_f.write("</S>\n")
    out_f.write("<S>\n")
    out_f.write("<UNK>\n")

    vocab_list = []
    min_count = argv.min_count
    for w, v in sorted(vocab.items(), key=lambda x: x[-1], reverse=True):
        if v < min_count:
            break
        vocab_list.append((w, v))

    if sum([len(w) for w, v in vocab_list]) == len(vocab_list):
        is_char = True
    else:
        is_char = False

    if is_char:
        vocab_list.sort()
        for c, v in vocab_list:
            out_f.write(c + "\n")
    else:
        for w, v in vocab_list:
            out_f.write(w + "\n")

    out_f.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Make vocab files')
    parser.add_argument('--input_fn')
    parser.add_argument('--output_fn')
    parser.add_argument('--min_count', type=int, default=0)

    main(parser.parse_args())
