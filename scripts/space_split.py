# coding: utf-8


def main(argv):
    out_f = open(argv.output_fn, "w")
    for line in open(argv.input_fn, "r"):
        line = list("".join(line.strip().split()))
        out_f.write('%s\n' % ' '.join(line))
    out_f.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Make spaced files')
    parser.add_argument('--input_fn')
    parser.add_argument('--output_fn')

    main(parser.parse_args())
