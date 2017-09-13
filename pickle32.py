# convert from pickle format 3 to pickle format 2 for use with Python 2.7

import sys
import gzip
import pickle

def main(argv):
    if len(argv)<2:
        print("usage: python pickle32.py file")
        return -1

    filename = argv[1]

    with gzip.open(filename, mode='rb') as f:
        unpickled = pickle.load(f)
        with gzip.open(filename + '2', mode='wb') as d:
            pickle.dump(unpickled, d, protocol=2)

if __name__ == '__main__':
    main(sys.argv)