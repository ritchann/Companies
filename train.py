import os
import random
import numpy as np
from sklearn.utils.fixes import sklearn
import argparse
from model import is_duplicate_levenshtein_distance, similar_names_tf_idf, similar_names_word2_vec

if __name__ == '__main__':
    SEED = 1988

    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    np.random.seed(SEED)
    sklearn.random.seed(SEED)

    parser = argparse.ArgumentParser(description='Training script.')
    parser.add_argument('--m', type=str, default='', help='Levenshtein distance.')
    parser.add_argument('--name1', type=str, default='', help='Name 1.')
    parser.add_argument('--name2', type=str, default='', help='Name 2.')
    args = parser.parse_args()

    if args.m == 'ld':
        is_duplicate_levenshtein_distance(args.name1, args.name2)
    elif args.m == 'tf':
        similar_names_tf_idf(args.name1)
    elif args.m == 'w2':
        similar_names_word2_vec(args.name1)
