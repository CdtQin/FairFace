import os
import sys
import timeit
import math
import numpy as np
from datetime import datetime as dt
from collections import defaultdict

def load_feat_file(feat_file_path):
    feats2index = {}
    with open(feat_file_path, 'r')  as f:
        for line_i, line in enumerate(f):
            line_s = line.rstrip().split(' ')
            feat = [float(x) for x in line_s[1:]]
            index = int(line_s[0].split('/')[-1].split('.')[0])
            feat = np.array(feat).astype('float32')
            feats2index[index] = feat
            if line_i > 0 and line_i % 2000 == 0:
                print("{}: {} lines read".format(dt.now(), line_i))
    return feats2index

if __name__ == "__main__":
    feature_path = sys.argv[1]
    pair_list = "fair_test_predictions.csv"
    predictions_result = "./predictions.csv"
    features_dict = load_feat_file(feature_path)
    print(len(features_dict))
    with open(pair_list, 'r') as reader, open(predictions_result, 'w') as writer:
        first_line = next(reader)
        writer.write(first_line)
        for line in reader:
            line = line.strip().split(',')
            first, second, similarity = int(line[0]), int(line[1]), float(line[2])
            first_feature = features_dict[first]
            second_feature = features_dict[second]
            first_feature /= np.linalg.norm(first_feature)
            second_feature /= np.linalg.norm(second_feature)
            similarity = np.dot(first_feature, second_feature)
            writer.write('%d,%d,%.17f\n'%(first, second, similarity))
