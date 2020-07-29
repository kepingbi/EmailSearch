''' Convert features and replace the original features. 
    Only include the features that are needed in the model. 
'''
import os
import sys
import argparse
import gzip
import random
import glob
from collections import defaultdict

RATING_MAP = {"Bad":0, "Fair":1, "Good":2, "Excellent":3, "Perfect":4}

def output_train_feat_file(feat_file, qset, output_feat_file):
    ''' parse the initial feature file, compute the converted features
        output the final features to a new file
    '''
    fout = gzip.open(output_feat_file, "wt") # training in tsv
    #ftest = gzip.open(output_feat_file+".gz", "wt") # evaluation in gz
    line_no = 0
    with gzip.open(feat_file, "rt") as fin:
        line = fin.readline().strip('\r\n')
        feat_col_name = line.split('\t')
        feat_name_dic = {feat_col_name[i]: i for i in range(len(feat_col_name))}
        qid_column = feat_name_dic['m:QueryId']
        max_split = qid_column + 1
        rating_column = feat_name_dic['m:Rating']
        head_str_arr = ["m:Rating", "m:QueryId", "ContextId"]
        feat_columns = ["Recency"] + ["DF_Term{}".format(i) for i in range(6)]
        excluded_feats = ["AdvancedPreferFeature_123", "AdvancedPreferFeature_11"]
        excluded_feats += ["NumMailboxDocuments"]
        excluded_feats += ["DocumentFrequency_{}U".format(i) for i in range(6)]
        excluded_feats += ["DocumentFrequency_{}L".format(i) for i in range(6)]
        excluded_feats = set(excluded_feats)
        other_columns = [x for x in feat_col_name \
            if x not in excluded_feats and not x.startswith('m:')]
        head_str_arr += feat_columns
        head_str_arr += other_columns
        fout.write("%s\n" % "\t".join(head_str_arr)) # head line
        for line in fin:
            line_no += 1
            if line_no % 100000 == 0:
                print("%d lines has been parsed!" % (line_no))
            line = line.strip('\r\n')
            segs = line.split('\t', maxsplit=max_split)
            if not segs[qid_column].isdigit():
                print("Illegal: %s" % line)
                continue
            qid = int(segs[qid_column])
            if qid not in qset:
                continue
            segs = line.split('\t')
            rating = RATING_MAP[segs[rating_column]]
            new_segs = [rating, qid, 1.0]
            new_segs = [str(x) for x in new_segs]
            new_segs += convert_features(segs, feat_name_dic)
            new_segs += [segs[feat_name_dic[x]] for x in other_columns]
            fout.write("%s\n" % ("\t".join(new_segs)))
    fout.close()

def output_test_feat_file(feat_file, qset, output_feat_file):
    ''' parse the initial feature file, compute the converted features
        output the final features to a new file
    '''
    fout = gzip.open(output_feat_file, "wt") # evaluation in gz
    line_no = 0
    with gzip.open(feat_file, "rt") as fin:
        line = fin.readline().strip('\r\n')
        feat_col_name = line.split('\t')
        feat_name_dic = {feat_col_name[i]: i for i in range(len(feat_col_name))}
        feat_columns = ["Recency"] + ["DF_Term{}".format(i) for i in range(6)]
        feat_col_name += feat_columns
        qid_column = feat_name_dic['m:QueryId']
        max_split = qid_column + 1
        fout.write("%s\n" % "\t".join(feat_col_name)) # head line
        for line in fin:
            line_no += 1
            if line_no % 100000 == 0:
                print("%d lines has been parsed!" % (line_no))
            line = line.strip('\r\n')
            segs = line.split('\t', maxsplit=max_split)
            if not segs[qid_column].isdigit():
                print("Illegal: %s" % line)
                continue
            qid = int(segs[qid_column])
            if qid not in qset:
                continue
            segs = line.split('\t')
            new_segs = segs
            new_segs += convert_features(segs, feat_name_dic)
            fout.write("%s\n" % ("\t".join(new_segs)))
    fout.close()

def convert_features(segs, feat_name_dic):
    search_time = int(segs[feat_name_dic['AdvancedPreferFeature_123']])
    created_time = int(segs[feat_name_dic['AdvancedPreferFeature_11']])
    recency = search_time - created_time
    new_segs = [str(recency)]
    mailbox_doc_count = float(segs[feat_name_dic['NumMailboxDocuments']])
    for i in range(6):
        doc_freq_u = float(segs[feat_name_dic['DocumentFrequency_{}U'.format(i)]])
        doc_freq_l = float(segs[feat_name_dic['DocumentFrequency_{}L'.format(i)]])
        df = 0 if mailbox_doc_count == 0 else max(doc_freq_u, doc_freq_l)/mailbox_doc_count
        df = int(df * 4294967295) # maximum unsigned int 32
        new_segs.append(str(df))
    return new_segs

def read_qid_from_file(fname, q_set):
    with gzip.open(fname, 'rt') as fin:
        line = fin.readline() # head line
        for line in fin:
            line = line.strip()
            qid, uid, _ = line.split() # qid, uid, search_time
            q_set.add(int(qid))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_file', '-f', \
        default="/home/keping2/data/input/extract_sample0.10_hist_len11_feat_file.txt.gz")
    parser.add_argument('--data_path', '-d', default="/home/keping2/data/input/by_time")
    paras = parser.parse_args()
    train_qset, valid_qset, test_qset = set(), set(), set()
    train_qids = os.path.join(paras.data_path, "train_qids.txt.gz")
    valid_qids = os.path.join(paras.data_path, "valid_qids.txt.gz")
    test_qids = os.path.join(paras.data_path, "test_qids.txt.gz")
    read_qid_from_file(train_qids, train_qset)
    read_qid_from_file(valid_qids, valid_qset)
    read_qid_from_file(test_qids, test_qset)
    train_feat_file = os.path.join(paras.data_path, "train_feat.txt.gz")
    valid_feat_file = os.path.join(paras.data_path, "valid_feat.txt.gz")
    # uncompress train and valid yourself
    test_feat_file = os.path.join(paras.data_path, "test_feat.txt.gz")
    output_train_feat_file(paras.feat_file, train_qset, train_feat_file)
    output_train_feat_file(paras.feat_file, valid_qset, valid_feat_file)
    output_test_feat_file(paras.feat_file, test_qset, test_feat_file)

if __name__ == "__main__":
    main()
