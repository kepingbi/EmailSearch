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

def read_doc_scores(rank_file, qdoc_dic):
    """ extract document scores under each query
        normalize the score; convert to int
    """
    with gzip.open(rank_file, 'rt') as frank:
        for line in frank:
            line = line.strip('\n')
            segs = line.split('\t')
            qid, uid, prev_qcount = map(int, segs[0].split('@'))
            ranklist = segs[1].split(';')
            doc_idxs = [int(x.split(':')[0]) for x in ranklist]
            doc_scores = [float(x.split(':')[2]) for x in ranklist]
            for x in ranklist:
                doc_id, doc_rating, doc_score = x.split(':')
                doc_score = float(doc_score)
                qdoc_dic[qid][int(doc_id)] = doc_score

def add_column_to_file(fname, foutname, qdoc_dic, min_score, max_score):
    fout = gzip.open(foutname, "wt") # evaluation in gz
    line_no = 0
    score_span = max_score - min_score
    with gzip.open(fname, "rt") as fin:
        line = fin.readline().strip('\r\n')
        feat_col_name = line.split('\t')
        feat_name_dic = {feat_col_name[i]: i for i in range(len(feat_col_name))}
        line += "\tContextModelScore\n"
        fout.write(line)
        qid_column = feat_name_dic['m:QueryId']
        doc_id_column = feat_name_dic['m:DocId']
        max_split = max(qid_column, doc_id_column) + 1
        for line in fin:
            line_no += 1
            if line_no % 100000 == 0:
                print("%d lines has been parsed!" % (line_no))
            line = line.strip('\r\n')
            segs = line.split('\t', maxsplit=max_split)
            qid = int(segs[qid_column])
            if qid not in qdoc_dic:
                # temporary
                continue
            doc_id = int(segs[doc_id_column])
            doc_score = 0
            if doc_id in qdoc_dic[qid]:
                doc_score = qdoc_dic[qid][doc_id]
                doc_score = (doc_score - min_score) / score_span
            # else:
            #     print(qid, doc_id)
            doc_score = int(doc_score * 4294967295) # maximum unsigned int 32
            line += "\t{}\n".format(doc_score)
            fout.write(line)
    fout.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_file', '-f', \
        default="/home/keping2/data/input/extract_sample0.10_hist_len11_feat_file.txt.gz")
    parser.add_argument('--data_path', '-d', default="/home/keping2/data/input/by_time")
    parser.add_argument('--rank_dir', '-r', \
        default="/home/keping2/data/working/pos_doc_context/by_users_usepopFalse_convTrue_docTrue")
    parser.add_argument('--option', \
        default="add_column", choices=["extract_feat", "add_column"], help='')
    paras = parser.parse_args()
    train_feat_file = os.path.join(paras.rank_dir, "train_feat.txt.gz")
    valid_feat_file = os.path.join(paras.rank_dir, "valid_feat.txt.gz")
    # uncompress train and valid yourself
    test_feat_file = os.path.join(paras.rank_dir, "test_feat.txt.gz")
    if paras.option == "extract_feat":
        train_qset, valid_qset, test_qset = set(), set(), set()
        train_qids = os.path.join(paras.data_path, "train_qids.txt.gz")
        valid_qids = os.path.join(paras.data_path, "valid_qids.txt.gz")
        test_qids = os.path.join(paras.data_path, "test_qids.txt.gz")
        read_qid_from_file(train_qids, train_qset)
        read_qid_from_file(valid_qids, valid_qset)
        read_qid_from_file(test_qids, test_qset)
        output_train_feat_file(paras.feat_file, train_qset, train_feat_file)
        output_train_feat_file(paras.feat_file, valid_qset, valid_feat_file)
        output_test_feat_file(paras.feat_file, test_qset, test_feat_file)
    elif paras.option == "add_column":
        test_rank_file = os.path.join(paras.rank_dir, "test.best_model.ranklist.gz")
        valid_rank_file = os.path.join(paras.rank_dir, "valid.best_model.ranklist.gz")
        train_rank_file = os.path.join(paras.rank_dir, "train.best_model.ranklist.gz")
        qdoc_dic = defaultdict(dict)
        read_doc_scores(train_rank_file, qdoc_dic)
        read_doc_scores(valid_rank_file, qdoc_dic)
        read_doc_scores(test_rank_file, qdoc_dic)
        max_score = max([max(qdoc_dic[qid].values()) for qid in qdoc_dic])
        min_score = min([min(qdoc_dic[qid].values()) for qid in qdoc_dic])
        print(min_score, max_score)
        outfname = os.path.splitext(os.path.splitext(paras.feat_file)[0])[0] + "_context.txt.gz"
        # out_file = os.path.join(os.path.dirname(paras.feat_file), outfname)
        out_file = os.path.join(paras.rank_dir, outfname)
        add_column_to_file(paras.feat_file, out_file, qdoc_dic, min_score, max_score)
        # test_out_file = os.path.join(paras.rank_dir, "test_context_feat.txt.gz")
        # valid_out_file = os.path.join(paras.rank_dir, "valid_context_feat.txt.gz")
        # train_out_file = os.path.join(paras.rank_dir, "train_context_feat.txt.gz")
        # add_column_to_file(train_feat_file, train_out_file, qdoc_dic, min_score, max_score)
        # add_column_to_file(valid_feat_file, valid_out_file, qdoc_dic, min_score, max_score)
        # add_column_to_file(test_feat_file, test_out_file, qdoc_dic, min_score, max_score)

if __name__ == "__main__":
    main()
