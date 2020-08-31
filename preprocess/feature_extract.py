''' Convert features and replace the original features.
    Only include the features that are needed in the model.
'''
import os
import sys
import argparse
import gzip
import random
import glob
import numpy as np
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
import fastcluster
from scipy.cluster.hierarchy import fcluster
from collections import defaultdict
import time
import pandas as pd
# from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
np.random.seed(4711)  # for repeatability

RATING_MAP = {"Bad":0, "Fair":1, "Good":2, "Excellent":3, "Perfect":4}
def str2bool(val):
    ''' parse bool type input parameters
    '''
    if val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def output_lightgbm_file(feat_file, qset, output_feat_file):
    ''' parse the initial feature file, compute the converted features
        sparse format. 1 0 1 0 -> 1:1 3:2
    '''
    fout = open(output_feat_file, "w") # training in txt
    line_no = 0
    qdoc_count_dic = defaultdict(int)
    qid_list = []
    with gzip.open(feat_file, "rt") as fin:
        line = fin.readline().strip('\r\n')
        feat_col_name = line.split('\t')
        feat_name_dic = {feat_col_name[i]: i for i in range(len(feat_col_name))}
        qid_column = feat_name_dic['m:QueryId']
        max_split = qid_column + 1
        rating_column = feat_name_dic['m:Rating']
        head_str_arr = ["Rating", "QueryId", "DocId", "RelevancePosition", "DateTimePosition"]
        # cannot start with "m:"
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
            if qdoc_count_dic[qid] == 0:
                qid_list.append(qid)
            qdoc_count_dic[qid] += 1
            segs = line.split('\t')
            docid = segs[feat_name_dic["m:DocId"]]
            rel_pos = segs[feat_name_dic["m:RelevancePosition"]]
            datetime_pos = segs[feat_name_dic["m:DateTimePosition"]]
            rating = RATING_MAP[segs[rating_column]]
            new_segs = [rating, qid, docid, rel_pos, datetime_pos]
            new_segs = [str(x) for x in new_segs]
            new_segs += convert_features(segs, feat_name_dic)
            new_segs += [segs[feat_name_dic[x]] for x in other_columns]
            fout.write("%s\n" % ("\t".join(new_segs)))
    fout.close()
    with open(output_feat_file + ".query", "w") as fout:
        for qid in qid_list:
            fout.write("%d\n" % qdoc_count_dic[qid])

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

def convert_features(segs, feat_name_dic, scale=1000000):
    search_time = int(segs[feat_name_dic['AdvancedPreferFeature_123']])
    created_time = int(segs[feat_name_dic['AdvancedPreferFeature_11']])
    recency = search_time - created_time
    new_segs = [str(recency)]
    mailbox_doc_count = float(segs[feat_name_dic['NumMailboxDocuments']])
    for i in range(6):
        doc_freq_u = float(segs[feat_name_dic['DocumentFrequency_{}U'.format(i)]])
        doc_freq_l = float(segs[feat_name_dic['DocumentFrequency_{}L'.format(i)]])
        df = 0 if mailbox_doc_count == 0 else max(doc_freq_u, doc_freq_l)/mailbox_doc_count
        df = int(df * scale) #4294967295 # maximum unsigned int 32
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
            query_infos = segs[0].split('@')
            qid = int(query_infos[0])
            uid = query_infos[1] # too large to be converted to int
            prev_qcount = int(query_infos[2])
            ranklist = segs[1].split(';')
            doc_idxs = [int(x.split(':')[0]) for x in ranklist]
            doc_scores = [float(x.split(':')[2]) for x in ranklist]
            for x in ranklist:
                doc_id, doc_rating, doc_score = x.split(':')
                doc_score = float(doc_score)
                qdoc_dic[qid][int(doc_id)] = doc_score

def add_column_to_file(fname, foutname, qdoc_dic, min_score, max_score, scale=1000000):
    ''' add converted doc scores to the feature file as an additional feature
    '''
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
                # the first query in the feature file is missing; not clear why
                continue
            doc_id = int(segs[doc_id_column])
            doc_score = 0
            if doc_id in qdoc_dic[qid]:
                doc_score = qdoc_dic[qid][doc_id]
                doc_score = (doc_score - min_score) / score_span
            # else:
            #     print(qid, doc_id)
            doc_score = int(doc_score * scale) # 4294967295 maximum unsigned int 32
            line += "\t{}\n".format(doc_score)
            fout.write(line)
    fout.close()

def read_context_emb(rank_file, context_emb_matrix, qlist, qcontext_dic, uq_dic):
    """ extract context embedding of each query
    """
    with gzip.open(rank_file, 'rt') as frank:
        for line in frank:
            line = line.strip('\n')
            segs = line.split('\t')
            query_infos = segs[0].split('@')
            qid = int(query_infos[0])
            uid = query_infos[1]
            prev_qcount = int(query_infos[2])
            context_emb = [float(x) for x in query_infos[3].split(',')]
            qcontext_dic[qid] = len(context_emb_matrix)
            context_emb_matrix.append(context_emb)
            uq_dic[uid].append(qid)
            qlist.append(qid)

def read_meta_info(feat_name, qid_dic):
    line_no = 0
    user_info = []
    cur_qid_meta_dic = dict()
    cur_qid_action_dic = dict()
    with gzip.open(feat_name, "rt") as fin:
        line = fin.readline().strip('\r\n')
        feat_col_name = line.split('\t')
        feat_name_dic = {feat_col_name[i]: i for i in range(len(feat_col_name))}
        qid_column = feat_name_dic['m:QueryId']
        max_split = qid_column + 1
        for line in fin:
            line_no += 1
            if line_no % 100000 == 0:
                print("%d lines has been parsed!" % (line_no))
            segs = line.split('\t', maxsplit=max_split)
            qid = int(segs[feat_name_dic['m:QueryId']])
            if qid not in qid_dic:
                continue
            segs = line.split('\t')
            item_type = int(segs[feat_name_dic['QueryLevelFeature_1997']])
            user_type = int(segs[feat_name_dic['QueryLevelFeature_2002']])
            q_lang_hash = int(segs[feat_name_dic['QueryLevelFeature_2001']])
            culture_lcid = int(segs[feat_name_dic['QueryLevelFeature_1999']])
            locale_lcid = int(segs[feat_name_dic['QueryLevelFeature_1998']])
            rating = RATING_MAP[segs[feat_name_dic['m:Rating']]]
            if qid not in cur_qid_meta_dic:
                cur_qid_meta_dic[qid] = [item_type, user_type, \
                    q_lang_hash, culture_lcid, locale_lcid]
            if qid not in cur_qid_action_dic:
                cur_qid_action_dic[qid] = defaultdict(int)
            if rating > 0:
                actions = segs[feat_name_dic['m:StrongAction']].split(';')
                for action in actions:
                    cur_qid_action_dic[qid][action] += 1
    for qid in cur_qid_action_dic:
        sorted_action = sorted(cur_qid_action_dic[qid], key=cur_qid_action_dic[qid].get, reverse=True)
        action_type = sorted_action[0] if len(sorted_action) > 0 else "None"
        if len(sorted_action) > 1 and sorted_action[0] == "ReadingPaneDisplayLongDwell":
            action_type = sorted_action[1]
        cur_qid_meta_dic[qid].append(action_type)
    return cur_qid_meta_dic

def add_context_to_file(fname, foutname, context_emb, id_dic, feat_type="QContext"):
    ''' add context emb to the feature file as additional features
        context_emb can be query_context_emb or user_context_emb
        id_dic can be query or user to row in context_emb
        feat_type is QContext or UContext
    '''
    print("Adding Context Information to the Feature File")
    fout = gzip.open(foutname, "wt") # evaluation in gz
    line_no = 0
    embed_size = context_emb.shape[1]
    with gzip.open(fname, "rt") as fin:
        line = fin.readline().strip('\r\n')
        feat_col_name = line.split('\t')
        feat_name_dic = {feat_col_name[i]: i for i in range(len(feat_col_name))}
        feat_col_name += ["{}_{}".format(feat_type, x) for x in range(embed_size)]
        fout.write("{}\n".format("\t".join(feat_col_name)))
        qid_column = feat_name_dic['m:QueryId']
        user_id_column = feat_name_dic['m:MailboxId']
        max_split = max(qid_column, user_id_column) + 1
        for line in fin:
            line_no += 1
            if line_no % 100000 == 0:
                print("%d lines has been parsed!" % (line_no))
            line = line.strip('\r\n')
            segs = line.split('\t', maxsplit=max_split)
            qid = int(segs[qid_column])
            u_str = segs[user_id_column]
            key = qid if "Q" in feat_type else u_str
            if key in id_dic:
                cur_context_arr = context_emb[id_dic[key]].tolist()
            else:
                print(key)
                cur_context_arr = [4294967295] * embed_size # -1 in unsigned int
            cur_context_arr = [str(int(x)) for x in cur_context_arr]
            line += "\t{}\n".format("\t".join(cur_context_arr))
            fout.write(line)
    fout.close()

def add_context_combination_to_file(\
    fname, foutname, cluster_id_matrix, id_dic, \
        feat_list=["Recency", "BM25f", "EmailSize"], context_type="QCluster"):
    ''' Put important features to corresponding cluster slot in the feature file as additional features
        cluster_id_matrx contains the cluster id for each query or user.
        id_dic can be query or user to row in context_emb
        feat_list includes what features to put to different cluster slot.
    '''
    print("Putting Important Features to corresponding slots in the feature file!")
    fout = gzip.open(foutname, "wt") # evaluation in gz
    line_no = 0
    n_clusters = len(np.unique(cluster_id_matrix))
    assert cluster_id_matrix.shape[1] == 1
    with gzip.open(fname, "rt") as fin:
        line = fin.readline().strip('\r\n')
        feat_col_name = line.split('\t')
        feat_name_dic = {feat_col_name[i]: i for i in range(len(feat_col_name))}
        if "BM25f_simpleEmail" in feat_name_dic:
            feat_name_dic["BM25f_simple"] = feat_name_dic["BM25f_simpleEmail"]
        for feat_type in feat_list:
            feat_col_name += ["{}_{}_{}".format(feat_type, context_type, x) for x in range(n_clusters)]
        fout.write("{}\n".format("\t".join(feat_col_name)))
        qid_column = feat_name_dic['m:QueryId']
        user_id_column = feat_name_dic['m:MailboxId']
        max_split = max(qid_column, user_id_column) + 1
        for line in fin:
            line_no += 1
            if line_no % 100000 == 0:
                print("%d lines has been parsed!" % (line_no))
            line = line.strip('\r\n')
            # segs = line.split('\t', maxsplit=max_split)
            segs = line.split('\t')
            qid = int(segs[qid_column])
            u_str = segs[user_id_column]
            search_time = int(segs[feat_name_dic['AdvancedPreferFeature_123']])
            created_time = int(segs[feat_name_dic['AdvancedPreferFeature_11']])
            recency = search_time - created_time
            BM25f = int(segs[feat_name_dic["BM25f_simple"]])
            email_size = int(segs[feat_name_dic['AdvancedPreferFeature_17']])
            key = qid if "Q" in context_type else u_str
            cur_context_arr = [0] * len(feat_list) * n_clusters
            if key in id_dic:
                cluster_id = cluster_id_matrix[id_dic[key]][0] # the only number
                for idx, feat_type in enumerate(feat_list):
                    feat_val = 0
                    if feat_type == "Recency":
                        feat_val = recency
                    elif feat_type == "BM25f":
                        feat_val = BM25f
                    elif feat_type == "EmailSize":
                        feat_val = email_size
                    cur_context_arr[idx * n_clusters + cluster_id] = feat_val
            cur_context_arr = [str(int(x)) for x in cur_context_arr]
            line += "\t{}\n".format("\t".join(cur_context_arr))
            fout.write(line)
    fout.close()


def fit_tsne(context_matrix, df, figname=''):
    # feat_cols = ['feat'+str(i) for i in range(context_matrix.shape[1])]
    # df = pd.DataFrame(context_matrix, columns=feat_cols)
    # df['y'] = cluster_id_matrix
    # randperm = np.random.permutation(context_matrix.shape[0])
    # #df['y'] = y
    # #df['label'] = df['y'].apply(lambda i: str(i))
    # df_subset = df.loc[randperm[:sample_size]].copy()
    # df_subset = df
    # data_subset = df_subset[feat_cols].values
    
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(context_matrix)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    df['tsne-2d-one'] = tsne_results[:, 0]
    df['tsne-2d-two'] = tsne_results[:, 1]
    column_names = ["cluster_id", "item_type", "user_type", \
        "q_lang_hash", "culture_lcid", "locale_lcid", "action"]
    for label in column_names:
        n_colors = len(np.unique(df[label]))
        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue=label,
            palette=sns.color_palette("hls", n_colors),
            data=df,
            legend="full",
            alpha=0.3
        )
        plt.savefig("tsne_figs/tsne_%s_%s%d.pdf" % (figname, label, n_colors))

def cluster_context_emb(X, n_clusters, cluster_method="MiniBatchKmeans", way="hard"):
    """ cluster each context to clusters
        X: sample_size, feature_count
        cluster_method
        way: hard - output the cluster id
             soft - output the probability of the sample being in each cluster
    """
    separate_set = set(["AgglCluster", "DBSCAN", "FastCluster"])
    if cluster_method == "KMeans":
        model = KMeans(n_clusters=n_clusters)
        model.fit(X)
    elif cluster_method == "MiniBatchKMeans":
        model = MiniBatchKMeans(n_clusters=n_clusters)
        model.fit(X)
    elif cluster_method == "GaussianMixture":
        model = GaussianMixture(n_components=n_clusters)
        model.fit(X)
    elif cluster_method == "AgglCluster":
        model = AgglomerativeClustering(n_clusters=n_clusters)
        y = model.fit_predict(X) # no probability
    elif cluster_method == "DBSCAN":
        model = DBSCAN(eps=3, min_samples=30)
        y = model.fit_predict(X) # no probability
    elif cluster_method == "FastCluster":
        Z = fastcluster.linkage(X, method='single', metric='euclidean', preserve_input=True)
        y = fcluster(Z, n_clusters, criterion='maxclust')
    if cluster_method not in separate_set:
        if not way == "soft":
            y = model.predict(X)
            y = y.reshape(X.shape[0], 1)
        else:
            if cluster_method == "GaussianMixture":
                y = model.predict_proba(X)
            else:
                y = model.transform(X)
            y = convert_to_int(y)
    return y # shape: [sample_size, ] or [sample_size, n_clusters]

def collect_ucontext_emb(context_emb_matrix, qcontext_dic, uq_dic):
    embed_size = context_emb_matrix.shape[1]
    context_emb_matrix = np.concatenate([context_emb_matrix, np.zeros((1, embed_size))], axis=0)
    # the last vector is the padding vector
    uid_dic = dict()
    u_qid_matrix = []
    for uid in uq_dic:
        uid_dic[uid] = len(u_qid_matrix)
        u_qids = [qcontext_dic[x] for x in uq_dic[uid]]
        u_qid_matrix.append(u_qids)
    width = max(len(d) for d in u_qid_matrix)
    u_qid_matrix = [d[:width] + [-1] * (width - len(d)) for d in u_qid_matrix]
    u_qid_matrix = np.asarray(u_qid_matrix)
    ucontext_emb_matrix = context_emb_matrix[u_qid_matrix] # ucount, max_qcount, embed_size
    mask = np.array(u_qid_matrix == -1, dtype=float)
    qcount = mask.sum(axis=1)
    qcount = np.ma.array(qcount, mask=(qcount == 0), fill_value=1).filled()
    qcount = np.expand_dims(qcount, -1)
    ucontext_mean_emb = ucontext_emb_matrix.sum(axis=1) / qcount
    ucontext_mean_emb = convert_to_int(ucontext_mean_emb)
    return ucontext_mean_emb, uid_dic

def convert_to_int(context_emb_matrix, scale=1000000):
    dim_max_val = context_emb_matrix.max(axis=0)
    dim_min_val = context_emb_matrix.min(axis=0)
    dim_val_span = dim_max_val - dim_min_val
    context_emb_matrix = (context_emb_matrix - dim_min_val) / dim_val_span
    # return np.array(context_emb_matrix * scale, dtype=int)
    return context_emb_matrix * scale

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_file', '-f', \
        default="/home/keping2/data/input/rand_small/extract_sample0.10_feat_file.txt.gz")
        # default="/home/keping2/data/input/extract_sample0.10_hist_len11_feat_file.txt.gz")
    #parser.add_argument('--data_path', '-d', default="/home/keping2/data/input/by_time")
    parser.add_argument('--data_path', '-d', \
        # default="/home/keping2/data/input/rand_small/all_sample/by_time")
        default="/home/keping2/EmailSearch/model")
    parser.add_argument('--rank_dir', '-r', \
        default="/home/keping2/EmailSearch/model")
        #default="/home/keping2/data/working/pos_doc_context/by_users_usepopFalse_convTrue_docTrue")
    parser.add_argument('--option', default="add_query_context", \
        choices=["extract_feat", "add_doc_score", "add_query_context", "add_user_context"], help='')
    parser.add_argument('--out_format', default="lightgbm", \
        choices=["lightgbm", "aether"], help='')
    parser.add_argument('--do_cluster', default="hard", \
        choices=["hard", "soft", "none", "combine"], help='')
    parser.add_argument('--cluster_method', default="MiniBatchKMeans", \
        choices=["MiniBatchKMeans", "KMeans", "GaussianMixture", "AgglCluster", "DBSCAN", "FastCluster", "none"], help='')
    parser.add_argument('--n_clusters', default=10, type=int, help='')
    parser.add_argument('--sample_size', default=10000, type=int, help='')
    parser.add_argument("--show_tsne", type=str2bool, nargs='?', const=True, default=False,
                        help="Analyze the clustering figure.")

    paras = parser.parse_args()
    print(paras)
    os.system("mkdir -p %s" % paras.rank_dir)
    if paras.option == "extract_feat":
        train_qset, valid_qset, test_qset = set(), set(), set()
        train_qids = os.path.join(paras.data_path, "train_qids.txt.gz")
        valid_qids = os.path.join(paras.data_path, "valid_qids.txt.gz")
        test_qids = os.path.join(paras.data_path, "test_qids.txt.gz")
        read_qid_from_file(train_qids, train_qset)
        read_qid_from_file(valid_qids, valid_qset)
        read_qid_from_file(test_qids, test_qset)
        if paras.out_format == "aether":
            train_feat_file = os.path.join(paras.rank_dir, "train_feat.txt.gz")
            valid_feat_file = os.path.join(paras.rank_dir, "valid_feat.txt.gz")
            # uncompress train and valid yourself
            test_feat_file = os.path.join(paras.rank_dir, "test_feat.txt.gz")
            output_train_feat_file(paras.feat_file, train_qset, train_feat_file)
            output_train_feat_file(paras.feat_file, valid_qset, valid_feat_file)
            output_test_feat_file(paras.feat_file, test_qset, test_feat_file)
        else:
            train_feat_file = os.path.join(paras.rank_dir, "train_feat.tsv")
            valid_feat_file = os.path.join(paras.rank_dir, "valid_feat.tsv")
            test_feat_file = os.path.join(paras.rank_dir, "test_feat.tsv")
            output_lightgbm_file(paras.feat_file, train_qset, train_feat_file)
            output_lightgbm_file(paras.feat_file, valid_qset, valid_feat_file)
            output_lightgbm_file(paras.feat_file, test_qset, test_feat_file)

    elif "add" in paras.option:
        test_rank_file = os.path.join(paras.data_path, "test.context.best_model.ranklist.gz")
        valid_rank_file = os.path.join(paras.data_path, "valid.context.best_model.ranklist.gz")
        train_rank_file = os.path.join(paras.data_path, "train.context.best_model.ranklist.gz")
        fname = os.path.basename(paras.feat_file)
        if paras.option == "add_doc_score":
            qdoc_dic = defaultdict(dict)
            read_doc_scores(train_rank_file, qdoc_dic)
            read_doc_scores(valid_rank_file, qdoc_dic)
            read_doc_scores(test_rank_file, qdoc_dic)
            max_score = max([max(qdoc_dic[qid].values()) for qid in qdoc_dic])
            min_score = min([min(qdoc_dic[qid].values()) for qid in qdoc_dic])
            print(min_score, max_score)
            outfname = os.path.splitext(os.path.splitext(fname)[0])[0] \
                + "_docscore.txt.gz"
            # out_file = os.path.join(os.path.dirname(paras.feat_file), outfname)
            out_file = os.path.join(paras.rank_dir, outfname)
            add_column_to_file(paras.feat_file, out_file, qdoc_dic, min_score, max_score)
        elif paras.option == "add_query_context":
            qcontext_dic = dict()
            uq_dic = defaultdict(list)
            qid_list = []
            context_emb_matrix = []
            read_context_emb(train_rank_file, context_emb_matrix, qid_list, qcontext_dic, uq_dic)
            read_context_emb(valid_rank_file, context_emb_matrix, qid_list, qcontext_dic, uq_dic)
            read_context_emb(test_rank_file, context_emb_matrix, qid_list, qcontext_dic, uq_dic)
            context_emb_matrix = np.asarray(context_emb_matrix, dtype=np.float32)
            qid_array = np.asarray(qid_list)
            print(context_emb_matrix.shape) # sample_size, hidden_emb_size(128)
            outfname = os.path.splitext(os.path.splitext(fname)[0])[0]
            if paras.do_cluster == "none":
                qcontext_int_matrix = convert_to_int(context_emb_matrix)
                outfname += "_qcontext_emb.txt.gz"
            else:
                if paras.show_tsne:
                    randperm = np.random.permutation(context_emb_matrix.shape[0])
                    context_emb_matrix = context_emb_matrix[randperm[:paras.sample_size]]
                    qid_array = qid_array[randperm[:paras.sample_size]]
                    qdic = set(qid_array)
                    cur_qid_meta_dic = read_meta_info(paras.feat_file, qdic)
                    column_names = ["item_type", "user_type", "q_lang_hash", "culture_lcid", "locale_lcid", "action"]
                    qmeta_info = [cur_qid_meta_dic[q] for q in qid_array]
                    df = pd.DataFrame(np.asarray(qmeta_info), columns=column_names)
                    print(len(df))

                qcontext_int_matrix = cluster_context_emb(
                    context_emb_matrix, paras.n_clusters, paras.cluster_method, paras.do_cluster)
                print(len(qcontext_int_matrix))
                data_version = "by_users" if "by_users" in paras.data_path else "by_time"
                fname = data_version + '_' + paras.cluster_method
                n_clusters = paras.n_clusters
                if paras.cluster_method == "DBSCAN":
                    n_clusters = len(np.unique(qcontext_int_matrix))
                if paras.show_tsne:
                    df["cluster_id"] = qcontext_int_matrix
                    fit_tsne(context_emb_matrix, df, figname="%s_spl%d" % (fname, paras.sample_size))
                    if paras.sample_size < len(qcontext_dic):
                        # do not perform adding context to feature file
                        return
                outfname += "_qcluster_{}.txt.gz".format(paras.do_cluster)
            out_file = os.path.join(paras.rank_dir, outfname)
            if paras.do_cluster == "combine":
                add_context_combination_to_file(paras.feat_file, out_file, qcontext_int_matrix, qcontext_dic)
            else:
                if paras.do_cluster == "hard":
                    sparse_y = np.zeros((qcontext_int_matrix.shape[0], n_clusters), dtype=int)
                    sparse_y[np.expand_dims(np.arange(qcontext_int_matrix.shape[0]), -1), qcontext_int_matrix] = 1
                    qcontext_int_matrix = sparse_y
                add_context_to_file(paras.feat_file, out_file, \
                    qcontext_int_matrix, qcontext_dic, feat_type="QContext")
        elif paras.option == "add_user_context":
            # only data in train and validation can be used to compute user embedding
            # which do not update often during test time
            qcontext_dic = dict()
            uq_dic = defaultdict(list)
            qid_list = []
            context_emb_matrix = []
            read_context_emb(train_rank_file, context_emb_matrix, qid_list, qcontext_dic, uq_dic)
            read_context_emb(valid_rank_file, context_emb_matrix, qid_list, qcontext_dic, uq_dic)
            context_emb_matrix = np.asarray(context_emb_matrix, dtype=np.float32)
            ucontext_mean_emb, uid_dic = collect_ucontext_emb(
                context_emb_matrix, qcontext_dic, uq_dic)
            outfname = os.path.splitext(os.path.splitext(fname)[0])[0]
            if paras.do_cluster == "none":
                ucontext_mean_emb = convert_to_int(ucontext_mean_emb)
                outfname += "_ucontext_emb.txt.gz"
            else:
                ucontext_mean_emb = cluster_context_emb(
                    ucontext_mean_emb, paras.n_clusters, paras.cluster_method, paras.do_cluster)
                outfname += "_ucluster_{}.txt.gz".format(paras.do_cluster)

            out_file = os.path.join(paras.rank_dir, outfname)
            add_context_to_file(paras.feat_file, out_file, \
                ucontext_mean_emb, uid_dic, feat_type="UContext")

if __name__ == "__main__":
    main()
