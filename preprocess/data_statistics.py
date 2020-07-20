""" Collect the data statistics from the collection
"""
import torch
import numpy as np

from collections import defaultdict
import copy
import gzip
import os
import argparse
import seaborn as sns
sns.set(color_codes=True)
sns.set(style="whitegrid")
import pandas as pd
import matplotlib.pyplot as plt

#Read files from gzip feature file
#Q, associated d (rating), uid,
#Sort m:QueryId(requestID) according to SearchTime

GRADE_DIC = {"Bad": 0, "Fair":1, "Good": 2, "Excellent": 3, "Perfect": 4}
GRADE_NAME_DIC = {GRADE_DIC[x]:x for x in GRADE_DIC}
CANDI_DOC_COUNT = 100
def read_feature_file(fname, q_info_dic):
    try:
        line_no = 0
        with gzip.open(fname, 'rt') as fin:
            line = fin.readline().strip()
            feat_col_name = line.split('\t')
            feat_name_dic = {feat_col_name[i]: i for i in range(len(feat_col_name))}
            for line in fin:
                line_no += 1
                if line_no % 100000 == 0:
                    print("%d lines has been parsed!" % line_no)
                segs = line.strip().split('\t')
                qid = int(segs[feat_name_dic['m:QueryId']])
                uid = int(segs[feat_name_dic['m:MailboxId']])
                rating = GRADE_DIC[segs[feat_name_dic['m:Rating']]]
                search_time = int(segs[feat_name_dic['AdvancedPreferFeature_123']])
                q_info_dic[qid].append((rating, search_time, uid))
    except Exception as e:
        print("Error Message: {}".format(e))

    return feat_col_name, feat_name_dic # , q_info_dic

def sort_query_per_user(q_info_dic):
    #sort queryID(requestID)
    #sort qid according to search time; assign qid an index to indicate its position for the uid
    u_queries_dic = defaultdict(list)
    for qid in q_info_dic:
        q_info_dic[qid] = q_info_dic[qid][:CANDI_DOC_COUNT]
        _, search_time, uid = q_info_dic[qid][0]
        u_queries_dic[uid].append((qid, search_time))

    # for uid in u_queries_dic:
    #    u_queries_dic[uid].sort(key=lambda x: x[1])

    return u_queries_dic

def stats(u_queries_dic, q_info_dic, figname="all_q_dist"):
    print("#Users:{}".format(len(u_queries_dic)))
    q_count_list = []
    rated_doc_count_list = [[] for _ in range(len(GRADE_DIC))]
    for u in u_queries_dic:
        q_count = len(u_queries_dic[u])
        q_count_list.append(q_count)
        doc_count = [0] * len(GRADE_DIC)
        for q, _ in u_queries_dic[u]:
            for rating, _, _ in q_info_dic[q]:
                doc_count[rating] += 1
        for rating in range(len(doc_count)):
            rated_doc_count_list[rating].append(doc_count[rating])
    q_count_list = np.asarray(q_count_list)
    rated_doc_count_list = np.asarray(rated_doc_count_list)
    print("#QueryPerUser:{:.2f}+/-{:.2f}".format(np.mean(q_count_list), np.std(q_count_list)))
    u_q_distribution(q_count_list, figname)
    for rating in range(rated_doc_count_list.shape[0]):
        print("#Rating:{} Per User: {:.2f}+/-{:.2f}".format(
            GRADE_NAME_DIC[rating],
            np.mean(rated_doc_count_list[rating]), np.std(rated_doc_count_list[rating])))
    presented_doc_list = rated_doc_count_list.sum(axis=0)
    print("#Doc Per User: {:.2f}+/-{:.2f}".format(
        np.mean(presented_doc_list), np.std(presented_doc_list)))

    print("#Queries:{}".format(len(q_info_dic)))
    rated_doc_count_list = [[] for _ in range(len(GRADE_DIC))]
    for q in q_info_dic:
        doc_count = [0 for _ in range(len(GRADE_DIC))]
        for rating, _, _ in q_info_dic[q]:
            doc_count[rating] += 1
        for rating in range(len(doc_count)):
            rated_doc_count_list[rating].append(doc_count[rating])
    rated_doc_count_list = np.asarray(rated_doc_count_list)
    for rating in range(rated_doc_count_list.shape[0]):
        print("#Rating:{} Per Query: {:.2f}+/-{:.2f}".format(
            GRADE_NAME_DIC[rating],
            np.mean(rated_doc_count_list[rating]), np.std(rated_doc_count_list[rating])))
    presented_doc_list = rated_doc_count_list.sum(axis=0)
    print("#Doc Per Query: {:.2f}+/-{:.2f}".format(
        np.mean(presented_doc_list), np.std(presented_doc_list)))


def u_q_distribution(q_count_list, figname, bin_count=10):
    #number of queries for every user
    q_bin = [0] * bin_count

    for q_count in q_count_list:
        q_idx = q_count -1 if q_count < bin_count else bin_count-1
        q_bin[q_idx] += 1
    plot_according_to_bin(q_bin, figname)

def plot_according_to_bin(q_bin, figname):
    total_u_count = sum(q_bin) + 0.0
    print(q_bin)
    for idx in range(len(q_bin)):
        q_bin[idx] /= total_u_count
    print(q_bin)
    x_axis = list(range(1, len(q_bin)+1))
    pdata = {'#Queries': x_axis, "Percentage": q_bin}
    pdata = pd.DataFrame(data=pdata)
    #sns.distplot(q_bin)
    ax = sns.barplot(data=pdata, x="#Queries", y="Percentage")
    plt.savefig("%s.pdf" % figname)

def read_qid_file(fname, u_q_dic):
    with gzip.open(fname, 'rt') as fin:
        for line in fin:
            line = line.strip()
            qid, uid, _ = line.split() # qid, uid, search_time
            u_q_dic[uid].add(qid)

def count_q_distr(data_path, partition_name):
    partition_qid_file = "%s/%s_qids.txt.gz" % (data_path, partition_name)
    u_q_dic = defaultdict(set)
    read_qid_file(partition_qid_file, u_q_dic)
    if "by_time" in data_path:
        vt_uq_dic = copy.deepcopy(u_q_dic)
        train_qid_file = "%s/train_qids.txt.gz" % (data_path)
        read_qid_file(train_qid_file, u_q_dic)
    else:
        vt_uq_dic = u_q_dic
    q_count_list = []
    for uid in vt_uq_dic:
        q_count_list.append(len(u_q_dic[uid]))
    figname = "%s/%s_q_dist" % (data_path, partition_name)
    u_q_distribution(q_count_list, figname)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_file1', '-f1', default="/home/keping2/data/input/1_6_1_13_data.gz")
    parser.add_argument('--feat_file2', '-f2', default="")
    parser.add_argument('--data_path', '-data', default="")
    # /home/keping2/data/input/1_27_2_2_data.gz
    paras = parser.parse_args()
    if os.path.exists(paras.data_path):
        count_q_distr(paras.data_path, "valid")
        count_q_distr(paras.data_path, "test")
    else:
        q_info_dic = defaultdict(list)
        read_feature_file(paras.feat_file1, q_info_dic)
        if os.path.exists(paras.feat_file2):
            read_feature_file(paras.feat_file2, q_info_dic)
        u_queries_dic = sort_query_per_user(q_info_dic)
        stats(u_queries_dic, q_info_dic)

if __name__ == "__main__":
    main()
    #q_bin = [560059, 250256, 140454, 89917, 61927, 45456, 34936, 27095, 21872, 175903]
    #plot_according_to_bin(q_bin)
