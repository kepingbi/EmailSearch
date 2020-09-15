import os
import sys
import gzip
import numpy as np
from collections import defaultdict

data_path = "/home/keping2/data/working/pos_doc_context/by_time_embsize128_ff512_h8_layer2_lr0.002_ws2000_epoch10_lnorm1e-05_prevq10_posTrue_qFalse_dTrue_qdFalse_curqFalse_rndprevFalse_unbiasTrue/by_time_add_query_context_hard_MiniBatchKMeans_n_clusters10"
#qid, uid, clusterid
def load_file(fname, uqcluster_dic):
    #uid: (qid, clusterid)
    with gzip.open(fname, "rt") as fin:
        for line in fin:
            segs = line.strip("\r\n").split("\t")
            qid = int(segs[0])
            uid_str = segs[1]
            cluster_id = int(segs[2])
            if cluster_id not in uqcluster_dic[uid_str]:
                uqcluster_dic[uid_str][cluster_id] = 0
            uqcluster_dic[uid_str][cluster_id] += 1

def calc_entropy(distr):
    distr = distr / distr.sum()
    entropy = sum(-np.log(distr) * distr)
    return entropy

def plot_distr(val_arr, up_bound, low_bound, n_sections=10):
    sec_len = (up_bound - low_bound) / n_sections
    distr = np.zeros(n_sections)
    for val in val_arr:
        index = min((val - low_bound) // sec_len, n_sections - 1)
        index = max(index, 0)
        distr[int(index)] += 1
    return distr.tolist()

def compute_entropy(n_clusters=10):
    up_bound = calc_entropy(np.asarray([1/n_clusters] * n_clusters))
    # low_bound = calc_entropy(np.asarray([1.] + [0.] * (n_clusters-1)))
    low_bound = calc_entropy(np.asarray([1.]))

    u_entropy_list = []
    uqcluster_dic = defaultdict(dict)
    for part in ["train", "valid", "test"]:
        fname = "%s/%s.clusters.txt.gz" % (data_path, part)
        load_file(fname, uqcluster_dic)
    print(len(uqcluster_dic))
    print(sum([len(uqcluster_dic[user]) for user in uqcluster_dic]))
    cluster_count_arr = np.zeros(n_clusters)
    for user in uqcluster_dic:
        distr = []
        for cluster_id in uqcluster_dic[user]:
            distr.append(uqcluster_dic[user][cluster_id])
        cluster_count_arr[len(uqcluster_dic[user]) - 1] += 1
        distr = np.asarray(distr)
        cur_entropy = calc_entropy(distr)
        u_entropy_list.append(cur_entropy)
    u_entropy_list = np.asarray(u_entropy_list)
    print("Lower Bound: %.6f Upper Bound: %.6f" % (low_bound, up_bound))
    print("Average Entropy is %.6f" % u_entropy_list.mean())

    distr = plot_distr(u_entropy_list, up_bound, low_bound)
    print(distr)
    print(cluster_count_arr)

if __name__ == "__main__":
    compute_entropy()