''' Dataset structure in order to batchify data
'''
import torch
from torch.utils.data import Dataset
import numpy as np
import random
import others.util as util
import gzip

from collections import defaultdict


class DocContextDataset(Dataset):
    """ load training, validation and test data
    u,Q,i for a purchase i given u, Q
            negative samples i- for u, Q
    read reviews of u, Q before i (previous m reviews)
    reviews of others before r_i (review of i from u)
    load test data
    for each review, collect random words in reviews or words in a sliding window in reviews.
                    or all the words in the review
    """

    def __init__(self, args, personal_data, partition_name, for_test=False):
        self.args = args
        self.candi_doc_count = args.candi_doc_count
        self.doc_pad_idx = personal_data.doc_pad_idx
        self.for_test = for_test
        random.seed(args.seed)
        # self.rating_pad_idx = 6
        # {"NotAppear": 0, "Bad": 1, "Fair":2, "Good": 3, "Excellent": 4, "Perfect": 5}
        self.prev_q_limit = args.prev_q_limit
        self.personal_data = personal_data
        self.query_list = self.collect_qid_samples(self.personal_data, partition_name)
        self._data = self.query_list
        self.train_ranker = False # for model match_patterns
        print(partition_name, len(self._data))

    def collect_qid_samples(self, personal_data, partition_name):
        # if partition_name == "train":
        #     partition_name = "filter_hl%d_train" % self.args.hist_len
        partition_qid_file = "%s/%s_qids.txt.gz" % (self.args.data_dir, partition_name)
        qid_list = []
        with gzip.open(partition_qid_file, 'rt') as fin:
            line = fin.readline()
            for line in fin:
                line = line.strip()
                qid, _, _ = line.split() # qid, uid, search_time
                qid_list.append(int(qid))
        # load qids for train, validation, and test from separate files.
        if partition_name == "train" and self.args.filter_train and not self.for_test:
            qid_list = self.filter_train_qids(qid_list, 2) # self.args.prev_q_limit//2
        return qid_list

    def initialize_epoch(self, train_ranker=False):
        """ Return negk qid for each qid that will be trained as positive samples
        """
        neg_qids = np.random.choice(self.query_list,
                                 size=(len(self.query_list), self.args.neg_k), replace=True).tolist()
        paired_qids = list(zip(self.query_list, neg_qids))
        self._data = paired_qids
        self.train_ranker = train_ranker

    def set_get_ranker_batch(self, train_ranker=False):
        self.train_ranker = train_ranker

    def filter_train_qids(self, qid_list, prev_qcount_thre=5):
        filtered_qids = []
        # rnd_ratio = self.args.rnd_ratio #
        # prev_qcount_thre = self.args.hist_len - 1
        # other_qids = []
        for qid in qid_list:
            _, prev_qid_count = self.personal_data.query_info_dic[qid][-1]
            #position of qid in the sequence of queries the user issued.
            if prev_qid_count >= prev_qcount_thre:
                filtered_qids.append(qid)
            # else:
            #     other_qids.append(qid)
        # sample_count = int(len(other_qids) * rnd_ratio)
        # sample_qids = random.sample(other_qids, sample_count)
        # filtered_qids += sample_qids
        return filtered_qids

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]
