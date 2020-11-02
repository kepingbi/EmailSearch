''' Collect detailed information in each batch in order to input to the model
'''
from torch.utils.data import DataLoader
import others.util as util
from data.batch_data import DocContextBatch, DocBaselineBatch
from data.data_util import PersonalSearchData
import random
from datetime import datetime

class DocContextDataloader(DataLoader):
    ''' Data loaded as batches and converted to gpu tensor
        which can be fed into the model.
    '''
    def __init__(self, args, dataset, batch_size=1,
                 shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, pin_memory=False,
                 drop_last=False, timeout=0, worker_init_fn=None):
        super(DocContextDataloader, self).__init__(
            dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
            batch_sampler=batch_sampler, num_workers=num_workers,
            pin_memory=pin_memory, drop_last=drop_last, timeout=timeout,
            worker_init_fn=worker_init_fn, collate_fn=self._collate_fn)
        self.args = args
        random.seed(args.seed)
        # {"NotAppear":0, "Bad":1, "Fair":2, "Good":3, "Excellent":4, "Perfect":5}
        self.personal_data = self.dataset.personal_data
        self.doc_pad_idx = self.dataset.personal_data.doc_pad_idx
        self.days_pad_idx = self.dataset.personal_data.days_pad_idx
        self.is_train = shuffle # when the data is shuffled, it is train batch

    def _collate_fn(self, batch):
        if self.args.model_name == 'baseline' or self.args.model_name == "mp_context":
            # validation or test
            return self.get_baseline_batch(batch)
        if self.args.model_name == 'pos_doc_context':
            return self.get_context_batch(batch)
        if self.args.model_name == "match_patterns":
            if not self.is_train and self.dataset.train_ranker:
                batch_qids = [qid for qid, _ in batch]
                return self.get_context_batch(batch_qids)
            return self.get_context_pos_qd_batch(batch)

    def get_baseline_batch(self, batch_qids):
        ''' get batch for baseline (without context information)
        '''
        batch_qids, batch_user_idxs, batch_qcontext_emb, batch_candi_doc_idxs, batch_candi_doc_ratings, \
            batch_candi_doc_rel_pos, batch_candi_doc_time_pos, \
            batch_candi_doc_qcont_features, batch_candi_doc_qdiscrete_features, \
                batch_candi_doc_dcont_features, batch_candi_doc_ddiscrete_features, \
                    batch_candi_doc_qdcont_features, batch_candi_doc_conv_hashes\
                        = self.collect_candi_doc_data(batch_qids)

        batch = DocBaselineBatch(batch_qids, batch_user_idxs,
                                 batch_candi_doc_idxs, batch_candi_doc_ratings,
                                 batch_candi_doc_rel_pos, batch_candi_doc_time_pos,
                                 batch_candi_doc_qcont_features, batch_candi_doc_qdiscrete_features,
                                 batch_candi_doc_dcont_features, batch_candi_doc_ddiscrete_features,
                                 batch_candi_doc_qdcont_features, batch_qcontext_emb=batch_qcontext_emb)

        return batch

    def get_context_pos_qd_batch(self, batch_qids):
        batch_qids, batch_user_idxs, batch_pos_doc_qids, \
            batch_candi_doc_qcont_features, batch_candi_doc_qdiscrete_features, \
                batch_candi_doc_dcont_features, batch_candi_doc_ddiscrete_features, \
                    batch_candi_doc_qdcont_features \
                        = self.collect_pos_neg_qd_pairs(batch_qids)

        batch_context_qidxs, batch_context_search_time, batch_context_pos_didxs, \
            batch_context_qcont_features, batch_context_qdiscrete_features, \
                batch_context_pos_dcont_features, batch_context_pos_ddiscrete_features, \
                    batch_context_pos_qdcont_features, batch_context_d_popularity, \
                        batch_candi_doc_popularity, \
                            batch_candi_conv_occur, batch_candi_doc_occur = \
                        self.collect_context_data(batch_qids, batch_pos_doc_qids, \
                            batch_pos_doc_qids)

        batch = DocContextBatch(batch_qids, batch_user_idxs, batch_pos_doc_qids, [], [], [],
                                batch_candi_doc_qcont_features, batch_candi_doc_qdiscrete_features,
                                batch_candi_doc_dcont_features, batch_candi_doc_ddiscrete_features,
                                batch_candi_doc_qdcont_features,
                                batch_candi_doc_popularity,
                                batch_candi_conv_occur, batch_candi_doc_occur,
                                batch_context_qidxs, batch_context_search_time,
                                batch_context_pos_didxs,
                                batch_context_qcont_features, batch_context_qdiscrete_features,
                                batch_context_pos_dcont_features,
                                batch_context_pos_ddiscrete_features,
                                batch_context_pos_qdcont_features,
                                batch_context_d_popularity)
        if self.dataset.train_ranker and self.is_train:
            ranker_batch = self.get_baseline_batch(batch_qids)
            return [batch, ranker_batch]

        return batch

    def get_context_batch(self, batch_qids):
        # qid, qid and associated positive and negative documents in the history
        # features of current document candidates and the historical context documents.
        # labels of the current document candidates.
        # (d11,d12), (d21,22), (d1,dk2), (dc1, dc2, ..., dcn)
        # d: r1,r2,...,rk as additional features.
        # original features processed
        # features combined to represent d

        batch_qids, batch_user_idxs, batch_qcontext_emb, batch_candi_doc_idxs, batch_candi_doc_ratings, \
            batch_candi_doc_rel_pos, batch_candi_doc_time_pos, \
            batch_candi_doc_qcont_features, batch_candi_doc_qdiscrete_features, \
                batch_candi_doc_dcont_features, batch_candi_doc_ddiscrete_features, \
                    batch_candi_doc_qdcont_features, batch_candi_doc_conv_hashes \
                        = self.collect_candi_doc_data(batch_qids)
        batch_context_qidxs, batch_context_search_time, batch_context_pos_didxs, \
            batch_context_qcont_features, batch_context_qdiscrete_features, \
                batch_context_pos_dcont_features, batch_context_pos_ddiscrete_features, \
                    batch_context_pos_qdcont_features, batch_context_d_popularity, \
                        batch_candi_doc_popularity, \
                            batch_candi_conv_occur, batch_candi_doc_occur = \
                        self.collect_context_data(batch_qids, batch_candi_doc_idxs, \
                            batch_candi_doc_conv_hashes)

        batch = DocContextBatch(batch_qids, batch_user_idxs,
                                batch_candi_doc_idxs, batch_candi_doc_ratings,
                                batch_candi_doc_rel_pos, batch_candi_doc_time_pos,
                                batch_candi_doc_qcont_features, batch_candi_doc_qdiscrete_features,
                                batch_candi_doc_dcont_features, batch_candi_doc_ddiscrete_features,
                                batch_candi_doc_qdcont_features,
                                batch_candi_doc_popularity,
                                batch_candi_conv_occur, batch_candi_doc_occur,
                                batch_context_qidxs, batch_context_search_time,
                                batch_context_pos_didxs,
                                batch_context_qcont_features, batch_context_qdiscrete_features,
                                batch_context_pos_dcont_features,
                                batch_context_pos_ddiscrete_features,
                                batch_context_pos_qdcont_features,
                                batch_context_d_popularity)
        # print(batch.context_search_time)
        mask_out_count = max(self.args.prev_q_limit - self.args.test_prev_q_limit, 0)
        if mask_out_count > 0:
            batch.context_qidxs[:, :mask_out_count] = 0
            batch.context_search_time[:, :mask_out_count] = self.days_pad_idx
            batch.context_pos_didxs[:, :mask_out_count] = self.doc_pad_idx
            batch.context_qcont_features[:, :mask_out_count] = 0
            batch.context_qdiscrete_features[:, :mask_out_count] = 0
            batch.context_pos_dcont_features[:, :mask_out_count] = 0
            batch.context_pos_ddiscrete_features[:, :mask_out_count] = 0
            batch.context_pos_qdcont_features[:, :mask_out_count] = 0
        return batch

    def collect_pos_neg_qd_pairs(self, batch_qids):
        # batch_qids should be (qid, [neg_qidxs])
        new_batch_qids = []
        batch_user_idxs = []
        batch_pos_doc_qids = []
        # candidate doc_ids, padded to args.candi_doc_count
        batch_candi_doc_qcont_features, batch_candi_doc_qdiscrete_features = [], []
        batch_candi_doc_dcont_features, batch_candi_doc_ddiscrete_features = [], []
        batch_candi_doc_qdcont_features = []
        for qid, neg_qidxs in batch_qids: #request id
            cur_pos_didxs = self.personal_data.q_pos_didx_dic[qid] # idx in segs
            candi_doc_rel_pos = [int(segs[self.personal_data.feature_name_dic['m:RelevancePosition']]) \
                for segs in self.personal_data.query_info_dic[qid][:-1]]
            if self.dataset.train_ranker and self.is_train and self.args.unbiased_train and candi_doc_rel_pos[0] != 1:
                # if there is no document that appeared in either window under the query
                # skip the query
                # to match qids of the ranker. 
                continue
            if len(cur_pos_didxs) == 0:
                continue
            for pos_didx in cur_pos_didxs:
                candi_doc_dcont_features, candi_doc_ddiscrete_features = [], []
                candi_doc_qdcont_features = []
                candi_doc_qcont_features, candi_doc_qdiscrete_features = [], []
                new_batch_qids.append(qid)
                cur_qidxs = [qid]
                segs = self.personal_data.query_info_dic[qid][pos_didx]
                query_cont_features, query_discrete_features, \
                    doc_cont_features, doc_discrete_features, \
                    qd_cont_features = self.personal_data.collect_group_features(
                        segs, self.personal_data.feature_name_dic)
                candi_doc_dcont_features.append(doc_cont_features)
                candi_doc_ddiscrete_features.append(doc_discrete_features[:-1])
                # conversation hash ([-1])
                candi_doc_qdcont_features.append(qd_cont_features)
                candi_doc_qcont_features.append(query_cont_features)
                candi_doc_qdiscrete_features.append(query_discrete_features)
                for neg_qid in neg_qidxs:
                    neg_pos_didxs = self.personal_data.q_pos_didx_dic[neg_qid] # idx in segs
                    if len(neg_pos_didxs) == 0:
                        continue
                    cur_qidxs.append(neg_qid)
                    neg_pos_did = random.choice(neg_pos_didxs)
                    segs = self.personal_data.query_info_dic[neg_qid][neg_pos_did]
                    query_cont_features, query_discrete_features, \
                        doc_cont_features, doc_discrete_features, \
                        qd_cont_features = self.personal_data.collect_group_features(
                            segs, self.personal_data.feature_name_dic)
                    candi_doc_dcont_features.append(doc_cont_features)
                    candi_doc_ddiscrete_features.append(doc_discrete_features[:-1])
                    # conversation hash ([-1])
                    candi_doc_qdcont_features.append(qd_cont_features)
                    candi_doc_qcont_features.append(query_cont_features)
                    candi_doc_qdiscrete_features.append(query_discrete_features)
                batch_pos_doc_qids.append(cur_qidxs)
                batch_candi_doc_qcont_features.append(candi_doc_qcont_features)
                # batch_size, cont_feature_count
                batch_candi_doc_qdiscrete_features.append(candi_doc_qdiscrete_features)
                batch_candi_doc_dcont_features.append(candi_doc_dcont_features)
                batch_candi_doc_ddiscrete_features.append(candi_doc_ddiscrete_features)
                batch_candi_doc_qdcont_features.append(candi_doc_qdcont_features)

        ##padding###
        batch_user_idxs = [self.personal_data.query_info_dic[x][-1][0] for x in new_batch_qids]
        #continous features pad float 0.0. discrete features pad 0.
        batch_pos_doc_qids = util.pad(batch_pos_doc_qids, pad_id=-1, width=1+self.args.neg_k)
        batch_candi_doc_qcont_features = util.pad_3d(
            batch_candi_doc_qcont_features, pad_id=0., dim=1, width=1+self.args.neg_k)
        batch_candi_doc_qdiscrete_features = util.pad_3d(
            batch_candi_doc_qdiscrete_features, pad_id=0, dim=1, width=1+self.args.neg_k)
        batch_candi_doc_dcont_features = util.pad_3d(
            batch_candi_doc_dcont_features, pad_id=0., dim=1, width=1+self.args.neg_k)
        batch_candi_doc_ddiscrete_features = util.pad_3d(
            batch_candi_doc_ddiscrete_features, pad_id=0, dim=1, width=1+self.args.neg_k)
        batch_candi_doc_qdcont_features = util.pad_3d(
            batch_candi_doc_qdcont_features, pad_id=0., dim=1, width=1+self.args.neg_k)

        return new_batch_qids, batch_user_idxs, batch_pos_doc_qids, \
            batch_candi_doc_qcont_features, batch_candi_doc_qdiscrete_features, \
                batch_candi_doc_dcont_features, batch_candi_doc_ddiscrete_features, \
                    batch_candi_doc_qdcont_features

    def collect_candi_doc_data(self, batch_qids):
        # (dc1, dc2, ..., dcn)
        batch_candi_doc_idxs, batch_candi_doc_ratings = [], []
        new_batch_qids = []
        # candidate doc_ids, padded to args.candi_doc_count
        batch_candi_doc_qcont_features, batch_candi_doc_qdiscrete_features = [], []
        batch_candi_doc_dcont_features, batch_candi_doc_ddiscrete_features = [], []
        batch_candi_doc_qdcont_features = []
        batch_candi_doc_conv_hashes = []
        batch_candi_doc_rel_pos, batch_candi_doc_time_pos = [], []
        for qid in batch_qids: #request id
            # batch_user_idxs = [int(self.personal_data.query_info_dic[qid][0][\
            #     self.personal_data.feature_name_dic['m:MailboxId']])]
            candi_doc_idxs = [int(segs[self.personal_data.feature_name_dic['m:DocId']]) \
                for segs in self.personal_data.query_info_dic[qid][:-1]]
            candi_doc_rel_pos = [int(segs[self.personal_data.feature_name_dic['m:RelevancePosition']]) \
                for segs in self.personal_data.query_info_dic[qid][:-1]]
            candi_doc_time_pos = [int(segs[self.personal_data.feature_name_dic['m:DateTimePosition']]) \
                for segs in self.personal_data.query_info_dic[qid][:-1]]
            if self.is_train and self.args.unbiased_train and candi_doc_rel_pos[0] != 1:
                # if there is no document that appeared in either window under the query
                # skip the query
                continue
            new_batch_qids.append(qid)
            batch_candi_doc_idxs.append(candi_doc_idxs)
            batch_candi_doc_rel_pos.append(candi_doc_rel_pos)
            batch_candi_doc_time_pos.append(candi_doc_time_pos)

            candi_doc_ratings = [PersonalSearchData.RATING_MAP[
                segs[self.personal_data.feature_name_dic['m:Rating']]] \
                    for segs in self.personal_data.query_info_dic[qid][:-1]]
            batch_candi_doc_ratings.append(candi_doc_ratings)
            candi_doc_dcont_features = []
            candi_doc_ddiscrete_features = []
            candi_doc_conv_hashes = []
            candi_doc_qdcont_features = []
            query_cont_features, query_discrete_features = None, None
            for segs in self.personal_data.query_info_dic[qid][:-1]:
                query_cont_features, query_discrete_features, \
                    doc_cont_features, doc_discrete_features, \
                    qd_cont_features = self.personal_data.collect_group_features(
                        segs, self.personal_data.feature_name_dic)
                candi_doc_dcont_features.append(doc_cont_features)
                candi_doc_ddiscrete_features.append(doc_discrete_features[:-1])
                # conversation hash ([-1])
                candi_doc_conv_hashes.append(doc_discrete_features[-1])
                candi_doc_qdcont_features.append(qd_cont_features)
            batch_candi_doc_conv_hashes.append(candi_doc_conv_hashes) # not padded here
            batch_candi_doc_qcont_features.append(query_cont_features)
            # batch_size, cont_feature_count
            batch_candi_doc_qdiscrete_features.append(query_discrete_features)
            batch_candi_doc_dcont_features.append(candi_doc_dcont_features)
            batch_candi_doc_ddiscrete_features.append(candi_doc_ddiscrete_features)
            batch_candi_doc_qdcont_features.append(candi_doc_qdcont_features)

        ##padding###
        batch_user_idxs = [self.personal_data.query_info_dic[qid][-1][0] for qid in new_batch_qids]
        batch_qcontext_emb = None
        if hasattr(self.personal_data, "qcontext_dic"):
            batch_qcontext_emb = [self.personal_data.qcontext_dic[qid] for qid in new_batch_qids]

        #continous features pad float 0.0. discrete features pad 0.
        batch_candi_doc_idxs = util.pad(batch_candi_doc_idxs, pad_id=self.doc_pad_idx, \
            width=self.args.candi_doc_count)
        batch_candi_doc_rel_pos = util.pad(batch_candi_doc_rel_pos, pad_id=-1, \
            width=self.args.candi_doc_count)
        batch_candi_doc_time_pos = util.pad(batch_candi_doc_time_pos, pad_id=-1, \
            width=self.args.candi_doc_count)

        batch_candi_doc_ratings = util.pad(batch_candi_doc_ratings, pad_id=0, \
            width=self.args.candi_doc_count)
        #batch_candi_doc_qcont_features = util.pad(batch_candi_doc_qcont_features, pad_id=0.)
        #batch_candi_doc_qdiscrete_features = util.pad(batch_candi_doc_qdiscrete_features, pad_id=0)
        batch_candi_doc_dcont_features = util.pad_3d(
            batch_candi_doc_dcont_features, pad_id=0., dim=1, width=self.args.candi_doc_count)
        batch_candi_doc_ddiscrete_features = util.pad_3d(
            batch_candi_doc_ddiscrete_features, pad_id=0, dim=1, width=self.args.candi_doc_count)
        batch_candi_doc_qdcont_features = util.pad_3d(
            batch_candi_doc_qdcont_features, pad_id=0., dim=1, width=self.args.candi_doc_count)

        return new_batch_qids, batch_user_idxs, batch_qcontext_emb, batch_candi_doc_idxs, batch_candi_doc_ratings, \
            batch_candi_doc_rel_pos, batch_candi_doc_time_pos, \
            batch_candi_doc_qcont_features, batch_candi_doc_qdiscrete_features, \
                batch_candi_doc_dcont_features, batch_candi_doc_ddiscrete_features, \
                    batch_candi_doc_qdcont_features, batch_candi_doc_conv_hashes

    def collect_context_data(self, batch_qids, batch_candi_doc_idxs, batch_candi_doc_conv_hashes):
        #get features of context query/positive documents.
        batch_context_qidxs, batch_context_pos_didxs = [], []
        batch_context_search_time = []
        batch_context_qcont_features, batch_context_qdiscrete_features = [], []
        batch_context_pos_dcont_features, batch_context_pos_ddiscrete_features = [], []
        batch_context_pos_qdcont_features = []
        batch_context_d_popularity = [] # 0,1,2,3,4,5 (not occur, occur with label 0,1,2,3,4)
        batch_candi_doc_popularity = [] # 0,1,2,3,4,5 (not occur, occur with label 0,1,2,3,4)
        batch_candi_conv_occur = [] # conversation hash occur in prev postive doc
        batch_candi_doc_occur = [] # doc id occur in prev positive doc
        prev_q_limit = 1 if self.args.prev_q_limit == 0 else self.args.prev_q_limit

        for qid, candi_doc_idxs, candi_conv_hash in zip(batch_qids, \
            batch_candi_doc_idxs, batch_candi_doc_conv_hashes): #request id

            uid, idx = self.personal_data.query_info_dic[qid][-1]
            #qidx, search_time
            #position of qid in the sequence of queries the user issued.
            all_prev_qidxs = self.personal_data.u_queries_dic[uid][:idx]
            if self.args.rand_prev:
                all_prev_qidxs = self.personal_data.u_queries_dic[uid][:idx] \
                    + self.personal_data.u_queries_dic[uid][idx+1:]
                prev_qidxs = all_prev_qidxs[:prev_q_limit]
                # sample_count = min(len(all_prev_qidxs), prev_q_limit)
                # prev_qidxs = random.sample(all_prev_qidxs, sample_count)
                # prev_qidxs.sort(key=lambda x: x[1])
            else:
                prev_qidxs = all_prev_qidxs[-prev_q_limit:]
            cur_q_time = datetime.fromtimestamp(self.personal_data.u_queries_dic[uid][idx][1])
            prev_qsearch_time = [cur_q_time.date() - datetime.fromtimestamp(t).date() for _, t in prev_qidxs]
            prev_qsearch_time = [min(t.days, self.days_pad_idx) for t in prev_qsearch_time]
            # in case error occurs.
            prev_qidxs = [qidx for qidx, _ in prev_qidxs]
            # prev_qidxs = [qidx for qidx, _ in \
            #     self.personal_data.u_queries_dic[uid][:idx][-self.args.prev_q_limit:]]
            batch_context_qidxs.append(prev_qidxs)
            batch_context_search_time.append(prev_qsearch_time)

            context_pos_idxs = [] # list of positive document for each query.
            # e.g., [[2,3],[1],[5]], #pos * prev_q_limit
            context_q_cont_features, context_q_discrete_features = [], []
            # prev_q_limit * # of q_cont_features or # of q_discrete_features
            context_pos_d_cont_features, context_pos_d_discrete_features = [], []
            # prev_q_limit, doc_limit_per_q, # of d_cont_features
            context_pos_qd_cont_features = []
            # prev_q_limit, doc_limit_per_q, # of qd_cont_features
            pos_doc_dic, pos_conv_hash_dic = dict(), dict()
            if self.args.use_popularity:
                doc_context_popularity = dict()
                for prev_q in prev_qidxs:
                    for segs in self.personal_data.query_info_dic[prev_q][:-1]:
                        doc_id = int(segs[self.personal_data.feature_name_dic['m:DocId']])
                        doc_context_popularity[doc_id] = [0] * prev_q_limit
                for doc_id in candi_doc_idxs: # already padded; -1 included
                    doc_context_popularity[doc_id] = [0] * prev_q_limit
            # for ridx in range(len(prev_qidxs))[::-1]:
            for idx in range(len(prev_qidxs)):
                # dist = len(prev_qidxs) - ridx - 1 # distance to current query
                # # prev_q = prev_qidxs[ridx]
                prev_q = prev_qidxs[idx]
                pos_docid_rating = []
                pos_doc_dcont_features, pos_doc_ddiscrete_features = dict(), dict()
                pos_doc_qdcont_features = dict()
                # idx = prev_q_limit - 1 - dist
                query_cont_features, query_discrete_features, _, _, _ \
                    = self.personal_data.collect_group_features(
                        self.personal_data.query_info_dic[prev_q][0], \
                            self.personal_data.feature_name_dic)
                # some queries may have no positive documents, so only query features are kept
                for segs in self.personal_data.query_info_dic[prev_q][:-1]:
                    doc_id = int(segs[self.personal_data.feature_name_dic['m:DocId']])
                    rating = PersonalSearchData.RATING_MAP[
                        segs[self.personal_data.feature_name_dic['m:Rating']]]
                    if self.args.use_popularity:
                        doc_context_popularity[doc_id][idx] = rating + 1
                        # 0->1, 1->2, 2->3, 3->4, 4->5
                    if rating == 0:
                        continue
                    _, _, doc_cont_features, doc_discrete_features, \
                        qd_cont_features = self.personal_data.collect_group_features(
                            segs, self.personal_data.feature_name_dic)
                    conversation_hash = int(
                        segs[self.personal_data.feature_name_dic['AdvancedPreferFeature_153']])
                    pos_doc_dic[doc_id] = 1 # dist from 0 to prev_q_limit
                    pos_conv_hash_dic[conversation_hash] = 1

                    pos_doc_dcont_features[doc_id] = doc_cont_features
                    pos_doc_ddiscrete_features[doc_id] = doc_discrete_features[:-1]
                    # the last dimension is conversation hash
                    pos_doc_qdcont_features[doc_id] = qd_cont_features
                    pos_docid_rating.append((doc_id, rating))
                context_q_cont_features.append(query_cont_features)
                context_q_discrete_features.append(query_discrete_features)
                pos_docid_rating.sort(key=lambda x: x[1], reverse=True)
                pos_docids = [doc_id for doc_id, _ in pos_docid_rating[:self.args.doc_limit_per_q]]
                context_pos_idxs.append(pos_docids)
                context_pos_d_cont_features.append(
                    [pos_doc_dcont_features[d_id] for d_id in pos_docids])
                context_pos_d_discrete_features.append(
                    [pos_doc_ddiscrete_features[d_id] for d_id in pos_docids])
                context_pos_qd_cont_features.append(
                    [pos_doc_qdcont_features[d_id] for d_id in pos_docids])

            batch_context_qcont_features.append(context_q_cont_features)
            batch_context_qdiscrete_features.append(context_q_discrete_features)
            batch_context_pos_didxs.append(context_pos_idxs)
            batch_context_pos_dcont_features.append(context_pos_d_cont_features)
            batch_context_pos_ddiscrete_features.append(context_pos_d_discrete_features)
            batch_context_pos_qdcont_features.append(context_pos_qd_cont_features)

            conv_occur_pos = []
            doc_occur_pos = []
            if self.args.conv_occur:
                conv_occur_pos = [pos_conv_hash_dic[ch] \
                        if ch in pos_conv_hash_dic \
                            else 0 for ch in candi_conv_hash]
            if self.args.doc_occur:
                doc_occur_pos = [pos_doc_dic[d] \
                    if d in pos_doc_dic else 0 for d in candi_doc_idxs]
            batch_candi_conv_occur.append(conv_occur_pos)
            batch_candi_doc_occur.append(doc_occur_pos)
            if self.args.use_popularity:
                batch_candi_doc_popularity.append(
                    [doc_context_popularity[d] for d in candi_doc_idxs])
                doc_context_pop_idxs = []
                if len(context_pos_idxs) > 0:
                    for pos_doc_per_q in context_pos_idxs: # prev_q_limit, doc_limit_per_q
                        doc_context_pop_idxs.append(
                            [doc_context_popularity[d] for d in pos_doc_per_q])
                else:
                    doc_context_pop_idxs.append([])
                batch_context_d_popularity.append(doc_context_pop_idxs)
                # batch_size, prev_q_limit, doc_per_q, prev_q_limit
                batch_context_d_popularity = util.left_pad_4d_dim1(
                    batch_context_d_popularity, pad_id=0)
                batch_context_d_popularity = util.left_pad_4d_dim2(
                    batch_context_d_popularity, pad_id=0)
                # batch_candi_doc_popularity = util.left_pad_3d(
                #     batch_candi_doc_popularity, pad_id=0, dim=2)
                # batch_size, candi_count, prev_q_limit
            else:
                batch_candi_doc_popularity.append([])
                batch_context_d_popularity.append([])

        sum_prev_qcount = sum([len(d) for d in batch_context_qcont_features])
        # sum_prev_pos_dcount = sum([len(d) for d in batch_context_pos_didxs])
        # sum_prev_pos_dcount2 = sum([len(d) for d in batch_context_pos_dcont_features])
        sum_prev_pos_dcount = 0
        for qinfo in batch_context_pos_dcont_features:
            for posd in qinfo:
                sum_prev_pos_dcount += len(posd)

        # print(sum_prev_qcount, sum_prev_pos_dcount, sum_prev_pos_dcount2, sum_prev_pos_dcount3)
        # print(sum_prev_qcount, sum_prev_pos_dcount)
        if sum_prev_pos_dcount == 0 or sum_prev_qcount == 0:
            # print("no prev positive document")
            batch_context_pos_dcont_features[-1] = [[[0.] * self.personal_data.dcont_feat_count]]
            ddiscrete_feat_count = len(self.personal_data.DOC_FEATURES) \
                - self.personal_data.dcont_feat_count - 1 # conversation hash
            batch_context_pos_ddiscrete_features[-1] = [[[0] * ddiscrete_feat_count]]
            batch_context_pos_qdcont_features[-1] = [[[0.] * self.personal_data.qdcont_feat_count]]
            if sum_prev_qcount == 0:
                # print("no prev query")
                batch_context_qcont_features[-1] = [[0.] * self.personal_data.qcont_feat_count]
                batch_context_qdiscrete_features[-1] = [[0] * len(self.personal_data.QUERY_FEATURES[6:])]

        ###paddding###
        batch_context_qidxs = util.left_pad(
            batch_context_qidxs, pad_id=0, width=prev_q_limit)
        batch_context_search_time = util.left_pad(
            batch_context_search_time, pad_id=self.days_pad_idx, width=prev_q_limit)
        batch_context_pos_didxs = util.left_pad_3d(
            batch_context_pos_didxs, pad_id=self.doc_pad_idx, dim=1, width=prev_q_limit) #prev q count
        batch_context_pos_didxs = util.left_pad_3d(
            batch_context_pos_didxs, pad_id=self.doc_pad_idx, dim=2) #doc per q
        batch_context_qcont_features = util.left_pad_3d(
            batch_context_qcont_features, pad_id=0, dim=1, width=prev_q_limit)
            #batch_size, prev_q_limit, #features
        batch_context_qdiscrete_features = util.left_pad_3d(
            batch_context_qdiscrete_features, pad_id=0, dim=1, width=prev_q_limit)
        batch_context_pos_dcont_features = util.left_pad_4d_dim1(
            batch_context_pos_dcont_features, pad_id=0, width=prev_q_limit)
            #batch_size, prev_q_limit, doc_count_per_q, #features
        batch_context_pos_dcont_features = util.left_pad_4d_dim2(
            batch_context_pos_dcont_features, pad_id=0)
            #batch_size, prev_q_limit, doc_count_per_q, #features
        batch_context_pos_ddiscrete_features = util.left_pad_4d_dim1(
            batch_context_pos_ddiscrete_features, pad_id=0, width=prev_q_limit)
            #batch_size, prev_q_limit, doc_count_per_q, #features
        batch_context_pos_ddiscrete_features = util.left_pad_4d_dim2(
            batch_context_pos_ddiscrete_features, pad_id=0)
            #batch_size, prev_q_limit, doc_count_per_q, #features
        batch_context_pos_qdcont_features = util.left_pad_4d_dim1(
            batch_context_pos_qdcont_features, pad_id=0, width=prev_q_limit)
        batch_context_pos_qdcont_features = util.left_pad_4d_dim2(
            batch_context_pos_qdcont_features, pad_id=0)
        # batch_candi_doc_idxs already padded.
        batch_candi_conv_occur = util.pad(batch_candi_conv_occur, pad_id=prev_q_limit)
        batch_candi_doc_occur = util.pad(batch_candi_doc_occur, pad_id=prev_q_limit)

        return batch_context_qidxs, batch_context_search_time , batch_context_pos_didxs, \
            batch_context_qcont_features, batch_context_qdiscrete_features, \
                batch_context_pos_dcont_features, batch_context_pos_ddiscrete_features, \
                    batch_context_pos_qdcont_features, batch_context_d_popularity, \
                        batch_candi_doc_popularity, batch_candi_conv_occur, batch_candi_doc_occur
