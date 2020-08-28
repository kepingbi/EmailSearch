''' model for personalized email search
'''
import os
import torch
import torch.nn as nn
from models.group_encoder import AVGEncoder, FSEncoder, RNNEncoder
from models.transformer import TransformerEncoder
from models.optimizers import Optimizer
from models.examination_model import ExaminationModel
from others.logging import logger
from others.util import pad

class BaseEmailRanker(nn.Module):
    ''' baseline ranker that use the same feature set as the lambdamart model.
    '''
    g_qdiscrete_features = ["num_q_words", "num_q_operators", "item_type",
                            "locale_lcid", "culture_id", "query_lang_hash", "user_type"]
    g_doc_discrete_features = ["response_requested", "importance", "is_read",
                               "flag_status", "tolist_size", "cclist_size",
                               "bcclist_size", "to_position", "cc_position",
                               "email_class", # "conversation_hash",
                               "subject_prefix_hash"]

    g_qdiscrete_feat_idx = {y:x for x, y in enumerate(g_qdiscrete_features)}
    g_doc_discrete_feat_idx = {y:x for x, y in enumerate(g_doc_discrete_features)}
    g_max_weight = 15.
    # the following should be the same with PersonalSearchData
    # g_operators_count = 10 # set to 10, otherwise cut
    # g_tolist_size = 10 # set to 10, otherwise cut
    # g_cclist_size = 10 # set to 10, otherwise cut
    # g_bcclist_size = 10 # set to 10, otherwise cut
    # g_to_position_count = 10 # set to 10, otherwise cut
    # g_cc_position_count = 10 # set to 10, otherwise cut

    def __init__(self, args, personal_data, device):
        super(BaseEmailRanker, self).__init__()
        self.args = args
        self.personal_data = personal_data
        self.doc_pad_idx = personal_data.doc_pad_idx
        self.device = device
        self.discrete_qfeat_emb_size = [10, 10, 10, 10, 10, 10, 30]
        # num_q_words, num_q_operators, item_type, locale_lcid, culture_id,
        # query_lang_hash, user_type (consumer or commercial)
        self.discrete_dfeat_emb_size = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        self.embedding_size = self.args.embedding_size
        # response_requested, importance, is_read, flag_status, email_class
        # conversation_hash, subject_prefix_hash
        # embeddings for query discrete features
        self.num_qwords_emb = nn.Embedding(
            self.personal_data.g_qwords_count + 1,
            self.discrete_qfeat_emb_size[self.g_qdiscrete_feat_idx['num_q_words']],
            padding_idx=0)
        # print(self.personal_data.g_max_q_operator_count)
        self.num_qoperator_emb = nn.Embedding(
            # self.personal_data.g_max_q_operator_count + 1,
            self.personal_data.g_operators_count + 1,
            self.discrete_qfeat_emb_size[self.g_qdiscrete_feat_idx['num_q_operators']],
            padding_idx=0)
        self.item_type_emb = nn.Embedding(
            len(self.personal_data.item_types),
            self.discrete_qfeat_emb_size[self.g_qdiscrete_feat_idx['item_type']],
            padding_idx=0)
        self.locale_lcid_emb = nn.Embedding(
            len(self.personal_data.locale_lcids),
            self.discrete_qfeat_emb_size[self.g_qdiscrete_feat_idx['locale_lcid']],
            padding_idx=0)
        self.culture_id_emb = nn.Embedding(
            len(self.personal_data.culture_ids),
            self.discrete_qfeat_emb_size[self.g_qdiscrete_feat_idx['culture_id']],
            padding_idx=0)
        self.query_lang_emb = nn.Embedding(
            len(self.personal_data.query_lang_hashes),
            self.discrete_qfeat_emb_size[self.g_qdiscrete_feat_idx['query_lang_hash']],
            padding_idx=0)
        self.user_type_emb = nn.Embedding(
            len(self.personal_data.user_types),
            self.discrete_qfeat_emb_size[self.g_qdiscrete_feat_idx['user_type']],
            padding_idx=0)
        self.q_discrete_emb_list = [self.num_qwords_emb, self.num_qoperator_emb, \
            self.item_type_emb, self.locale_lcid_emb, self.culture_id_emb, \
                self.query_lang_emb, self.user_type_emb]
        # embeddings for document discrete features
        self.response_request_emb = nn.Embedding(
            3, # 0,1,2 (bool and 0 for padding)
            self.discrete_dfeat_emb_size[self.g_doc_discrete_feat_idx['response_requested']],
            padding_idx=0)
        self.importance_emb = nn.Embedding(
            4, # 0 for padding, 1,2,3
            self.discrete_dfeat_emb_size[self.g_doc_discrete_feat_idx['importance']],
            padding_idx=0)
        self.is_read_emb = nn.Embedding(
            3, # 0,1,2 (bool and 0 for padding)
            self.discrete_dfeat_emb_size[self.g_doc_discrete_feat_idx['is_read']],
            padding_idx=0)
        self.flag_emb = nn.Embedding(
            4, # 0, for padding, 0+1,1+1,2+1
            self.discrete_dfeat_emb_size[self.g_doc_discrete_feat_idx['flag_status']],
            padding_idx=0)
        self.tolist_size_emb = nn.Embedding(
            self.personal_data.g_tolist_size + 1, # 0, for padding,
            self.discrete_dfeat_emb_size[self.g_doc_discrete_feat_idx['tolist_size']],
            padding_idx=0)
        self.cclist_size_emb = nn.Embedding(
            self.personal_data.g_cclist_size + 1, # 0, for padding,
            self.discrete_dfeat_emb_size[self.g_doc_discrete_feat_idx['cclist_size']],
            padding_idx=0)
        self.bcclist_size_emb = nn.Embedding(
            self.personal_data.g_bcclist_size + 1, # 0, for padding,
            self.discrete_dfeat_emb_size[self.g_doc_discrete_feat_idx['bcclist_size']],
            padding_idx=0)
        self.to_position_emb = nn.Embedding(
            self.personal_data.g_to_position_count + 1, # 0, for padding,
            self.discrete_dfeat_emb_size[self.g_doc_discrete_feat_idx['to_position']],
            padding_idx=0)
        self.cc_position_emb = nn.Embedding(
            self.personal_data.g_cc_position_count + 1, # 0, for padding,
            self.discrete_dfeat_emb_size[self.g_doc_discrete_feat_idx['cc_position']],
            padding_idx=0)
        self.email_class_emb = nn.Embedding(
            len(self.personal_data.email_class_hashes),
            self.discrete_dfeat_emb_size[self.g_doc_discrete_feat_idx['email_class']],
            padding_idx=0)
        # self.conversation_hash_emb = nn.Embedding(
        #     len(self.personal_data.conversation_hashes),
        #     self.discrete_dfeat_emb_size[self.g_doc_discrete_feat_idx['conversation_hash']],
        #     padding_idx=0)
        self.subject_prefix_hash_emb = nn.Embedding(
            len(self.personal_data.subject_prefix_hashes),
            self.discrete_dfeat_emb_size[self.g_doc_discrete_feat_idx['subject_prefix_hash']],
            padding_idx=0)
        self.d_discrete_emb_list = [self.response_request_emb, self.importance_emb, \
            self.is_read_emb, self.flag_emb, self.tolist_size_emb, self.cclist_size_emb, \
                self.bcclist_size_emb, self.to_position_emb, self.cc_position_emb, \
                    self.email_class_emb, self.subject_prefix_hash_emb]
            # self.conversation_hash_emb,
        self.qcont_W1 = nn.Linear(
            self.personal_data.qcont_feat_count, self.args.embedding_size//2)
        self.qcont_batch_norm = nn.BatchNorm1d(self.args.embedding_size//2)
        self.dcont_W1 = nn.Linear(
            self.personal_data.dcont_feat_count, self.args.embedding_size//2)
        self.dcont_batch_norm = nn.BatchNorm1d(self.args.candi_doc_count)
        self.qdcont_W1 = nn.Linear(
            self.personal_data.qdcont_feat_count, self.args.embedding_size)
        self.qdcont_batch_norm = nn.BatchNorm1d(self.args.candi_doc_count)
        self.qdiscrete_W1 = nn.Linear(sum(self.discrete_qfeat_emb_size), int(self.embedding_size/2))
        self.qdiscrete_batch_norm = nn.BatchNorm1d(self.args.embedding_size//2)
        self.discrete_dfeat_hidden_size = sum(self.discrete_dfeat_emb_size) # + self.rating_emb_size
        self.ddiscrete_W1 = nn.Linear(self.discrete_dfeat_hidden_size, int(self.embedding_size/2))
        self.ddiscrete_batch_norm = nn.BatchNorm1d(self.args.candi_doc_count)
        self.emb_dropout = args.dropout
        self.attn_W1 = nn.Linear(self.embedding_size, self.embedding_size)
        if self.args.qinteract:
            self.qInteractFeatW = nn.Bilinear(self.embedding_size, self.embedding_size, 1)
        else:
            self.mlp_layer = nn.Linear(self.embedding_size, self.embedding_size//2)
            self.final_layer = nn.Linear(self.embedding_size//2, 1)
        self.dropout_layer = nn.Dropout(p=args.dropout)
        self.attn_batch_norm = nn.BatchNorm1d(self.args.candi_doc_count)
        if self.args.unbiased_train:
            self.exam_model = ExaminationModel(self.embedding_size, self.emb_dropout)
        #for each q,u,i
        #Q, previous purchases of u, current available reviews for i, padding value
        #self.logsoftmax = torch.nn.LogSoftmax(dim = -1)
        #self.bce_logits_loss = torch.nn.BCEWithLogitsLoss(reduction='none')#by default it's mean
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)
        self.initialize_parameters(logger) #logger
        self.to(device) #change model in place


    def load_cp(self, pt, strict=True):
        self.load_state_dict(pt['model'], strict=strict)

    def test(self, batch_data):
        candi_doc_idxs = batch_data.candi_doc_idxs
        #candi_doc_mask = candi_doc_idxs.ne(self.personal_data.doc_pad_idx)

        # batch_size, candi_count
        doc_scores, context_emb = self.compute_candi_doc_scores(batch_data)
        # mask already applied when computing the scores.
        if self.args.unbiased_train and self.args.show_propensity:
            relevance_pos = batch_data.candi_doc_rel_pos
            datetime_pos = batch_data.candi_doc_time_pos
            exam_output = self.exam_model(relevance_pos, datetime_pos)
            detached_exam_output = torch.softmax(exam_output.detach().clone(), dim=-1)
            norm_exam_output = detached_exam_output[:, 0].reshape(-1, 1) / detached_exam_output
            norm_exam_output = torch.clamp(\
                norm_exam_output, min=1./self.g_max_weight, max=self.g_max_weight)
            relevance_pos = relevance_pos.cpu().tolist()
            datetime_pos = datetime_pos.cpu().tolist()
            norm_exam_output = norm_exam_output.cpu().tolist()
            for rel, datetime, exam in zip(relevance_pos, datetime_pos, norm_exam_output):
                for idx, doc_rel in enumerate(rel):
                    print("{}_{}:{}".format(doc_rel, datetime[idx], exam[idx]))

            # should be larger than 1
        return doc_scores, context_emb

    def forward(self, batch_data):
        candi_doc_ratings = batch_data.candi_doc_ratings
        candi_doc_idxs = batch_data.candi_doc_idxs
        candi_doc_mask = candi_doc_idxs.ne(self.personal_data.doc_pad_idx)

        # batch_size, candi_count
        doc_scores, _ = self.compute_candi_doc_scores(batch_data)
        if self.args.unbiased_train:
            relevance_pos = batch_data.candi_doc_rel_pos
            datetime_pos = batch_data.candi_doc_time_pos
            exam_output = self.exam_model(relevance_pos, datetime_pos)
            detached_exam_output = torch.softmax(exam_output.detach().clone(), dim=-1)
            norm_exam_output = detached_exam_output[:, 0].reshape(-1, 1) / detached_exam_output
            norm_exam_output = torch.clamp(\
                norm_exam_output, min=1./self.g_max_weight, max=self.g_max_weight)
            # should be larger than 1
            rel_loss = -self.logsoftmax(doc_scores) * candi_doc_ratings.float() * norm_exam_output
            detached_doc_scores = torch.softmax(doc_scores.detach().clone(), dim=-1)
            norm_doc_scores = detached_doc_scores[:, 0].reshape(-1, 1) / detached_doc_scores
            norm_doc_scores = torch.clamp(\
                norm_doc_scores, min=1./self.g_max_weight, max=self.g_max_weight)
            exam_loss = -self.logsoftmax(exam_output) * candi_doc_ratings.float() * norm_doc_scores
            exam_loss = (exam_loss * candi_doc_mask.float()).sum(-1).mean()
        else:
            rel_loss = -self.logsoftmax(doc_scores) * candi_doc_ratings.float()
            # loss = -self.logsoftmax(doc_scores) * (candi_doc_ratings.float().exp()-1)
            exam_loss = None
        rel_loss = rel_loss * candi_doc_mask.float()
        rel_loss = rel_loss.sum(-1).mean()
        return rel_loss, exam_loss

    def compute_candi_doc_scores(self, batch_data):
        """ compute the scores of candidate documents in the batch
        """
        candi_doc_idxs = batch_data.candi_doc_idxs
        candi_doc_qcont_features = batch_data.candi_doc_qcont_features
        candi_doc_qdiscrete_features = batch_data.candi_doc_qdiscrete_features
        candi_doc_dcont_features = batch_data.candi_doc_dcont_features
        candi_doc_ddiscrete_features = batch_data.candi_doc_ddiscrete_features
        candi_doc_qdcont_features = batch_data.candi_doc_qdcont_features
        candi_doc_mask = candi_doc_idxs.ne(self.personal_data.doc_pad_idx)
        candi_doc_qcont_hidden, candi_doc_dcont_hidden, candi_doc_qdcont_hidden, \
            candi_doc_qdiscrete_hidden, candi_doc_ddiscrete_hidden = self.get_hidden_features(
                candi_doc_qcont_features, candi_doc_dcont_features,
                candi_doc_qdcont_features, candi_doc_qdiscrete_features,
                candi_doc_ddiscrete_features, candi_doc_mask)
        # print(candi_doc_idxs)
        # print(candi_doc_qcont_features)
        # print(candi_doc_qdiscrete_features)
        # print(candi_doc_dcont_features)
        # print(candi_doc_ddiscrete_features)
        # print(candi_doc_qdcont_features)
        # print(candi_doc_mask)
        candi_doc_q_hidden = torch.cat([candi_doc_qcont_hidden, candi_doc_qdiscrete_hidden], dim=-1)
        candi_doc_d_hidden = torch.cat([candi_doc_dcont_hidden, candi_doc_ddiscrete_hidden], dim=-1)
        _, candi_doc_count = candi_doc_idxs.size()

        expanded_candi_doc_q_hidden = candi_doc_q_hidden.unsqueeze(1).expand(\
            -1, candi_doc_count, -1)

        aggr_candi_emb = self.self_attn_weighted_avg(
            expanded_candi_doc_q_hidden, candi_doc_d_hidden, candi_doc_qdcont_hidden)
        aggr_candi_emb = self.attn_batch_norm(aggr_candi_emb)
        # collect the representation of the current query.
        # Current query features; current candidate documents;
        if self.args.qinteract:
            scores = self.qInteractFeatW(expanded_candi_doc_q_hidden.contiguous(), aggr_candi_emb)
        else:
            scores = self.final_layer(torch.tanh(self.mlp_layer(aggr_candi_emb)))
        # or dot product, probably not as good
        scores = scores.squeeze(-1) * candi_doc_mask
        return scores, candi_doc_q_hidden

    def self_attn_weighted_avg(self, doc_q_hidden, doc_d_hidden, doc_qdcont_hidden, is_candidate=True):
        ''' doc_q_hidden: batch_size, pre_q_limit/candi_count, embedding_size
            doc_d_hidden: batch_size, pre_q_limit/candi_count, embedding_size
            doc_qd_hidden: batch_size, pre_q_limit/candi_count, embedding_size
        '''
        all_feat_arr = []
        if self.args.qfeat or is_candidate:
            doc_q_hidden = torch.tanh(self.attn_W1(doc_q_hidden))
            all_feat_arr.append(doc_q_hidden)
        if self.args.dfeat or is_candidate:
            doc_d_hidden = torch.tanh(self.attn_W1(doc_d_hidden))
            all_feat_arr.append(doc_d_hidden)
        if self.args.qdfeat or is_candidate:
            doc_qdcont_hidden = torch.tanh(self.attn_W1(doc_qdcont_hidden))
            all_feat_arr.append(doc_qdcont_hidden)
        all_units = torch.stack(
            all_feat_arr, dim=-2)
            # batch_size, prev_q_limit/candi_count, 3, embedding_size

        attn = torch.softmax(torch.matmul(all_units, all_units.transpose(2, 3)), dim=-1)
        attn = self.dropout_layer(attn)
        # batch_size, prev_q_limit/candi_count, 3, 3
        aggr_emb = torch.matmul(attn, all_units).sum(dim=-2)
        # batch_size, prev_q_limit/candi_count, embedding_size
        return aggr_emb

    def get_hidden_features(self, doc_qcont_features,
                            doc_dcont_features, doc_qdcont_features, doc_qdiscrete_features,
                            doc_ddiscrete_features, doc_mask):
        ''' doc can be candi_doc, or context_doc
        '''
        doc_qcont_hidden = torch.tanh(self.qcont_W1(doc_qcont_features))
        doc_dcont_hidden = torch.tanh(self.dcont_W1(doc_dcont_features))
        doc_qdcont_hidden = torch.tanh(self.qdcont_W1(doc_qdcont_features))

        doc_qcont_hidden = self.qcont_batch_norm(doc_qcont_hidden)
        doc_dcont_hidden = self.dcont_batch_norm(doc_dcont_hidden)
        doc_qdcont_hidden = self.qdcont_batch_norm(doc_qdcont_hidden)
        # batch_size, qdiscrete_feature_count
        # cat_qdiscret_list = []
        # for idx in range(len(self.q_discrete_emb_list)):
        #     print(idx)
        #     cur_discrete_feat = self.q_discrete_emb_list[idx](doc_qdiscrete_features[:, idx])
        #     cat_qdiscret_list.append(cur_discrete_feat)
        # doc_qdiscrete_mapped = torch.cat(cat_qdiscret_list, dim=-1)

        doc_qdiscrete_mapped = torch.cat([self.q_discrete_emb_list[idx](
            doc_qdiscrete_features[:, idx]) for idx in range(
                len(self.q_discrete_emb_list))], dim=-1)
        # batch_size, sum(discrete_qfeat_emb_size)
        doc_qdiscrete_hidden = torch.tanh(self.qdiscrete_W1(doc_qdiscrete_mapped))
        # batch_size, doc_count, self.embedding_size
        doc_ddiscrete_mapped = torch.cat([self.d_discrete_emb_list[idx](
            doc_ddiscrete_features[:, :, idx]) for idx in range(
                len(self.d_discrete_emb_list))], dim=-1)
        # the last index conversation hash not used
        # concatenate with the popularity embedding
        doc_ddiscrete_hidden = torch.tanh(self.ddiscrete_W1(doc_ddiscrete_mapped))

        doc_qdiscrete_hidden = self.qdiscrete_batch_norm(doc_qdiscrete_hidden)
        doc_ddiscrete_hidden = self.ddiscrete_batch_norm(doc_ddiscrete_hidden)

        return doc_qcont_hidden, doc_dcont_hidden, doc_qdcont_hidden, \
            doc_qdiscrete_hidden, doc_ddiscrete_hidden


    def initialize_parameters(self, logger=None):
        if logger:
            logger.info("BaseEmailRanker initialization started.")
        # embeddings for query discrete features
        nn.init.normal_(self.num_qwords_emb.weight)
        nn.init.normal_(self.num_qoperator_emb.weight)
        nn.init.normal_(self.item_type_emb.weight)
        nn.init.normal_(self.locale_lcid_emb.weight)
        nn.init.normal_(self.culture_id_emb.weight)
        nn.init.normal_(self.query_lang_emb.weight)
        nn.init.normal_(self.user_type_emb.weight)

        # embeddings for document discrete features
        nn.init.normal_(self.response_request_emb.weight)
        nn.init.normal_(self.importance_emb.weight)
        nn.init.normal_(self.is_read_emb.weight)
        nn.init.normal_(self.flag_emb.weight)
        nn.init.normal_(self.tolist_size_emb.weight)
        nn.init.normal_(self.cclist_size_emb.weight)
        nn.init.normal_(self.bcclist_size_emb.weight)
        nn.init.normal_(self.to_position_emb.weight)
        nn.init.normal_(self.cc_position_emb.weight)
        nn.init.normal_(self.email_class_emb.weight)
        # nn.init.normal_(self.conversation_hash_emb.weight)
        nn.init.normal_(self.subject_prefix_hash_emb.weight)

        for name, p in self.named_parameters():
            if "W1.weight" in name:
                if logger:
                    logger.info(" {} ({}): Xavier normal init.".format(
                        name, ",".join([str(x) for x in p.size()])))
                nn.init.xavier_normal_(p)
            elif "W1.bias" in name:
                if logger:
                    logger.info(" {} ({}): constant (0) init.".format(
                        name, ",".join([str(x) for x in p.size()])))
                nn.init.constant_(p, 0)
            # else:
            #     if logger:
            #         logger.info(" {} ({}): random normal init.".format(
            #             name, ",".join([str(x) for x in p.size()])))
            #     nn.init.normal_(p)
        if logger:
            logger.info("BaseEmailRanker initialization finished.")
