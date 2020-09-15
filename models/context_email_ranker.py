''' model for personalized email search
'''
import os
import torch
import torch.nn as nn
from models.group_encoder import AVGEncoder, FSEncoder, RNNEncoder
from models.transformer import TransformerEncoder
from models.optimizers import Optimizer
from models.base_email_ranker import BaseEmailRanker
from others.logging import logger
from others.util import pad

def build_optim(args, model, checkpoint):
    """ Build optimizer """
    saved_optimizer_state_dict = None

    if args.train_from != '' and checkpoint is not None:
        optim = checkpoint['optim']
        saved_optimizer_state_dict = optim.optimizer.state_dict()
    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method=args.decay_method,
            warmup_steps=args.warmup_steps,
            weight_decay=args.l2_lambda)
        #self.start_decay_steps take effect when decay_method is not noam

    optim.set_parameters(list(model.named_parameters()))

    if args.train_from != '' and checkpoint is not None:
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.device == "cuda":
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    return optim

class ContextEmailRanker(BaseEmailRanker):
    def __init__(self, args, personal_data, device):
        super(ContextEmailRanker, self).__init__(args, personal_data, device)
        self.additional_dim = 0
        add_dfeat_size = [30, 30, 30]
        if self.args.use_popularity:
            self.rating_emb_size = add_dfeat_size[0]
            self.rating_emb = nn.Embedding(
                self.personal_data.rating_levels,
                self.rating_emb_size,
                padding_idx=0)
            self.additional_dim += self.rating_emb_size
            #popularity
        if self.args.conv_occur:
            self.conv_occur_emb_size = add_dfeat_size[1]
            self.conv_occur_emb = nn.Embedding(self.args.prev_q_limit + 1, \
                self.conv_occur_emb_size, \
                padding_idx=self.args.prev_q_limit)
            self.d_discrete_emb_list.append(self.conv_occur_emb)
            self.additional_dim += self.conv_occur_emb_size
        if self.args.doc_occur:
            self.doc_occur_emb_size = add_dfeat_size[2]
            self.doc_occur_emb = nn.Embedding(self.args.prev_q_limit + 1, \
                self.doc_occur_emb_size, \
                padding_idx=self.args.prev_q_limit)
            self.d_discrete_emb_list.append(self.doc_occur_emb)
            self.additional_dim += self.doc_occur_emb_size

        if self.args.sep_mapping:
            self.pos_qcont_W1 = nn.Linear(
                self.personal_data.qcont_feat_count, self.args.embedding_size//2)
            self.pos_dcont_W1 = nn.Linear(
                self.personal_data.dcont_feat_count, self.args.embedding_size//2)
            self.pos_qdcont_W1 = nn.Linear(
                self.personal_data.qdcont_feat_count, self.args.embedding_size)
            self.pos_qdiscrete_W1 = nn.Linear(sum(self.discrete_qfeat_emb_size), int(self.embedding_size/2))
            self.pos_ddiscrete_W1 = nn.Linear(self.discrete_dfeat_hidden_size, int(self.embedding_size/2))
            self.pos_attn_W1 = nn.Linear(self.embedding_size, self.embedding_size)

        if self.additional_dim > 0:
            self.discrete_dfeat_hidden_size = sum(self.discrete_dfeat_emb_size) \
                + self.additional_dim
            self.ddiscrete_W1 = nn.Linear(self.discrete_dfeat_hidden_size, self.embedding_size//2)
        #self.all_feat_hidden_size = 3 * self.embedding_size
        self.transformer_emb_size = self.embedding_size // 2 if self.args.compress else self.embedding_size
        if self.args.date_emb:
            self.date_gap_emb = nn.Embedding(
                self.personal_data.days_pad_idx + 1,
                self.transformer_emb_size,
                padding_idx=self.personal_data.days_pad_idx)

        if not self.args.do_curq:
            self.end_cls_emb = nn.Parameter(torch.rand(1, self.transformer_emb_size), requires_grad=True)
        self.prev_q_limit = 1 if self.args.prev_q_limit == 0 else self.args.prev_q_limit
        # if self.args.mode != "test":
        self.context_q_batch_norm = nn.BatchNorm1d(self.prev_q_limit)
        self.context_d_batch_norm = nn.BatchNorm1d(self.prev_q_limit)
        self.context_qd_batch_norm = nn.BatchNorm1d(self.prev_q_limit)
        self.context_attn_batch_norm = nn.BatchNorm1d(self.prev_q_limit)
        if self.args.compress:
            self.compress_layer = nn.Linear(self.embedding_size, self.embedding_size // 2)
        if self.args.do_curd:
            self.final_layer = nn.Linear(self.transformer_emb_size, 1)
        if self.args.qinteract:
            self.contextq_all_feat_W = nn.Bilinear(self.embedding_size, self.embedding_size, 1)
        else:
            self.context_mlp_layer = nn.Linear(self.transformer_emb_size + self.embedding_size, self.embedding_size//2)
            self.context_final_layer = nn.Linear(self.embedding_size//2, 1)
        # args.ff_size
        self.transformer_encoder = TransformerEncoder(
            self.transformer_emb_size, self.transformer_emb_size//3 * 2, args.heads,
            args.dropout, args.inter_layers)
        if args.use_popularity:
            self.popularity_encoder_name = args.popularity_encoder_name
            if args.popularity_encoder_name == "transformer":
                self.popularity_encoder = TransformerEncoder(
                    self.rating_emb_size, args.pop_ff_size, args.pop_heads,
                    args.dropout, args.pop_inter_layers)
            else:
                # LSTM
                self.popularity_encoder = RNNEncoder(
                    False, args.pop_inter_layers, self.rating_emb_size,
                    self.rating_emb_size, args.dropout)

        if args.query_encoder_name == "fs":
            self.query_encoder = FSEncoder(self.embedding_size, self.emb_dropout)
        else:
            self.query_encoder = AVGEncoder(self.embedding_size, self.emb_dropout)
        # for each q,u,i
        # Q, previous purchases of u, current available reviews for i, padding value
        # self.logsoftmax = torch.nn.LogSoftmax(dim = -1)
        # self.bce_logits_loss = torch.nn.BCEWithLogitsLoss(reduction='none')#by default it's mean
        # self.logsoftmax = torch.nn.LogSoftmax(dim=-1)
        self.initialize_extra_parameters(logger) #logger
        self.to(device) #change model in place

    # def test(self, batch_data):
    #     candi_doc_idxs = batch_data.candi_doc_idxs
    #     candi_doc_mask = candi_doc_idxs.ne(self.personal_data.doc_pad_idx)

    #     # batch_size, candi_count
    #     doc_scores = self.compute_candi_doc_scores(batch_data)
    #     return doc_scores

    # def forward(self, batch_data):
    #     candi_doc_ratings = batch_data.candi_doc_ratings
    #     candi_doc_idxs = batch_data.candi_doc_idxs
    #     candi_doc_mask = candi_doc_idxs.ne(self.personal_data.doc_pad_idx)

    #     # batch_size, candi_count
    #     doc_scores = self.compute_candi_doc_scores(batch_data)
    #     loss = -self.logsoftmax(doc_scores) * candi_doc_ratings.float()
    #     return loss

    def compute_candi_doc_scores(self, batch_data):
        """ compute the scores of candidate documents in the batch
        """
        candi_doc_idxs = batch_data.candi_doc_idxs
        candi_doc_qcont_features = batch_data.candi_doc_qcont_features
        candi_doc_qdiscrete_features = batch_data.candi_doc_qdiscrete_features
        candi_doc_dcont_features = batch_data.candi_doc_dcont_features
        candi_doc_ddiscrete_features = batch_data.candi_doc_ddiscrete_features
        candi_doc_qdcont_features = batch_data.candi_doc_qdcont_features
        candi_doc_popularity = batch_data.candi_doc_popularity
        candi_conv_occur = batch_data.candi_conv_occur.unsqueeze(-1)
        candi_doc_occur = batch_data.candi_doc_occur.unsqueeze(-1)
        if self.args.conv_occur:
            candi_doc_ddiscrete_features = torch.cat(
                [candi_doc_ddiscrete_features, candi_conv_occur], dim=-1)
        if self.args.doc_occur:
            candi_doc_ddiscrete_features = torch.cat(
                [candi_doc_ddiscrete_features, candi_doc_occur], dim=-1)

        # batch_size, candi_doc_count, prev_q_limit
        context_qidxs = batch_data.context_qidxs
        context_search_time = batch_data.context_search_time
        context_pos_didxs = batch_data.context_pos_didxs
        context_qcont_features = batch_data.context_qcont_features
        context_qdiscrete_features = batch_data.context_qdiscrete_features
        context_pos_dcont_features = batch_data.context_pos_dcont_features
        context_pos_ddiscrete_features = batch_data.context_pos_ddiscrete_features
        context_pos_qdcont_features = batch_data.context_pos_qdcont_features
        context_d_popularity = batch_data.context_d_popularity

        candi_doc_mask = candi_doc_idxs.ne(self.personal_data.doc_pad_idx)
        candi_doc_qcont_hidden, candi_doc_dcont_hidden, candi_doc_qdcont_hidden, \
            candi_doc_qdiscrete_hidden, candi_doc_ddiscrete_hidden = super().get_hidden_features(
                candi_doc_qcont_features, candi_doc_dcont_features,
                candi_doc_qdcont_features, candi_doc_qdiscrete_features,
                candi_doc_ddiscrete_features, candi_doc_mask)
        # candi_doc_ddiscrete_features, candi_doc_popularity, candi_doc_mask)
        # candi_doc_qcont_hidden = self.qcont_batch_norm(candi_doc_qcont_hidden)
        # candi_doc_dcont_hidden = self.dcont_batch_norm(candi_doc_dcont_hidden)
        # candi_doc_qdcont_hidden = self.qdcont_batch_norm(candi_doc_qdcont_hidden)
        # candi_doc_qdiscrete_hidden = self.qdiscrete_batch_norm(candi_doc_qdiscrete_hidden)
        # candi_doc_ddiscrete_hidden = self.ddiscrete_batch_norm(candi_doc_ddiscrete_hidden)

        # batch_size, qcont_hidden_size
        batch_size, candi_doc_count, _ = candi_doc_dcont_hidden.size()
        candi_doc_q_hidden = torch.cat([candi_doc_qcont_hidden, candi_doc_qdiscrete_hidden], dim=-1)
        candi_doc_d_hidden = torch.cat([candi_doc_dcont_hidden, candi_doc_ddiscrete_hidden], dim=-1)
        candi_doc_q_hidden = candi_doc_q_hidden.unsqueeze(1).expand(-1, candi_doc_count, -1)
        if self.args.query_attn:
            aggr_candi_emb = self.attn_weighted_avg(
                candi_doc_q_hidden, candi_doc_d_hidden, candi_doc_qdcont_hidden)
        else:
            aggr_candi_emb = self.self_attn_weighted_avg(self.attn_W1, \
                candi_doc_q_hidden, candi_doc_d_hidden, candi_doc_qdcont_hidden)
        aggr_candi_emb = self.attn_batch_norm(aggr_candi_emb)
        # collect the representation of the current query.
        # Current query features; current candidate documents;

        if self.args.do_curq:
            last_hidden = candi_doc_q_hidden[:, 0, :].unsqueeze(1)
            # batch_size,1,embedding_size
        else:
            last_hidden = self.end_cls_emb.unsqueeze(0).expand(batch_size, -1, -1)
        context_seq_emb = last_hidden
        # batch_size, 1, embedding_size
        context_seq_mask = torch.ones(batch_size, 1, dtype=bool, device=last_hidden.device)
        if self.args.date_emb:
            context_seq_days = torch.ones(
                batch_size, 1, dtype=int, device=last_hidden.device) * (
                    self.personal_data.days_pad_idx - 1)
        if self.args.prev_q_limit > 0 or self.args.test_prev_q_limit > 0:
            batch_size, prev_q_limit, doc_per_q, _ = context_pos_dcont_features.size()
            # actual maximum context query count in the batch
            # context_qcont_features = context_qcont_features.view(
            #     batch_size * prev_q_limit, -1)
            context_pos_dcont_features = context_pos_dcont_features.view(
                batch_size * prev_q_limit, doc_per_q, -1)
            context_pos_qdcont_features = context_pos_qdcont_features.view(
                batch_size * prev_q_limit, doc_per_q, -1)
            context_qdiscrete_features = context_qdiscrete_features.view(
                batch_size * prev_q_limit, -1)
                # to be same as query discrete features for candidate doc
            context_pos_ddiscrete_features = context_pos_ddiscrete_features.view(
                batch_size * prev_q_limit, doc_per_q, -1)
            if self.args.conv_occur:
                context_pos_ddiscrete_features = torch.cat(
                    [context_pos_ddiscrete_features, \
                        context_pos_ddiscrete_features.new(
                            batch_size * prev_q_limit, doc_per_q, 1).fill_(
                                self.args.prev_q_limit)], dim=-1)
            if self.args.doc_occur:
                context_pos_ddiscrete_features = torch.cat(
                    [context_pos_ddiscrete_features, \
                        context_pos_ddiscrete_features.new(
                            batch_size * prev_q_limit, doc_per_q, 1).fill_(
                                self.args.prev_q_limit)], dim=-1)
            context_d_popularity = context_d_popularity.view(
                batch_size * prev_q_limit, doc_per_q, -1)
            context_doc_mask = context_pos_didxs.ne(self.personal_data.doc_pad_idx)
            context_doc_mask = context_doc_mask.view(batch_size * prev_q_limit, -1)
            # print(context_pos_dcont_features.size())
            # print(context_pos_ddiscrete_features.size())

            context_doc_qcont_hidden, context_doc_dcont_hidden, context_doc_qdcont_hidden, \
                context_doc_qdiscrete_hidden, context_doc_ddiscrete_hidden = self.get_hidden_features(
                    context_qcont_features, context_pos_dcont_features,
                    context_pos_qdcont_features, context_qdiscrete_features,
                    context_pos_ddiscrete_features, context_d_popularity, context_doc_mask)
            context_doc_d_hidden = torch.cat(
                [context_doc_dcont_hidden, context_doc_ddiscrete_hidden], dim=-1)

            context_doc_d_hidden = self.query_encoder(context_doc_d_hidden, context_doc_mask)
            context_doc_qdcont_hidden = self.query_encoder(context_doc_qdcont_hidden, context_doc_mask)
            context_doc_qdcont_hidden = context_doc_qdcont_hidden.view(
                batch_size, prev_q_limit, -1)
            context_doc_qdiscrete_hidden = context_doc_qdiscrete_hidden.view(
                batch_size, prev_q_limit, -1)
            context_doc_q_hidden = torch.cat(
                [context_doc_qcont_hidden, context_doc_qdiscrete_hidden], dim=-1)
            context_doc_d_hidden = context_doc_d_hidden.view(
                batch_size, prev_q_limit, -1)
            # if self.args.mode != "test":
            #in case the prev_q_limit during test is different from during training
            context_doc_q_hidden = self.context_q_batch_norm(context_doc_q_hidden)
            context_doc_d_hidden = self.context_d_batch_norm(context_doc_d_hidden)
            context_doc_qdcont_hidden = self.context_qd_batch_norm(context_doc_qdcont_hidden)
            attn_W1 = self.pos_attn_W1 if self.args.sep_mapping else self.attn_W1
            if self.args.query_attn:
                aggr_context_emb = self.attn_weighted_avg(
                    context_doc_q_hidden, context_doc_d_hidden, \
                        context_doc_qdcont_hidden, is_candidate=False)
            else:
                aggr_context_emb = self.self_attn_weighted_avg(attn_W1, \
                    context_doc_q_hidden, context_doc_d_hidden, \
                        context_doc_qdcont_hidden, is_candidate=False)
            
            # if self.args.mode != "test":
            aggr_context_emb = self.context_attn_batch_norm(aggr_context_emb)

            if self.args.compress:
                aggr_context_emb = torch.tanh(self.compress_layer(aggr_context_emb))
                if self.args.do_curd:
                    aggr_candi_emb = torch.tanh(self.compress_layer(aggr_candi_emb))
            # batch_size, prev_q_limit, embedding_size
            context_q_mask = context_qidxs.ne(0) # query pad id
                
            if self.args.do_curd:
                cat_seq_emb = torch.cat([aggr_context_emb.unsqueeze(1).expand(
                    -1, candi_doc_count, -1, -1), aggr_candi_emb.unsqueeze(2)], dim=2)
                # batch_size, candi_count, prev_q_limit+1, embedding_size
                context_seq_emb = cat_seq_emb.view(batch_size * candi_doc_count, -1, self.transformer_emb_size)
                cat_q_mask = torch.cat([context_q_mask.unsqueeze(1).expand(
                    -1, candi_doc_count, -1), context_q_mask.new_ones(batch_size, candi_doc_count, 1)], dim=2)
                context_seq_mask = cat_q_mask.view(batch_size * candi_doc_count, -1)
                if self.args.date_emb:
                    cat_seq_days = torch.cat([context_search_time.unsqueeze(1).expand(
                        -1, candi_doc_count, -1), context_search_time.new_ones(
                            batch_size, candi_doc_count, 1) * (self.personal_data.days_pad_idx - 1)], dim=-1)
                    context_seq_days = cat_seq_days.view(batch_size * candi_doc_count, -1)
            else:
                context_seq_emb = torch.cat([aggr_context_emb, last_hidden], dim=1)
                # batch_size, prev_q_limit+1, embedding_size
                context_seq_mask = torch.cat(
                    [context_q_mask, context_q_mask.new_ones(batch_size, 1)], dim=-1)

                if self.args.date_emb:
                    context_seq_days = torch.cat([context_search_time, context_seq_days], dim=-1)

        if self.args.date_emb:
            context_date_emb = self.date_gap_emb(context_seq_days)
            # batch_size, prev_q_limit+1, transformer_embed_size
            context_seq_emb = context_seq_emb + context_date_emb
            # or batch_size * candi_doc_count, prev_q_limit+1, transformer_embed_size

        context_overall_emb = self.transformer_encoder(
            context_seq_emb, context_seq_mask, self.args.use_pos_emb)
        # the embedding corresponding to the last position
        # batch_size, candi_count, embedding_size
        # candidate score: batch_size, candi_count
        if self.args.qinteract:
            expanded_context_overall_emb = context_overall_emb.unsqueeze(1).expand_as(aggr_candi_emb)
            scores = self.contextq_all_feat_W(expanded_context_overall_emb.contiguous(), aggr_candi_emb)
        elif self.args.do_curd:
            scores = self.final_layer(context_overall_emb).view(batch_size, candi_doc_count, -1)
        else:
            expanded_context_overall_emb = context_overall_emb.unsqueeze(1).expand(-1, candi_doc_count, -1)
            concat_overall_emb = torch.cat([aggr_candi_emb, expanded_context_overall_emb], dim=-1)
            scores = self.context_final_layer(torch.tanh(self.context_mlp_layer(concat_overall_emb)))

        # or dot product, probably not as good
        scores = scores.squeeze(-1) * candi_doc_mask
        return scores, context_overall_emb

    def get_hidden_features(self, doc_qcont_features,
                            doc_dcont_features, doc_qdcont_features, doc_qdiscrete_features,
                            doc_ddiscrete_features, doc_popularity, doc_mask):
        ''' doc can be candi_doc, or context_doc
        '''
        # batch_size, hidden_size
        qcont_W1 = self.pos_qcont_W1 if self.args.sep_mapping else self.qcont_W1
        dcont_W1 = self.pos_dcont_W1 if self.args.sep_mapping else self.dcont_W1
        qdcont_W1 = self.pos_qdcont_W1 if self.args.sep_mapping else self.qdcont_W1
        qdiscrete_W1 = self.pos_qdiscrete_W1 if self.args.sep_mapping else self.qdiscrete_W1
        ddiscrete_W1 = self.pos_ddiscrete_W1 if self.args.sep_mapping else self.ddiscrete_W1

        doc_qcont_hidden = torch.tanh(qcont_W1(doc_qcont_features))
        doc_dcont_hidden = torch.tanh(dcont_W1(doc_dcont_features))
        doc_qdcont_hidden = torch.tanh(qdcont_W1(doc_qdcont_features))

        # batch_size, qdiscrete_feature_count
        doc_qdiscrete_mapped = torch.cat([self.q_discrete_emb_list[idx](
            doc_qdiscrete_features[:, idx]) for idx in range(
                len(self.q_discrete_emb_list))], dim=-1)
        # batch_size, sum(discrete_qfeat_emb_size)
        doc_qdiscrete_hidden = torch.tanh(qdiscrete_W1(doc_qdiscrete_mapped))
        # batch_size, doc_count, self.embedding_size
        doc_ddiscrete_mapped = torch.cat([self.d_discrete_emb_list[idx](
            doc_ddiscrete_features[:, :, idx]) for idx in range(
                len(self.d_discrete_emb_list))], dim=-1)
        if self.args.use_popularity:
            # concatenate with the popularity embedding
            batch_size, doc_count, _ = doc_popularity.size()

            if self.popularity_encoder_name == "transformer":
                doc_popularity = torch.cat([doc_popularity, doc_popularity.new(
                    batch_size, doc_count, 1).fill_(self.personal_data.rating_levels-1)], dim=-1)
                # the last position corresponds to the final representation of the sequence.
            # prev_q_limit is pre-defined size
            doc_popularity_emb = self.rating_emb(doc_popularity)
            # batch_size, doc_count, prev_q_limit(+1), rating_embed_size
            pop_seq_emb = doc_popularity_emb.view(
                -1, self.prev_q_limit, self.rating_emb_size)
            doc_pop_emb = self.popularity_encoder(pop_seq_emb, doc_mask)
            doc_pop_emb = doc_pop_emb.view(
                batch_size, doc_count, self.rating_emb_size)
            #doc_ddiscrete_hidden = torch.tanh(self.ddiscrete_W1(doc_ddiscrete_mapped))
            doc_ddiscrete_mapped = torch.cat([doc_ddiscrete_mapped, doc_pop_emb], dim=-1)
        doc_ddiscrete_hidden = torch.tanh(ddiscrete_W1(doc_ddiscrete_mapped))

        return doc_qcont_hidden, doc_dcont_hidden, doc_qdcont_hidden, \
            doc_qdiscrete_hidden, doc_ddiscrete_hidden

    def initialize_extra_parameters(self, logger=None):
        if logger:
            logger.info("Extra parameters in ContextEmailRanker initialization started.")
        # embeddings for query discrete features

        # recent focus
        if self.args.use_popularity:
            nn.init.normal_(self.rating_emb.weight)
            self.popularity_encoder.initialize_parameters(logger)
        if self.args.conv_occur:
            nn.init.normal_(self.conv_occur_emb.weight)
        if self.args.doc_occur:
            nn.init.normal_(self.doc_occur_emb.weight)
        if self.args.date_emb:
            nn.init.normal_(self.date_gap_emb.weight)

        self.query_encoder.initialize_parameters(logger)
        self.transformer_encoder.initialize_parameters(logger)
        if self.additional_dim > 0:
            nn.init.xavier_normal_(self.ddiscrete_W1.weight)
            nn.init.constant_(self.ddiscrete_W1.bias, 0)
        if self.args.qinteract:
            nn.init.xavier_normal_(self.contextq_all_feat_W.weight)
            nn.init.constant_(self.contextq_all_feat_W.bias, 0)

        for name, p in self.named_parameters():
            if "layer.weight" in name or "W1.weight" in name:
                if logger:
                    logger.info(" {} ({}): Xavier normal init.".format(
                        name, ",".join([str(x) for x in p.size()])))
                nn.init.xavier_normal_(p)
            elif "layer.bias" in name or "W1.bias" in name:
                if logger:
                    logger.info(" {} ({}): constant (0) init.".format(
                        name, ",".join([str(x) for x in p.size()])))
                nn.init.constant_(p, 0)

        if logger:
            logger.info("ContextEmailRanker initialization finished.")
