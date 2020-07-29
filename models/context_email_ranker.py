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
        additional_dim = 0
        add_dfeat_size = [30, 30, 30]
        if self.args.use_popularity:
            self.rating_emb_size = add_dfeat_size[0]
            self.rating_emb = nn.Embedding(
                self.personal_data.rating_levels,
                self.rating_emb_size,
                padding_idx=0)
            additional_dim += self.rating_emb_size
            #popularity
        if self.args.conv_occur:
            self.conv_occur_emb_size = add_dfeat_size[1]
            self.conv_occur_emb = nn.Embedding(self.args.prev_q_limit + 1, \
                self.conv_occur_emb_size, \
                padding_idx=self.args.prev_q_limit)
            self.d_discrete_emb_list.append(self.conv_occur_emb)
            additional_dim += self.conv_occur_emb_size
        if self.args.doc_occur:
            self.doc_occur_emb_size = add_dfeat_size[2]
            self.doc_occur_emb = nn.Embedding(self.args.prev_q_limit + 1, \
                self.doc_occur_emb_size, \
                padding_idx=self.args.prev_q_limit)
            self.d_discrete_emb_list.append(self.doc_occur_emb)
            additional_dim += self.doc_occur_emb_size

        self.discrete_dfeat_hidden_size = sum(self.discrete_dfeat_emb_size) + additional_dim
        self.ddiscrete_W1 = nn.Linear(self.discrete_dfeat_hidden_size, self.embedding_size//2)
        #self.all_feat_hidden_size = 3 * self.embedding_size

        self.contextq_all_feat_W = nn.Bilinear(self.embedding_size, self.embedding_size, 1)
        self.transformer_encoder = TransformerEncoder(
            self.embedding_size, args.ff_size, args.heads,
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
        context_pos_didxs = batch_data.context_pos_didxs
        context_qcont_features = batch_data.context_qcont_features
        context_qdiscrete_features = batch_data.context_qdiscrete_features
        context_pos_dcont_features = batch_data.context_pos_dcont_features
        context_pos_ddiscrete_features = batch_data.context_pos_ddiscrete_features
        context_pos_qdcont_features = batch_data.context_pos_qdcont_features
        context_d_popularity = batch_data.context_d_popularity

        candi_doc_mask = candi_doc_idxs.ne(self.personal_data.doc_pad_idx)
        candi_doc_qcont_hidden, candi_doc_dcont_hidden, candi_doc_qdcont_hidden, \
            candi_doc_qdiscrete_hidden, candi_doc_ddiscrete_hidden = self.get_hidden_features(
                candi_doc_qcont_features, candi_doc_dcont_features,
                candi_doc_qdcont_features, candi_doc_qdiscrete_features,
                candi_doc_ddiscrete_features, candi_doc_popularity, candi_doc_mask)
        # batch_size, qcont_hidden_size
        _, candi_doc_count, _ = candi_doc_dcont_hidden.size()
        candi_doc_q_hidden = torch.cat([candi_doc_qcont_hidden, candi_doc_qdiscrete_hidden], dim=-1)
        candi_doc_d_hidden = torch.cat([candi_doc_dcont_hidden, candi_doc_ddiscrete_hidden], dim=-1)
        candi_doc_q_hidden = candi_doc_q_hidden.unsqueeze(1).expand(-1, candi_doc_count, -1)
        aggr_candi_emb = self.self_attn_weighted_avg(
            candi_doc_q_hidden, candi_doc_d_hidden, candi_doc_qdcont_hidden)

        # collect the representation of the current query.
        # Current query features; current candidate documents;
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
        context_doc_qcont_hidden, context_doc_dcont_hidden, context_doc_qdcont_hidden, \
            context_doc_qdiscrete_hidden, context_doc_ddiscrete_hidden = self.get_hidden_features(
                context_qcont_features, context_pos_dcont_features,
                context_pos_qdcont_features, context_qdiscrete_features,
                context_pos_ddiscrete_features, context_d_popularity, context_doc_mask)
        #context_doc_qcont_hidden = self.query_encoder(context_doc_qcont_hidden, context_doc_mask)
        context_doc_d_hidden = torch.cat(
            [context_doc_dcont_hidden, context_doc_ddiscrete_hidden], dim=-1)

        context_doc_d_hidden = self.query_encoder(context_doc_d_hidden, context_doc_mask)
        context_doc_qdcont_hidden = self.query_encoder(context_doc_qdcont_hidden, context_doc_mask)
        context_doc_dcont_hidden = context_doc_dcont_hidden.view(
            batch_size, prev_q_limit, -1)
        context_doc_qdcont_hidden = context_doc_qdcont_hidden.view(
            batch_size, prev_q_limit, -1)
        context_doc_qdiscrete_hidden = context_doc_qdiscrete_hidden.view(
            batch_size, prev_q_limit, -1)
        context_doc_ddiscrete_hidden = context_doc_ddiscrete_hidden.view(
            batch_size, prev_q_limit, -1)
        context_doc_q_hidden = torch.cat(
            [context_doc_qcont_hidden, context_doc_qdiscrete_hidden], dim=-1)
        context_doc_d_hidden = context_doc_d_hidden.view(
            batch_size, prev_q_limit, -1)

        aggr_context_emb = self.self_attn_weighted_avg(
            context_doc_q_hidden, context_doc_d_hidden, context_doc_qdcont_hidden)
        candi_doc_q_hidden = candi_doc_q_hidden[:, 0, :].unsqueeze(1) # batch_size,1,embedding_size
        context_seq_emb = torch.cat([aggr_context_emb, candi_doc_q_hidden], dim=1)
        # batch_size, prev_q_limit+1, embedding_size
        context_q_mask = context_qidxs.ne(0) # query pad id
        context_seq_mask = torch.cat(
            [context_q_mask, context_q_mask.new_ones(batch_size, 1)], dim=-1)
        context_overall_emb = self.transformer_encoder(context_seq_emb, context_seq_mask)
        context_overall_emb = context_overall_emb.unsqueeze(1).expand_as(aggr_candi_emb)
        # batch_size, candi_count, embedding_size
        # candidate score: batch_size, candi_count
        scores = self.contextq_all_feat_W(context_overall_emb.contiguous(), aggr_candi_emb)
        # or dot product, probably not as good
        scores = scores.squeeze(-1) * candi_doc_mask
        return scores

    def get_hidden_features(self, doc_qcont_features,
                            doc_dcont_features, doc_qdcont_features, doc_qdiscrete_features,
                            doc_ddiscrete_features, doc_popularity, doc_mask):
        ''' doc can be candi_doc, or context_doc
        '''
        # batch_size, hidden_size
        doc_qcont_hidden = torch.tanh(self.qcont_W1(doc_qcont_features))
        doc_dcont_hidden = torch.tanh(self.dcont_W1(doc_dcont_features))
        doc_qdcont_hidden = torch.tanh(self.qdcont_W1(doc_qdcont_features))

        # batch_size, qdiscrete_feature_count
        doc_qdiscrete_mapped = torch.cat([self.q_discrete_emb_list[idx](
            doc_qdiscrete_features[:, idx]) for idx in range(
                len(self.q_discrete_emb_list))], dim=-1)
        # batch_size, sum(discrete_qfeat_emb_size)
        doc_qdiscrete_hidden = torch.tanh(self.qdiscrete_W1(doc_qdiscrete_mapped))
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
                -1, self.args.prev_q_limit, self.rating_emb_size)
            doc_pop_emb = self.popularity_encoder(pop_seq_emb, doc_mask)
            doc_pop_emb = doc_pop_emb.view(
                batch_size, doc_count, self.rating_emb_size)
            #doc_ddiscrete_hidden = torch.tanh(self.ddiscrete_W1(doc_ddiscrete_mapped))
            doc_ddiscrete_mapped = torch.cat([doc_ddiscrete_mapped, doc_pop_emb], dim=-1)
        doc_ddiscrete_hidden = torch.tanh(self.ddiscrete_W1(doc_ddiscrete_mapped))

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

        self.query_encoder.initialize_parameters(logger)
        self.transformer_encoder.initialize_parameters(logger)
        nn.init.xavier_normal_(self.ddiscrete_W1.weight)
        nn.init.constant_(self.ddiscrete_W1.bias, 0)
        nn.init.xavier_normal_(self.contextq_all_feat_W.weight)
        nn.init.constant_(self.contextq_all_feat_W.bias, 0)

        if logger:
            logger.info("ContextEmailRanker initialization finished.")
