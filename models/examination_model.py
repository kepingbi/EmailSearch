''' model for personalized email search
'''
import torch
import torch.nn as nn
from others.logging import logger

class ExaminationModel(nn.Module):
    ''' examination model that learns the propensity of examing each position
    '''
    g_max_relevance_pos = 10
    g_max_datetime_pos = 50

    def __init__(self, embedding_size, dropout=0.0):
        super(ExaminationModel, self).__init__()
        self.embedding_size = embedding_size // 2
        # response_requested, importance, is_read, flag_status, email_class
        # conversation_hash, subject_prefix_hash
        # embeddings for query discrete features
        self.relevance_pos_emb = nn.Embedding(
            self.g_max_relevance_pos + 1,
            self.embedding_size,
            padding_idx=0)
        self.datetime_pos_emb = nn.Embedding(
            self.g_max_datetime_pos + 1,
            self.embedding_size,
            padding_idx=0)
        self.mlp_layer = nn.Linear(2 * self.embedding_size, self.embedding_size)
        self.final_layer = nn.Linear(self.embedding_size, 1)
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, batch_rel_pos, batch_time_pos):
        """ batch_rel_pos: batch_size, max_doc_count(50) relevance position
            batch_time_pos: batch_size, max_doc_count(50) datetime position
        """
        # cut position -1 to 0
        batch_rel_pos = torch.clamp(batch_rel_pos, min=0, max=self.g_max_relevance_pos)
        batch_time_pos = torch.clamp(batch_time_pos, min=0, max=self.g_max_datetime_pos)
        doc_mask = batch_rel_pos.ne(0) | batch_time_pos.ne(0)
        # documents that are shown in either window
        hidden0 = torch.cat([self.relevance_pos_emb(batch_rel_pos), \
            self.datetime_pos_emb(batch_time_pos)], dim=-1)
        # batch_size, max_doc_count, 2*embedding_size
        hidden1 = torch.tanh(self.mlp_layer(hidden0))
        hidden1 = self.dropout_layer(hidden1)
        final_output = torch.tanh(self.final_layer(hidden1))
        final_output = final_output.squeeze(-1) * doc_mask.float()
        # batch_size, max_doc_count
        return final_output

    def initialize_parameters(self, logger=None):
        if logger:
            logger.info("ExaminationModel initialization started.")
        # embeddings for query discrete features
        nn.init.normal_(self.relevance_pos_emb.weight)
        nn.init.normal_(self.datetime_pos_emb.weight)

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
        if logger:
            logger.info("ExaminationModel initialization finished.")
