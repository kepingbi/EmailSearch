""" query encoder can be the same as HEM: (fs)
"""
import torch
import torch.nn as nn
from models.rnn import LayerNormLSTM

def get_vector_mean(inputs, input_mask):
    ''' get the mean vector of the input.
    '''
    #batch_size, max_word_count, embedding_size
    inputs_sum = (inputs * input_mask.float().unsqueeze(-1)).sum(1)
    word_count = input_mask.sum(-1)

    word_count = word_count.masked_fill(
        word_count.eq(0), 1).unsqueeze(-1)

    inputs_mean = inputs_sum / word_count.float()

    return inputs_mean


class FSEncoder(nn.Module):
    def __init__(self, embedding_size, dropout=0.0):
        super(FSEncoder, self).__init__()
        self.dropout_ = dropout
        self.output_size_ = embedding_size
        self.f_W = nn.Linear(embedding_size, embedding_size)
        self.drop_layer = nn.Dropout(p=self.dropout_)
        #by default bias=True

    @property
    def size(self):
        return self.output_size_

    def forward(self, inputs, input_mask):
        #batch_size, max_word_count, embedding_size
        inputs_mean = get_vector_mean(inputs, input_mask)
        inputs_mean = torch.dropout(
            inputs_mean, p=self.dropout_, train=self.training)
        #inputs_mean = self.drop_layer(inputs_mean)

        f_s = torch.tanh(self.f_W(inputs_mean))
        return f_s

    def initialize_parameters(self, logger=None):
        if logger:
            logger.info(" FSEncoder initialization started.")
        for name, p in self.named_parameters():
            if "weight" in name:
                if logger:
                    logger.info(" {} ({}): Xavier normal init.".format(
                        name, ",".join([str(x) for x in p.size()])))
                nn.init.xavier_normal_(p)
            elif "bias" in name:
                if logger:
                    logger.info(" {} ({}): constant (0) init.".format(
                        name, ",".join([str(x) for x in p.size()])))
                nn.init.constant_(p, 0)
            else:
                if logger:
                    logger.info(" {} ({}): random normal init.".format(
                        name, ",".join([str(x) for x in p.size()])))
                nn.init.normal_(p)
        if logger:
            logger.info(" FSEncoder initialization finished.")

class AVGEncoder(nn.Module):
    def __init__(self, embedding_size, dropout=0.0):
        super(AVGEncoder, self).__init__()
        self.dropout_ = dropout
        self.output_size_ = embedding_size
        self.drop_layer = nn.Dropout(p=self.dropout_)

    @property
    def size(self):
        return self.output_size_

    def forward(self, inputs, input_mask):
        #batch_size, max_word_count, embedding_size
        inputs_mean = get_vector_mean(inputs, input_mask)
        #inputs_mean = torch.dropout(
        #    inputs_mean, p=self.dropout_, train=self.training)
        inputs_mean = self.drop_layer(inputs_mean) #better managed than using torch.dropout

        return inputs_mean

    def initialize_parameters(self, logger=None):
        if logger:
            logger.info(" AveragingEncoder initialization skipped"
                        " (no parameters).")

#CNN or RNN


class RNNEncoder(nn.Module):

    def __init__(self, bidirectional, num_layers, input_size,
                 hidden_size, dropout=0.0):
        super(RNNEncoder, self).__init__()
        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions

        self.rnn = LayerNormLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional)

    def forward(self, x, mask):
        """See :func:`EncoderBase.forward()`"""
        # batch_size, sequence_len, embedding_size
        x = torch.transpose(x, 1, 0)
        # sequence_len, batch_size, embedding_size
        memory_bank, _ = self.rnn(x)
        # memory_bank: (sequence_len, batch_size, num_directions * hidden_size)
        # memory_bank = self.dropout(memory_bank) + x
        # memory_bank = torch.transpose(memory_bank, 1, 0)
        final_state = memory_bank[-1]

        return final_state # batch_size, hidden_size

    def initialize_parameters(self, logger=None):
        if logger:
            logger.info(" RNNEncoder initialization started.")
        for name, p in self.named_parameters():
            #print(name)
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
            else:
                nn.init.normal_(p)
        if logger:
            logger.info(" RNNEncoder initialization finished.")
