''' Convert python list to cpu and gpu tensors when needed
'''
import torch


class DocContextBatch():
    ''' Contextual positive documents and queries.
    '''
    def __init__(self, batch_qids, batch_user_idxs, candi_doc_idxs, candi_doc_ratings,
                 candi_doc_qcont_features, candi_doc_qdiscrete_features,
                 candi_doc_dcont_features, candi_doc_ddiscrete_features,
                 candi_doc_qdcont_features, candi_doc_popularity,
                 candi_conv_occur, candi_doc_occur,
                 context_qidxs, context_pos_didxs,
                 context_qcont_features, context_qdiscrete_features,
                 context_pos_dcont_features, context_pos_ddiscrete_features,
                 context_pos_qdcont_features, context_d_popularity,
                 to_tensor=True): #"cpu" or "cuda"
        self.query_idxs = batch_qids
        self.user_idxs = batch_user_idxs
        self.candi_doc_idxs = candi_doc_idxs
        self.candi_doc_ratings = candi_doc_ratings
        self.candi_doc_qcont_features = candi_doc_qcont_features
        self.candi_doc_qdiscrete_features = candi_doc_qdiscrete_features
        self.candi_doc_dcont_features = candi_doc_dcont_features
        self.candi_doc_ddiscrete_features = candi_doc_ddiscrete_features
        self.candi_doc_qdcont_features = candi_doc_qdcont_features
        self.candi_doc_popularity = candi_doc_popularity
        self.candi_conv_occur = candi_conv_occur
        self.candi_doc_occur = candi_doc_occur
        self.context_qidxs = context_qidxs
        self.context_pos_didxs = context_pos_didxs
        self.context_qcont_features = context_qcont_features
        self.context_qdiscrete_features = context_qdiscrete_features
        self.context_pos_dcont_features = context_pos_dcont_features
        self.context_pos_ddiscrete_features = context_pos_ddiscrete_features
        self.context_pos_qdcont_features = context_pos_qdcont_features
        self.context_d_popularity = context_d_popularity

        if to_tensor:
            self.to_tensor()

    def to_tensor(self):
        self.query_idxs = torch.tensor(self.query_idxs)
        self.user_idxs = torch.tensor(self.user_idxs)
        self.candi_doc_idxs = torch.tensor(self.candi_doc_idxs)
        self.candi_doc_ratings = torch.tensor(self.candi_doc_ratings)
        self.candi_doc_qcont_features = torch.tensor(self.candi_doc_qcont_features)
        self.candi_doc_qdiscrete_features = torch.tensor(self.candi_doc_qdiscrete_features)
        self.candi_doc_dcont_features = torch.tensor(self.candi_doc_dcont_features)
        self.candi_doc_ddiscrete_features = torch.tensor(self.candi_doc_ddiscrete_features)
        self.candi_doc_qdcont_features = torch.tensor(self.candi_doc_qdcont_features)
        self.candi_doc_popularity = torch.tensor(self.candi_doc_popularity)
        self.candi_conv_occur = torch.tensor(self.candi_conv_occur)
        self.candi_doc_occur = torch.tensor(self.candi_doc_occur)

        self.context_qidxs = torch.tensor(self.context_qidxs)
        self.context_pos_didxs = torch.tensor(self.context_pos_didxs)
        self.context_qcont_features = torch.tensor(self.context_qcont_features)
        self.context_qdiscrete_features = torch.tensor(self.context_qdiscrete_features)
        self.context_pos_dcont_features = torch.tensor(self.context_pos_dcont_features)
        self.context_pos_ddiscrete_features = torch.tensor(self.context_pos_ddiscrete_features)
        self.context_pos_qdcont_features = torch.tensor(self.context_pos_qdcont_features)
        self.context_d_popularity = torch.tensor(self.context_d_popularity)

    def to(self, device):
        if device == "cpu":
            return self
        else:
            # query and user idxs not used during training or inference
            candi_doc_idxs = self.candi_doc_idxs.to(device)
            candi_doc_ratings = self.candi_doc_ratings.to(device)
            candi_doc_qcont_features = self.candi_doc_qcont_features.to(device).float()
            candi_doc_qdiscrete_features = self.candi_doc_qdiscrete_features.to(device)
            candi_doc_dcont_features = self.candi_doc_dcont_features.to(device).float()
            candi_doc_ddiscrete_features = self.candi_doc_ddiscrete_features.to(device)
            candi_doc_qdcont_features = self.candi_doc_qdcont_features.to(device).float()
            candi_doc_popularity = self.candi_doc_popularity.to(device)
            candi_conv_occur = self.candi_conv_occur.to(device)
            candi_doc_occur = self.candi_doc_occur.to(device)
            context_qidxs = self.context_qidxs.to(device)
            context_pos_didxs = self.context_pos_didxs.to(device)
            context_qcont_features = self.context_qcont_features.to(device).float()
            context_qdiscrete_features = self.context_qdiscrete_features.to(device)
            context_pos_dcont_features = self.context_pos_dcont_features.to(device).float()
            context_pos_ddiscrete_features = self.context_pos_ddiscrete_features.to(device)
            context_pos_qdcont_features = self.context_pos_qdcont_features.to(device).float()
            context_d_popularity = self.context_d_popularity.to(device)

            return self.__class__(
                self.query_idxs, self.user_idxs,
                candi_doc_idxs, candi_doc_ratings,
                candi_doc_qcont_features, candi_doc_qdiscrete_features,
                candi_doc_dcont_features, candi_doc_ddiscrete_features,
                candi_doc_qdcont_features, candi_doc_popularity,
                candi_conv_occur, candi_doc_occur,
                context_qidxs, context_pos_didxs,
                context_qcont_features, context_qdiscrete_features,
                context_pos_dcont_features, context_pos_ddiscrete_features,
                context_pos_qdcont_features, context_d_popularity, to_tensor=False)


class DocBaselineBatch(object):
    ''' Contextual positive documents and queries.
    '''
    def __init__(self, batch_qids, batch_user_idxs, candi_doc_idxs, candi_doc_ratings,
                 candi_doc_qcont_features, candi_doc_qdiscrete_features,
                 candi_doc_dcont_features, candi_doc_ddiscrete_features,
                 candi_doc_qdcont_features,
                 to_tensor=True): #"cpu" or "cuda"
        self.query_idxs = batch_qids
        self.user_idxs = batch_user_idxs
        self.candi_doc_idxs = candi_doc_idxs
        self.candi_doc_ratings = candi_doc_ratings
        self.candi_doc_qcont_features = candi_doc_qcont_features
        self.candi_doc_qdiscrete_features = candi_doc_qdiscrete_features
        self.candi_doc_dcont_features = candi_doc_dcont_features
        self.candi_doc_ddiscrete_features = candi_doc_ddiscrete_features
        self.candi_doc_qdcont_features = candi_doc_qdcont_features

        if to_tensor:
            self.to_tensor()

    def to_tensor(self):
        self.query_idxs = torch.tensor(self.query_idxs)
        self.user_idxs = torch.tensor(self.user_idxs)
        self.candi_doc_idxs = torch.tensor(self.candi_doc_idxs)
        self.candi_doc_ratings = torch.tensor(self.candi_doc_ratings)
        self.candi_doc_qcont_features = torch.tensor(self.candi_doc_qcont_features).float()
        self.candi_doc_qdiscrete_features = torch.tensor(self.candi_doc_qdiscrete_features)
        self.candi_doc_dcont_features = torch.tensor(self.candi_doc_dcont_features).float()
        self.candi_doc_ddiscrete_features = torch.tensor(self.candi_doc_ddiscrete_features)
        self.candi_doc_qdcont_features = torch.tensor(self.candi_doc_qdcont_features).float()

    def to(self, device):
        if device == "cpu":
            return self
        else:
            candi_doc_idxs = self.candi_doc_idxs.to(device)
            candi_doc_ratings = self.candi_doc_ratings.to(device)
            candi_doc_qcont_features = self.candi_doc_qcont_features.to(device)
            candi_doc_qdiscrete_features = self.candi_doc_qdiscrete_features.to(device)
            candi_doc_dcont_features = self.candi_doc_dcont_features.to(device)
            candi_doc_ddiscrete_features = self.candi_doc_ddiscrete_features.to(device)
            candi_doc_qdcont_features = self.candi_doc_qdcont_features.to(device)

            return self.__class__(
                self.query_idxs, self.user_idxs,
                candi_doc_idxs, candi_doc_ratings,
                candi_doc_qcont_features, candi_doc_qdiscrete_features,
                candi_doc_dcont_features, candi_doc_ddiscrete_features,
                candi_doc_qdcont_features, to_tensor=False)
