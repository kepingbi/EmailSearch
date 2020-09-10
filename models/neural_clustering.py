''' model for personalized email search
'''
import torch
import torch.nn as nn
from others.logging import logger

class ClusteringModel(nn.Module):
    """
    https://www.dlology.com/blog/how-to-do-unsupervised-clustering-with-keras/
    Refer to https://github.com/Tony607/Keras_Deep_Clustering/blob/master/Keras-DEC.ipynb
    Clustering model that optimize the representation towards the clustering loss
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: degrees of freedom parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """
    def __init__(self, n_clusters, input_dim, device="cpu",center_weights=None, alpha=1.0):
        super(ClusteringModel, self).__init__()
        if center_weights is None:
            self.cluster_centers = torch.nn.Parameter(torch.rand(n_clusters, input_dim), requires_grad=True)
            nn.init.xavier_normal_(self.cluster_centers)
        else:
            self.set_center_weights(center_weights)
        # use the center collected from KMeans as the initial center
        self.embedding_size = input_dim
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.device = device
        self.KL_Div = nn.KLDivLoss()

    def set_center_weights(self, center_weights):
        center_weights = torch.tensor(center_weights, device=self.device)
        self.cluster_centers = torch.nn.Parameter(center_weights, requires_grad=True)

    def forward(self, input_vecs, target):
        """ input_vecs: batch_size, input_dim
            target: batch_size, n_clusters
        """
        # cut position -1 to 0
        # batch_size, max_doc_count
        q = self.soft_t_distribution(input_vecs)
        loss = self.KL_Div(q.log(), target)
        return loss.mean()

    def soft_t_distribution(self, input_vecs):
        """ student t-distribution, as same as used in t-SNE algorithm.
         Measure the similarity between embedded point z_i and centroid µ_j.
                 q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
                 q_ij can be interpreted as the probability of assigning sample i to cluster j.
                 (i.e., a soft assignment)
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        n_samples can be the size of a batch or all the available samples in the training set
        """
        q = 1.0 + (input_vecs.unsqueeze(1) - self.cluster_centers.unsqueeze(0)).square().sum(-1) / self.alpha
        q = q.pow(-(self.alpha + 1.0)/2.0) # n_samples, n_clusters
        q = q / q.sum(-1, keepdim=True)
        return q

    def target_distribution(self, q):
        """ q: n_samples, n_clusters
            return: n_samples, n_clusters
            n_samples should be all the available samples in the training set
        """
        q = q.square() / q.sum(0, keepdim=True)
        p = q / q.sum(-1, keepdim=True)
        return p
