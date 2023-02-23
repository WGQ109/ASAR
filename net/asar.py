import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlight import import_class
import numpy as np
# from fast_pytorch_kmeans import KMeans
from kmeans_pytorch import kmeans
import math
import torch
from time import time
import numpy as np

import math
import torch
from time import time
import numpy as np

'''
fast_pytorch_kmeans
'''

class KMeans:

    '''
    Referring to the code of fast_pytorch_kmeans


    Kmeans clustering algorithm implemented with PyTorch

    Parameters:
      n_clusters: int,
        Number of clusters

      max_iter: int, default: 100
        Maximum number of iterations

      tol: float, default: 0.0001
        Tolerance

      verbose: int, default: 0
        Verbosity

      mode: {'euclidean', 'cosine'}, default: 'euclidean'
        Type of distance measure

      minibatch: {None, int}, default: None
        Batch size of MinibatchKmeans algorithm
        if None perform full KMeans algorithm

    Attributes:
      centroids: torch.Tensor, shape: [n_clusters, n_features]
        cluster centroids
    '''

    def __init__(self, n_clusters, max_iter=100, tol=0.0001, verbose=0, mode="euclidean", minibatch=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.mode = mode
        self.minibatch = minibatch
        self._loop = False
        self._show = False

        try:
            import PYNVML
            self._pynvml_exist = True
        except ModuleNotFoundError:
            self._pynvml_exist = False

        self.centroids = None
        self.num_points_in_clusters = None

    @staticmethod
    def cos_sim(a, b):
        """
          Compute cosine similarity of 2 sets of vectors

          Parameters:
          a: torch.Tensor, shape: [m, n_features]

          b: torch.Tensor, shape: [n, n_features]
        """
        a_norm = a.norm(dim=-1, keepdim=True)
        b_norm = b.norm(dim=-1, keepdim=True)
        a = a / (a_norm + 1e-8)
        b = b / (b_norm + 1e-8)
        return a @ b.transpose(-2, -1)

    @staticmethod
    def euc_sim(a, b):
        """
          Compute euclidean similarity of 2 sets of vectors

          Parameters:
          a: torch.Tensor, shape: [m, n_features]

          b: torch.Tensor, shape: [n, n_features]
        """
        return 2 * a @ b.transpose(-2, -1) - (a ** 2).sum(dim=1)[..., :, None] - (b ** 2).sum(dim=1)[..., None, :]

    def remaining_memory(self):
        """
          Get remaining memory in gpu
        """
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if self._pynvml_exist:
            pynvml.nvmlInit()
            gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            remaining = info.free
        else:
            remaining = torch.cuda.memory_allocated()
        return remaining

    def max_sim(self, a, b):
        """
          Compute maximum similarity (or minimum distance) of each vector
          in a with all of the vectors in b

          Parameters:
          a: torch.Tensor, shape: [m, n_features]

          b: torch.Tensor, shape: [n, n_features]
        """
        device = a.device.type
        batch_size = a.shape[0]
        if self.mode == 'cosine':
            sim_func = self.cos_sim
        elif self.mode == 'euclidean':
            sim_func = self.euc_sim

        if device == 'cpu':
            sim = sim_func(a, b)
            max_sim_v, max_sim_i = sim.max(dim=-1)
            return max_sim_v, max_sim_i
        else:
            if a.dtype == torch.float:
                expected = a.shape[0] * a.shape[1] * b.shape[0] * 4
            elif a.dtype == torch.half:
                expected = a.shape[0] * a.shape[1] * b.shape[0] * 2
            ratio = math.ceil(expected / self.remaining_memory())
            subbatch_size = math.ceil(batch_size / ratio)
            msv, msi = [], []
            for i in range(ratio):
                if i * subbatch_size >= batch_size:
                    continue
                sub_x = a[i * subbatch_size: (i + 1) * subbatch_size]
                sub_sim = sim_func(sub_x, b)

                sub_max_sim_v, sub_max_sim_i = sub_sim.max(dim=-1)
                del sub_sim
                msv.append(sub_max_sim_v)
                msi.append(sub_max_sim_i)
            if ratio == 1:
                max_sim_v, max_sim_i = msv[0], msi[0]
            else:
                max_sim_v = torch.cat(msv, dim=0)
                max_sim_i = torch.cat(msi, dim=0)
            return max_sim_v, max_sim_i

    def fit_predict(self, X, centroids=None):
        """
          Combination of fit() and predict() methods.
          This is faster than calling fit() and predict() seperately.

          Parameters:
          X: torch.Tensor, shape: [n_samples, n_features]

          centroids: {torch.Tensor, None}, default: None
            if given, centroids will be initialized with given tensor
            if None, centroids will be randomly chosen from X

          Return:
          labels: torch.Tensor, shape: [n_samples]
        """
        batch_size, emb_dim = X.shape
        device = X.device.type
        start_time = time()
        if centroids is None:
            self.centroids = X[np.random.choice(batch_size, size=[self.n_clusters], replace=False)]
        else:
            self.centroids = centroids

        if self.num_points_in_clusters is None:
            self.num_points_in_clusters = torch.ones(self.n_clusters, device=device)
        closest = None
        for i in range(self.max_iter):

            if self.minibatch is not None:
                x = X[np.random.choice(batch_size, size=[self.minibatch], replace=False)]
            else:
                x = X

            closest = self.max_sim(a=x, b=self.centroids)[1]


            matched_clusters, counts = closest.unique(return_counts=True)


            c_grad = torch.zeros_like(self.centroids)
            if self._loop:
                for j, count in zip(matched_clusters, counts):
                    c_grad[j] = x[closest == j].sum(dim=0) / count
            else:
                if self.minibatch is None:
                    expanded_closest = closest[None].expand(self.n_clusters, -1)
                    mask = (expanded_closest == torch.arange(self.n_clusters, device=device)[:, None]).float()
                    c_grad = mask @ x / mask.sum(-1)[..., :, None]
                    c_grad[c_grad != c_grad] = 0  # remove NaNs
                else:
                    expanded_closest = closest[None].expand(len(matched_clusters), -1)
                    mask = (expanded_closest == matched_clusters[:, None]).float()
                    c_grad[matched_clusters] = mask @ x / mask.sum(-1)[..., :, None]

                # if x.dtype == torch.float:
                #   expected = closest.numel() * len(matched_clusters) * 5 # bool+float
                # elif x.dtype == torch.half:
                #   expected = closest.numel() * len(matched_clusters) * 3 # bool+half
                # if device == 'cpu':
                #   ratio = 1
                # else:
                #   ratio = math.ceil(expected / self.remaining_memory() )
                # # ratio = 1
                # subbatch_size = math.ceil(len(matched_clusters)/ratio)
                # for j in range(ratio):
                #   if j*subbatch_size >= batch_size:
                #     continue
                #   sub_matched_clusters = matched_clusters[j*subbatch_size: (j+1)*subbatch_size]
                #   sub_expanded_closest = closest[None].expand(len(sub_matched_clusters), -1)
                #   sub_mask = (sub_expanded_closest==sub_matched_clusters[:, None]).to(x.dtype)
                #   sub_prod = sub_mask @ x / sub_mask.sum(1)[:, None]
                #   c_grad[sub_matched_clusters] = sub_prod
            error = (c_grad - self.centroids).pow(2).sum()

            if error <= self.tol:
                break


            if self.minibatch is not None:
                lr = 1 / self.num_points_in_clusters[:, None] * 0.9 + 0.1
                # lr = 1/self.num_points_in_clusters[:,None]**0.1
            else:
                lr = 1
            self.num_points_in_clusters[matched_clusters] += counts
            self.centroids = self.centroids * (1 - lr) + c_grad * lr




        return closest,matched_clusters,len(matched_clusters), counts

    def predict(self, X):
        """
          Predict the closest cluster each sample in X belongs to

          Parameters:
          X: torch.Tensor, shape: [n_samples, n_features]

          Return:
          labels: torch.Tensor, shape: [n_samples]
        """
        return self.max_sim(a=X, b=self.centroids)[1]

    def fit(self, X, centroids=None):
        """
          Perform kmeans clustering

          Parameters:
          X: torch.Tensor, shape: [n_samples, n_features]
        """
        self.fit_predict(X, centroids)


class ASAR(nn.Module):
    """ Referring to the code of MOCO, https://arxiv.org/abs/1911.05722 """

    def __init__(self, base_encoder=None, pretrain=True, feature_dim=128, queue_size=32768,
                 momentum=0.999, Temperature=0.07, mlp=True, in_channels=3, hidden_channels=64,
                 hidden_dim=256, num_class=60, dropout=0.5,
                 graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                 edge_importance_weighting=True, **kwargs):
        """
        K: queue size; number of negative keys (default: 32768)
        m: momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """

        super().__init__()
        base_encoder = import_class(base_encoder)
        self.pretrain = pretrain

        if not self.pretrain:
            self.encoder_q = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=num_class,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
        else:
            self.K = queue_size
            self.m = momentum
            self.T = Temperature

            self.T1 = 0.07
            self.T2 = 0.07


            self.nn_num = 5

            self.max_entropy = np.log(self.K)


            self.encoder_q = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=feature_dim,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
            self.encoder_k = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=feature_dim,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)

            if mlp:  # hack: brute-force replacement
                dim_mlp = self.encoder_q.fc.weight.shape[1]
                self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                  nn.ReLU(),
                                                  self.encoder_q.fc)
                self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                  nn.ReLU(),
                                                  self.encoder_k.fc)

            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)    # initialize
                param_k.requires_grad = False       # not update by gradient

            # create the queue
            K = queue_size
            self.register_buffer("queue", torch.randn(128, K))
            self.queue = nn.functional.normalize(self.queue, dim=0)
            self.register_buffer('labels', -1 * torch.ones(self.K).long())
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
        #     param_k.data.copy_(param_q.data)  # initialize
        #     param_k.requires_grad = False  # not update by gradient

            self.max_entropy = np.log(self.K)
            self.criterion = nn.CrossEntropyLoss()
            self.w = 0


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()  # transpose

        self.w = ptr

        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).cuda()

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]

    # @torch.no_grad()
    # def _get_dequeue(self, bs):
    #     batch_size = bs
    #     return self.queue[:, self.w:self.w+batch_size]

    @torch.no_grad()
    def _get_dequeue(self, bs):
        batch_size = bs

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        return self.queue[:, ptr:ptr + batch_size]


    def forward(self, im_q, im_k, epoch):
        if not self.pretrain:
            return self.encoder_q(im_q)
        # update the key encoder
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)  # already normalized

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            im_k_, idx_unshufflek = self._batch_shuffle_single_gpu(im_k)
            k = self.encoder_k(im_k_)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)  # already normalized
            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshufflek)

            # pseudo logits: NxK
            logits_pd = torch.einsum('nc,ck->nk', [k, self.queue.clone().detach()])
            logits_pd /= self.T2

            labels = torch.zeros(logits_pd.size(0), logits_pd.size(1)+1).cuda()

            labels[:, 0] = 1.0
            _, nn_index = logits_pd.topk(self.nn_num, dim=1, largest=True)
            hard_labels = torch.zeros_like(logits_pd, device=logits_pd.device).scatter(1, nn_index, 1)
            pseudo_labels = F.softmax(logits_pd, 1)
            log_pseudo_labels = F.log_softmax(logits_pd, 1)
            entropy = -torch.sum(pseudo_labels * log_pseudo_labels, dim=1, keepdim=True)
            c = 1 - entropy / self.max_entropy
            labels[:, 1:] = hard_labels * c



        # label normalization
        labels = labels / labels.sum(dim=1, keepdim=True)

        # forward pass
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T1

        loss = -torch.sum(labels.detach() * F.log_softmax(logits, 1), 1).mean()

        if epoch > 150:

            q1 = q.clone()

            k2 = k.clone()
            out1 = self._get_dequeue(k2.shape[0])
            k1 = torch.cat([k2, out1], dim=0)
            k1.requires_grad = True
            k1.retain_grad()
            bs = q1.size(0)
            target = torch.arange(bs).cuda()
            sim = torch.mm(q1, k1.t())
            loss_act = self.criterion(sim,target)
            loss_act = loss_act.mean()
            loss_act.backward(retain_graph=True)


            kmeans = KMeans(n_clusters=bs, mode='euclidean', verbose=0)
            label,out,lenout,count = kmeans.fit_predict(k1.grad)


            ls1 = []
            ls2 = []
            ls3 = []
            ls4 = []

            for j, value in enumerate(label):

                if count[torch.nonzero(out==value.item()).squeeze()] == 1:
                    ls1.append(j)
                elif count[torch.nonzero(out==value.item()).squeeze()] == 2:
                    ls2.append(j)
                    count[torch.nonzero(out==value.item()).squeeze()] = count[torch.nonzero(out==value.item()).squeeze()] - 1
                elif count[torch.nonzero(out==value.item()).squeeze()] == 3:
                    ls3.append(j)
                    count[torch.nonzero(out==value.item()).squeeze()] = count[torch.nonzero(out==value.item()).squeeze()] - 1
                else:
                    ls4.append(j)
                    count[torch.nonzero(out==value.item()).squeeze()] = count[torch.nonzero(out==value.item()).squeeze()] - 1

            len1 = len(ls1)
            len2 = len(ls2)
            len3 = len(ls3)

            if len1 < bs:
                if len1 + len2 >= bs:
                    ls_temp = ls1 + ls2
                    ls = ls_temp[0:bs]
                elif len1 + len2 + len3 >= bs:
                    ls_temp = ls1 + ls2 + ls3
                    ls = ls_temp[0:bs]
                else:
                    ls = ls1 + ls2 + ls3 + ls4
            else:
                ls = ls1

            x = k1[ls]


            self._dequeue_and_enqueue(x)
        else:
            self._dequeue_and_enqueue(k)

        return loss
