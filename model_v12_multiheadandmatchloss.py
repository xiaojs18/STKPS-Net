import torch
import torch.nn as nn
from collections import OrderedDict
from utils import split_first_dim_linear
import math
from itertools import combinations 
from adafocus_v12 import AdaFocus
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.models as models

NUM_SAMPLES=1


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000, pe_scale_factor=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe_scale_factor = pe_scale_factor
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) * self.pe_scale_factor
        pe[:, 1::2] = torch.cos(position * div_term) * self.pe_scale_factor
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class DistanceLoss(nn.Module):
    "Compute the Query-class similarity on the patch-enriched features."

    def __init__(self, args, mode):
        super(DistanceLoss, self).__init__()

        self.args = args
        # self.temporal_set_size = temporal_set_size

        max_len = int(self.args.seq_len * 1.5)
        self.dropout = nn.Dropout(p=0.1)

        # generate all ordered tuples corresponding to the temporal set size 2 or 3.
        # frame_idxs = [i for i in range(self.args.seq_len)]
        # frame_combinations = combinations(frame_idxs, temporal_set_size)
        # self.tuples = [torch.tensor(comb).cuda() for comb in frame_combinations]
        # self.tuples_len = len(self.tuples) # 28 for tempset_2

        # nn.Linear(4096, 1024)
        # self.clsW = nn.Linear(self.args.trans_linear_in_dim * self.temporal_set_size, self.args.trans_linear_in_dim//2)
        # if(mode=='all'):
        # self.clsW = nn.Linear(self.args.trans_linear_in_dim, self.args.trans_linear_in_dim//2)
        # else:
        # self.clsW = nn.Linear(self.args.trans_linear_in_dim//2, self.args.trans_linear_in_dim//4)
        # self.relu = torch.nn.ReLU()

    # def forward(self, support_set, support_labels, queries):
    def forward(self, support_set_global, support_set_local, support_labels, queries_global, queries_local):
        # support_set : 25 x 8 x 2048, support_labels: 25, queries: 20 x 8 x 2048
        device = support_set_global.device
        support_set = torch.cat((support_set_global, support_set_local), 1)
        queries = torch.cat((queries_global, queries_local), 1)
        n_queries = queries.shape[0]  # 20
        # n_support = support_set.shape[0]//self.args.way#5
        n_support = support_set.shape[0]  # 25
        frame_num = support_set.shape[1]  # 8
        linear_in_dim = support_set.shape[2]  # 2048
        # lengt=support_set.shape[2] #2048
        # support_set=support_set.view(self.args.way,self.args.shot,frames,lengt)

        # Add a dropout before creating tuples
        support_set = self.dropout(support_set)  # 25 x 8 x 2048
        queries = self.dropout(queries)  # 20 x 8 x 2048

        # construct new queries and support set made of tuples of images after pe
        # Support set s = number of tuples(28 for 2/56 for 3) stacked in a list form containing elements of form 25 x 4096(2 x 2048 - (2 frames stacked)) 28=C82
        # s = [torch.index_select(support_set, -2, p).reshape(n_support, -1) for p in self.tuples]
        # q = [torch.index_select(queries, -2, p).reshape(n_queries, -1) for p in self.tuples]

        # support_set = torch.stack(s, dim=-2).to(device) # 25 x 28 x 4096
        # queries = torch.stack(q, dim=-2) # 20 x 28 x 4096
        # support_set = support_set.cuda()
        # queries = queries.cuda()
        # support_labels = support_labels.cuda()
        unique_labels = torch.unique(support_labels)  # 5
        # print(support_labels)
        # print(unique_labels)

        # target_set = self.clsW(queries.view(-1, linear_in_dim)) # 160[20x8] x 1024
        target_set = queries.view(-1, linear_in_dim)  # 160[20x8] x 1024
        # Add relu after clsW
        # target_set = self.relu(target_set)
        '''
        dist_all = torch.zeros(n_queries, self.args.way).cuda() # 20 x 5

        for label_idx, c in enumerate(unique_labels):
            # Select keys corresponding to this class from the support set tuples
            class_k = torch.index_select(support_set, 0, self._extract_class_indices(support_labels, c)) # 5 x 8 x 2048

            # Reshaping the selected keys
            class_k = class_k.view(-1, linear_in_dim) # 40 x 2048

            # Get the support set projection from the current class
            support_embed = self.clsW(class_k.to(queries.device))  # 40[5 x 8] x1024

            # Add relu after clsW
            support_embed = self.relu(support_embed) # 40 x 1024

            # Calculate p-norm distance between the query embedding and the support set embedding
            #distmat = torch.cdist(query_embed, support_embed) # 560[20 x 28] x 140[28 x 5]
            frame_sim = torch.matmul(F.normalize(target_set, dim=1), F.normalize(support_embed, dim=1).permute(1,0)).view(n_queries, frame_num, n_support, frame_num).transpose(2,1).contiguous()#好像没问题 20 x 5 x 8 x 8  
            dists = 1 - frame_sim


            cum_dists = dists.min(3)[0].sum(2) + dists.min(2)[0].sum(2)  #20*5
            # Across the 140 tuples compared against, get the minimum distance for each of the 560 queries


            # Average across the 28 tuples
            cum_dists = cum_dists.mean(dim=1)  # 20

            # Make it negative as this has to be reduced.
            distance = -1.0 * cum_dists
            c_idx = c.long()
            dist_all[:,c_idx] = distance # Insert into the required location.
        '''
        support_set = support_set.view(-1, linear_in_dim)  # 200[25x8] x 1024
        # support_set=self.clsW(support_set.view(-1, linear_in_dim)) # 200[25x8] x 1024
        # support_set=self.relu(support_set)
        # support_set=self.clsW(support_set)
        # support_set=self.relu(support_set)
        # target_set=self.clsW(target_set)
        # target_set=self.relu(target_set)
        frame_sim = torch.matmul(F.normalize(support_set, dim=1), F.normalize(target_set, dim=1).permute(1, 0)).view(
            n_support, frame_num, n_queries, frame_num).transpose(2, 1).contiguous()  # 好像没问题 25 x 20 x 8 x 8
        frame_dists = 1 - frame_sim
        # frame_sim = torch.matmul(support_set, target_set.permute(1,0)).view(n_support, frame_num, n_queries, frame_num).transpose(2,1).contiguous()#好像没问题 25 x 20 x 8 x 8
        # print(frame_sim)
        # frame_dists = torch.div(frame_sim, frame_num)
        frame_dists = torch.norm(frame_dists, dim=[-2, -1], keepdim=True) ** 2
        dists = frame_dists

        cum_dists = dists.min(3)[0].sum(2) + dists.min(2)[0].sum(2)  # 25 x 20
        # cum_dists = dists.min(3)[0].mean(2) + dists.min(2)[0].mean(2) #25 x 20
        # print(cum_dists)
        # print(cum_dists.size())
        # query_embed = self.clsW(queries.view(-1, self.args.trans_linear_in_dim*self.temporal_set_size)) # 560[20x28] x 1024

        # Add relu after clsW
        # query_embed = self.relu(query_embed) # 560 x 1024

        # init tensor to hold distances between every support tuple and every target tuple. It is of shape 20  x 5
        '''
            4-queries * 5 classes x 5(5 classes) and store this in a logit vector
        '''
        # class_dists = [
        #     torch.mean(torch.index_select(cum_dists, 0, self._extract_class_indicesthis(support_labels, c)), dim=0) for
        #     c in unique_labels]
        class_dists = [
            torch.mean(torch.index_select(cum_dists, 0, self._extract_class_indicesthis(support_labels, c).to(device)),
                       dim=0) for c in unique_labels]
        class_dists = torch.stack(class_dists)  # [5, 20]
        # print(class_dists.size())
        # class_dists = rearrange(class_dists, 'c q -> q c')
        class_dists = class_dists.permute(1, 0)  # [20,5] Hyrsm 对应公式（9）
        return_dict = {'logits': - class_dists}
        # return_dict = {'logits': dist_all}

        return return_dict

    @staticmethod
    def _extract_class_indicesthis(labels, which_class):
        """
        Helper method to extract the indices of elements which have the specified label.
        :param labels: (torch.tensor) Labels of the context set.
        :param which_class: Label for which indices are extracted.
        :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
        """
        # print(labels)
        # print(which_class)
        # class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        # class_mask_indices = torch.nonzero(class_mask, as_tuple=False)  # indices of labels equal to which class
        # return torch.reshape(class_mask_indices, (-1,)) # reshape to be a 1D vector
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector


class TemporalCrossTransformer_multihead(nn.Module):
    def __init__(self, args, mode, temporal_set_size=3):
        super(TemporalCrossTransformer_multihead, self).__init__()

        self.args = args
        self.temporal_set_size = temporal_set_size

        max_len = int(self.args.seq_len * 1.5)
        # if(mode=='all')
        # args.trans_linear_in_dim=4096

        self.pe = PositionalEncoding((self.args.trans_linear_in_dim), self.args.trans_dropout, max_len=max_len)
        self.k_linear = nn.Linear((self.args.trans_linear_in_dim) * temporal_set_size,
                                  (self.args.trans_linear_out_dim))  # .cuda()
        self.v_linear = nn.Linear((self.args.trans_linear_in_dim) * temporal_set_size,
                                  (self.args.trans_linear_out_dim))  # .cuda()
        self.k_linear_local = nn.Linear((self.args.trans_linear_in_dim) * temporal_set_size,
                                        (self.args.trans_linear_out_dim))  # .cuda()
        self.v_linear_local = nn.Linear((self.args.trans_linear_in_dim) * temporal_set_size,
                                        (self.args.trans_linear_out_dim))  # .cuda()
        self.norm_k = nn.LayerNorm((self.args.trans_linear_out_dim))
        self.norm_v = nn.LayerNorm((self.args.trans_linear_out_dim))

        self.class_softmax = torch.nn.Softmax(dim=1)

        # ===== 修复1: 不再直接使用.cuda()，而是注册为buffer =====
        # generate all tuples
        frame_idxs = [i for i in range(self.args.seq_len)]
        frame_combinations = combinations(frame_idxs, temporal_set_size)
        # 将tuples存储为CPU tensor，在forward时再移到正确的设备
        tuples_list = [torch.tensor(comb) for comb in frame_combinations]
        # 注册为buffer，这样它们会自动跟随模型移动到正确的设备
        for i, t in enumerate(tuples_list):
            self.register_buffer(f'tuple_{i}', t)
        self.tuples_len = len(tuples_list)

    def _get_tuples(self, device):
        """获取在正确设备上的tuples"""
        tuples = []
        for i in range(self.tuples_len):
            tuples.append(getattr(self, f'tuple_{i}').to(device))
        return tuples

    def forward(self, support_set, support_labels, queries):  # support_set=support_set_global queries=queries_global
        # ===== 修复2: 获取当前设备 =====
        device = support_set.device

        support_set, support_set_local = torch.split(support_set, [2048, 2048], dim=2)
        queries, queries_local = torch.split(queries, [2048, 2048], dim=2)
        n_queries = queries.shape[0]
        n_support = support_set.shape[0]

        # static pe
        support_set = self.pe(support_set)
        queries = self.pe(queries)
        support_set_local = self.pe(support_set_local)
        queries_local = self.pe(queries_local)

        # ===== 修复3: 使用正确设备上的tuples =====
        tuples = self._get_tuples(device)

        # construct new queries and support set made of tuples of images after pe
        s = [torch.index_select(support_set, -2, p).reshape(n_support, -1) for p in tuples]
        q = [torch.index_select(queries, -2, p).reshape(n_queries, -1) for p in tuples]
        support_set = torch.stack(s, dim=-2)
        queries = torch.stack(q, dim=-2)

        s_local = [torch.index_select(support_set_local, -2, p).reshape(n_support, -1) for p in tuples]
        q_local = [torch.index_select(queries_local, -2, p).reshape(n_queries, -1) for p in tuples]
        support_set_local = torch.stack(s_local, dim=-2)
        queries_local = torch.stack(q_local, dim=-2)

        # apply linear maps
        support_set_ks = self.k_linear(support_set)
        queries_ks = self.k_linear(queries)
        support_set_vs = self.v_linear(support_set)
        queries_vs = self.v_linear(queries)

        support_set_ks_local = self.k_linear_local(support_set_local)
        queries_ks_local = self.k_linear_local(queries_local)
        support_set_vs_local = self.v_linear_local(support_set_local)
        queries_vs_local = self.v_linear_local(queries_local)

        # apply norms where necessary
        mh_support_set_ks = self.norm_k(support_set_ks)
        mh_queries_ks = self.norm_k(queries_ks)
        mh_support_set_vs = support_set_vs
        mh_queries_vs = queries_vs

        mh_support_set_ks_local = self.norm_k(support_set_ks_local)
        mh_queries_ks_local = self.norm_k(queries_ks_local)
        mh_support_set_vs_local = support_set_vs_local
        mh_queries_vs_local = queries_vs_local

        mh_queries_vs_total = torch.cat((mh_queries_vs, mh_queries_vs_local), 2)

        unique_labels = torch.unique(support_labels)

        # ===== 修复4: 使用正确的设备创建张量 =====
        # init tensor to hold distances between every support tuple and every target tuple
        all_distances_tensor_total = torch.zeros(n_queries, self.args.way, device=device)
        all_distances_tensor = torch.zeros(n_queries, self.args.way, device=device)
        all_distances_tensor_local = torch.zeros(n_queries, self.args.way, device=device)

        for label_idx, c in enumerate(unique_labels):
            # ===== 修复5: 确保索引与张量在同一设备 =====
            # select keys and values for just this class
            indices = self._extract_class_indices(support_labels, c).to(device)
            class_k = torch.index_select(mh_support_set_ks, 0, indices)
            class_v = torch.index_select(mh_support_set_vs, 0, indices)
            class_k_local = torch.index_select(mh_support_set_ks_local, 0, indices)
            class_v_local = torch.index_select(mh_support_set_vs_local, 0, indices)
            k_bs = class_k.shape[0]

            class_scores = torch.matmul(mh_queries_ks.unsqueeze(1), class_k.transpose(-2, -1)) / math.sqrt(
                self.args.trans_linear_out_dim)
            class_scores_local = torch.matmul(mh_queries_ks_local.unsqueeze(1),
                                              class_k_local.transpose(-2, -1)) / math.sqrt(
                self.args.trans_linear_out_dim)

            # reshape etc. to apply a softmax for each query tuple
            class_scores = class_scores.permute(0, 2, 1, 3)
            class_scores = class_scores.reshape(n_queries, self.tuples_len, -1)
            class_scores = [self.class_softmax(class_scores[i]) for i in range(n_queries)]
            class_scores = torch.stack(class_scores)
            class_scores = class_scores.reshape(n_queries, self.tuples_len, -1, self.tuples_len)
            class_scores = class_scores.permute(0, 2, 1, 3)

            # get query specific class prototype
            query_prototype = torch.matmul(class_scores, class_v)
            query_prototype = torch.sum(query_prototype, dim=1)

            # calculate distances from queries to query-specific class prototypes
            diff = mh_queries_vs - query_prototype
            norm_sq = torch.norm(diff, dim=[-2, -1]) ** 2
            distance = torch.div(norm_sq, self.tuples_len)

            # multiply by -1 to get logits
            distance = distance * -1
            c_idx = c.long()
            all_distances_tensor[:, c_idx] = distance

            # reshape etc. to apply a softmax for each query tuple
            class_scores_local = class_scores_local.permute(0, 2, 1, 3)
            class_scores_local = class_scores_local.reshape(n_queries, self.tuples_len, -1)
            class_scores_local = [self.class_softmax(class_scores_local[i]) for i in range(n_queries)]
            class_scores_local = torch.stack(class_scores_local)
            class_scores_local = class_scores_local.reshape(n_queries, self.tuples_len, -1, self.tuples_len)
            class_scores_local = class_scores_local.permute(0, 2, 1, 3)

            # get query specific class prototype
            query_prototype_local = torch.matmul(class_scores_local, class_v_local)
            query_prototype_local = torch.sum(query_prototype_local, dim=1)

            # calculate distances from queries to query-specific class prototypes
            diff_local = mh_queries_vs_local - query_prototype_local
            norm_sq_local = torch.norm(diff_local, dim=[-2, -1]) ** 2
            distance_local = torch.div(norm_sq_local, self.tuples_len)

            # multiply by -1 to get logits
            distance_local = distance_local * -1
            c_idx = c.long()
            all_distances_tensor_local[:, c_idx] = distance_local

            query_prototype_total = torch.cat((query_prototype, query_prototype_local), 2)
            diff_total = mh_queries_vs_total - query_prototype_total
            norm_sq_total = torch.norm(diff_total, dim=[-2, -1]) ** 2
            distance_total = torch.div(norm_sq_total, self.tuples_len)

            # multiply by -1 to get logits
            distance_total = distance_total * -1
            c_idx = c.long()
            all_distances_tensor_total[:, c_idx] = distance_total

        return_dict = {'logits': all_distances_tensor_total}

        return return_dict

    @staticmethod
    def _extract_class_indices(labels, which_class):
        """
        Helper method to extract the indices of elements which have the specified label.
        ===== 修复6: 确保返回的索引保持在CPU上，让调用者决定设备 =====
        :param labels: (torch.tensor) Labels of the context set.
        :param which_class: Label for which indices are extracted.
        :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
        """
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector


class TemporalCrossTransformer(nn.Module):
    def __init__(self, args, mode, temporal_set_size=3):
        super(TemporalCrossTransformer, self).__init__()

        self.args = args
        self.temporal_set_size = temporal_set_size

        max_len = int(self.args.seq_len * 1.5)

        self.pe = PositionalEncoding((self.args.trans_linear_in_dim), self.args.trans_dropout, max_len=max_len)

        self.k_linear = nn.Linear((self.args.trans_linear_in_dim) * temporal_set_size,
                                  (self.args.trans_linear_out_dim))  # .cuda()
        self.v_linear = nn.Linear((self.args.trans_linear_in_dim) * temporal_set_size,
                                  (self.args.trans_linear_out_dim))  # .cuda()

        self.norm_k = nn.LayerNorm((self.args.trans_linear_out_dim))
        self.norm_v = nn.LayerNorm((self.args.trans_linear_out_dim))

        self.class_softmax = torch.nn.Softmax(dim=1)

        # ===== 修复7: 同样的修复应用到TemporalCrossTransformer =====
        # generate all tuples
        frame_idxs = [i for i in range(self.args.seq_len)]
        frame_combinations = combinations(frame_idxs, temporal_set_size)
        tuples_list = [torch.tensor(comb) for comb in frame_combinations]
        for i, t in enumerate(tuples_list):
            self.register_buffer(f'tuple_{i}', t)
        self.tuples_len = len(tuples_list)

    def _get_tuples(self, device):
        """获取在正确设备上的tuples"""
        tuples = []
        for i in range(self.tuples_len):
            tuples.append(getattr(self, f'tuple_{i}').to(device))
        return tuples

    def forward(self, support_set, support_labels, queries):
        # ===== 获取当前设备 =====
        device = support_set.device

        n_queries = queries.shape[0]
        n_support = support_set.shape[0]

        # static pe
        support_set = self.pe(support_set)
        queries = self.pe(queries)

        # ===== 使用正确设备上的tuples =====
        tuples = self._get_tuples(device)

        # construct new queries and support set made of tuples of images after pe
        s = [torch.index_select(support_set, -2, p).reshape(n_support, -1) for p in tuples]
        q = [torch.index_select(queries, -2, p).reshape(n_queries, -1) for p in tuples]
        support_set = torch.stack(s, dim=-2)
        queries = torch.stack(q, dim=-2)

        # apply linear maps
        support_set_ks = self.k_linear(support_set)
        queries_ks = self.k_linear(queries)
        support_set_vs = self.v_linear(support_set)
        queries_vs = self.v_linear(queries)

        # apply norms where necessary
        mh_support_set_ks = self.norm_k(support_set_ks)
        mh_queries_ks = self.norm_k(queries_ks)
        mh_support_set_vs = support_set_vs
        mh_queries_vs = queries_vs

        unique_labels = torch.unique(support_labels)

        # ===== 使用正确的设备创建张量 =====
        # init tensor to hold distances between every support tuple and every target tuple
        all_distances_tensor = torch.zeros(n_queries, self.args.way, device=device)

        for label_idx, c in enumerate(unique_labels):
            # ===== 确保索引与张量在同一设备 =====
            # select keys and values for just this class
            indices = self.extract_class_indices(support_labels, c).to(device)
            class_k = torch.index_select(mh_support_set_ks, 0, indices)
            class_v = torch.index_select(mh_support_set_vs, 0, indices)
            k_bs = class_k.shape[0]

            class_scores = torch.matmul(mh_queries_ks.unsqueeze(1), class_k.transpose(-2, -1)) / math.sqrt(
                self.args.trans_linear_out_dim)

            # reshape etc. to apply a softmax for each query tuple
            class_scores = class_scores.permute(0, 2, 1, 3)
            class_scores = class_scores.reshape(n_queries, self.tuples_len, -1)
            class_scores = [self.class_softmax(class_scores[i]) for i in range(n_queries)]
            class_scores = torch.stack(class_scores)
            class_scores = class_scores.reshape(n_queries, self.tuples_len, -1, self.tuples_len)
            class_scores = class_scores.permute(0, 2, 1, 3)

            # get query specific class prototype
            query_prototype = torch.matmul(class_scores, class_v)
            query_prototype = torch.sum(query_prototype, dim=1)

            # calculate distances from queries to query-specific class prototypes
            diff = mh_queries_vs - query_prototype
            # print(diff)
            norm_sq = torch.norm(diff, dim=[-2, -1]) ** 2
            # print(norm_sq)
            distance = torch.div(norm_sq, self.tuples_len)

            # multiply by -1 to get logits
            distance = distance * -1
            # print(distance)
            c_idx = c.long()
            all_distances_tensor[:, c_idx] = distance

        return_dict = {'logits': all_distances_tensor}

        return return_dict

    def extract_class_indices(self, labels, which_class):
        """
        Helper method to extract the indices of elements which have the specified label.
        :param labels: (torch.tensor) Labels of the context set.
        :param which_class: Label for which indices are extracted.
        :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
        """
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector


class CNN_TRX(nn.Module):
    """
    Standard Resnet connected to a Temporal Cross Transformer.

    """

    def __init__(self, args):
        super(CNN_TRX, self).__init__()

        self.train()
        self.args = args

        # if self.args.method == "resnet18":
        #    resnet = models.resnet18(pretrained=True)
        # elif self.args.method == "resnet34":
        #    resnet = models.resnet34(pretrained=True)
        # elif self.args.method == "resnet50":
        #    resnet = models.resnet50(pretrained=True)

        last_layer_idx = -1
        # self.resnet = nn.Sequential(*list(resnet.children())[:last_layer_idx])

        self.resnet = AdaFocus(64, 'image')
        self.transformers_global = nn.ModuleList([TemporalCrossTransformer(args, 'global', s) for s in args.temp_set])
        self.transformers_local = nn.ModuleList([TemporalCrossTransformer(args, 'local', s) for s in args.temp_set])
        self.transformers = nn.ModuleList([TemporalCrossTransformer_multihead(args, 'all', s) for s in args.temp_set])

        self.setmatcher_all = DistanceLoss(args, 'all')

    def forward(self, context_images, context_labels, target_images, mode):
        n_support_images = context_images.shape[0]
        total_images = torch.cat((context_images, target_images), 0)
        # total_features=self.resnet(total_images).squeeze()
        total_features, total_features_global, total_features_local = self.resnet(total_images, mode)
        context_features = total_features[0:n_support_images, :].squeeze()
        target_features = total_features[n_support_images:, :].squeeze()
        context_features_global = total_features_global[0:n_support_images, :].squeeze()
        target_features_global = total_features_global[n_support_images:, :].squeeze()
        context_features_local = total_features_local[0:n_support_images, :].squeeze()
        target_features_local = total_features_local[n_support_images:, :].squeeze()
        # context_features = self.resnet(context_images).squeeze()
        # target_features = self.resnet(target_images).squeeze()

        dim = int(context_features.shape[1])
        dim_notall = int(context_features_local.shape[1])

        context_features = context_features.reshape(-1, self.args.seq_len, dim)
        target_features = target_features.reshape(-1, self.args.seq_len, dim)
        context_features_global = context_features_global.reshape(-1, self.args.seq_len, dim_notall)
        target_features_global = target_features_global.reshape(-1, self.args.seq_len, dim_notall)
        context_features_local = context_features_local.reshape(-1, self.args.seq_len, dim_notall)
        target_features_local = target_features_local.reshape(-1, self.args.seq_len, dim_notall)

        all_logits = [t(context_features, context_labels, target_features)['logits'] for t in self.transformers]
        all_logits = torch.stack(all_logits, dim=-1)
        sample_logits = all_logits
        sample_logits = torch.mean(sample_logits, dim=[-1])

        all_logits_global = [t(context_features_global, context_labels, target_features_global)['logits'] for t in
                             self.transformers_global]
        all_logits_global = torch.stack(all_logits_global, dim=-1)
        sample_logits_global = all_logits_global
        sample_logits_global = torch.mean(sample_logits_global, dim=[-1])

        all_logits_local = [t(context_features_local, context_labels, target_features_local)['logits'] for t in
                            self.transformers_local]
        all_logits_local = torch.stack(all_logits_local, dim=-1)
        sample_logits_local = all_logits_local
        sample_logits_local = torch.mean(sample_logits_local, dim=[-1])

        match_logits_all = [
            self.setmatcher_all(context_features_global, context_features_local, context_labels, target_features_global,
                                target_features_local)['logits']]
        match_logits_all = torch.stack(match_logits_all, dim=-1)
        match_logits_all = torch.mean(match_logits_all, dim=[-1])

        return_dict = {'logits': split_first_dim_linear(sample_logits, [NUM_SAMPLES, target_features.shape[0]]),
                       'logits_global': split_first_dim_linear(sample_logits_global,
                                                               [NUM_SAMPLES, target_features_global.shape[0]]),
                       'logits_local': split_first_dim_linear(sample_logits_local,
                                                              [NUM_SAMPLES, target_features_local.shape[0]]),
                       'matcher': split_first_dim_linear(match_logits_all, [NUM_SAMPLES, target_features.shape[0]])}
        return return_dict

    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        if self.args.num_gpus > 1:
            self.resnet.cuda(0)
            self.resnet = torch.nn.DataParallel(self.resnet, device_ids=[i for i in range(0, self.args.num_gpus)])

            self.transformers.cuda(0)


if __name__ == "__main__":
    class ArgsObject(object):
        def __init__(self):
            self.trans_linear_in_dim = 512
            self.trans_linear_out_dim = 128

            self.way = 5
            self.shot = 1
            self.query_per_class = 5
            self.trans_dropout = 0.1
            self.seq_len = 4
            self.img_size = 84
            self.method = "resnet18"
            self.num_gpus = 1
            self.temp_set = [2, 3]


    args = ArgsObject()
    torch.manual_seed(0)

    device = 'cuda:0'
    model = CNN_TRX(args).to(device)

    support_imgs = torch.rand(args.way * args.shot * args.seq_len, 3, args.img_size, args.img_size).to(device)
    target_imgs = torch.rand(args.way * args.query_per_class * args.seq_len, 3, args.img_size, args.img_size).to(device)
    support_labels = torch.tensor([0, 1, 2, 3, 4]).to(device)

    print("Support images input shape: {}".format(support_imgs.shape))
    print("Target images input shape: {}".format(target_imgs.shape))
    print("Support labels input shape: {}".format(support_labels.shape))

    out = model(support_imgs, support_labels, target_imgs)

    print("TRX returns the distances from each query to each class prototype.  Use these as logits.  Shape: {}".format(
        out['logits'].shape))

    print("TRX returns the distances from each query to each class prototype.  Use these as logits.  Shape: {}".format(out['logits'].shape))





