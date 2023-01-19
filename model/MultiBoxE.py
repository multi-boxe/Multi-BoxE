import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from .BaseModule import BaseModule


class MultiBoxE(BaseModule):

    def __init__(self, ent_tot, rel_tot, box_tot, start_idx, box_num, embedding_dim=100, weight_init_args=(0, 1),
                 norm_flag=True,
                 bound_norm=True, use_gpu=True,
                 score_ord=1,
                 distance=0,
                 dis_arg=1):
        """
        :param box_tot: 所有关系包含的所有box数量
        :param start_idx: 每个关系box的初始index
        :param box_num: 每个关系 head/tail box 的个数 (rel_tot, 2)
        :param weight_init_args: 初始化的范围
        :param norm_flag: 是否用 Tanh
        :param bound_norm: boxe特殊的box处理
        :param score_ord: 计算得分的范数
        """
        super(MultiBoxE, self).__init__(ent_tot, rel_tot)
        self.init_f = nn.init.uniform_  # 用均匀分布来初始化tensor

        if norm_flag:
            self.embedding_norm_fn = nn.Tanh()  # 论文中提到：作用于embedding
        else:
            self.embedding_norm_fn = nn.Identity()  # 不改变输入

        self.box_tot = box_tot  # 头尾box的数量 (2,)
        self.start_idx = start_idx  # (346,2)
        self.box_num = box_num  # (346,2)
        self.use_gpu = use_gpu

        self.bounded_norm = bound_norm
        self.NORM_LOG_BOUND = nn.Parameter(torch.Tensor([1]))
        self.NORM_LOG_BOUND.requires_grad = False
        # 距离函数 0-boxe 1-query2box 3-newEmb
        if distance == 0:
            self.score_ord = score_ord  # score 的范数
        elif distance == 1:
            self.score_ord = 1
            self.dis_arg = dis_arg
        self.distance = distance
            
        self.dim = embedding_dim
        # entity base point, translation bump
        self.entity_bases = nn.Embedding(self.ent_tot, embedding_dim)
        self.entity_bumps = nn.Embedding(self.ent_tot, embedding_dim)

        self.init_f(self.entity_bases.weight, *weight_init_args)
        self.init_f(self.entity_bumps.weight, *weight_init_args)

        # relation box: head box and tail box; base point(center or lower boundary?) + widths
        self.r_head_base_points = nn.Embedding(self.box_tot[0], embedding_dim)
        self.r_head_widths = nn.Embedding(self.box_tot[0], embedding_dim)
        self.r_head_size_scales = nn.Embedding(self.box_tot[0], 1)  # 头尾各对应一个scale
        self.r_tail_base_points = nn.Embedding(self.box_tot[1], embedding_dim)
        self.r_tail_widths = nn.Embedding(self.box_tot[1], embedding_dim)
        self.r_tail_size_scales = nn.Embedding(self.box_tot[1], 1)

        self.init_f(self.r_head_base_points.weight, *weight_init_args)  # *(0,1)将可迭代序列拆开，作为函数的实参
        self.init_f(self.r_head_widths.weight, *weight_init_args)
        self.init_f(self.r_head_size_scales.weight, -1, 1)  # ?
        self.init_f(self.r_tail_base_points.weight, *weight_init_args)
        self.init_f(self.r_tail_widths.weight, *weight_init_args)
        self.init_f(self.r_tail_size_scales.weight, -1, 1)
        
    def shape_norm(self, t, dim, bounded_norm=False):
        """
        taken from original BoxE implementation
        :param t: width -> (nb, dim)
        :param dim:
        :param bounded_norm:
        :return:
        """
        step1_tensor = torch.abs(t)
        step2_tensor = step1_tensor + (10 ** -8)
        log_norm_tensor = torch.log(step2_tensor)

        if not bounded_norm:
            step3_tensor = torch.mean(log_norm_tensor, dim=dim, keepdim=True)
            norm_volume = torch.exp(step3_tensor)
            return t / norm_volume
        else:
            minsize_tensor = torch.minimum(torch.min(log_norm_tensor, dim=1, keepdim=True).values, -self.NORM_LOG_BOUND)
            maxsize_tensor = torch.maximum(torch.max(log_norm_tensor, dim=1, keepdim=True).values, self.NORM_LOG_BOUND)
            minsize_ratio = -self.NORM_LOG_BOUND / minsize_tensor
            maxsize_ratio = self.NORM_LOG_BOUND / maxsize_tensor
            size_norm_ratio = torch.minimum(minsize_ratio, maxsize_ratio)
            normed_tensor = log_norm_tensor * size_norm_ratio
            return torch.exp(normed_tensor)

    def locate_idx(self, r_idx, hb, tb):
        """
        计算具体box的index
        :param r_idx: rid
        :param hb: head box id 必须是np.array 或者 list
        :param tb: tail box id 必须是np.array 或者 list
        :return: index of head/tail box
        """
        hb_idx = self.start_idx[r_idx, 0] + hb
        tb_idx = self.start_idx[r_idx, 1] + tb

        if self.use_gpu:
            hb_idx = Variable(torch.from_numpy(hb_idx).cuda())
            tb_idx = Variable(torch.from_numpy(tb_idx).cuda())
        else:
            hb_idx = Variable(torch.from_numpy(hb_idx))
            tb_idx = Variable(torch.from_numpy(tb_idx))

        return hb_idx, tb_idx

    def get_rel_embedding(self, hb_idx, tb_idx):
        """
        :param hb_idx: index of head box (nb_examples,)
        :param tb_idx:
        :return: (nb, dim, 2, 2) -> 第一个2: h & t; 第二个2: upper & lower
        """
        rhb = self.r_head_base_points(hb_idx)
        rtb = self.r_tail_base_points(tb_idx)

        rhw = self.shape_norm(self.r_head_widths(hb_idx), dim=1,
                              bounded_norm=self.bounded_norm)  # normalize relative widths
        rtw = self.shape_norm(self.r_tail_widths(tb_idx), dim=1,
                              bounded_norm=self.bounded_norm)  # dim=1

        rhs = nn.functional.elu(self.r_head_size_scales(hb_idx)) + 1  # ensure scales > 0
        rts = nn.functional.elu(self.r_tail_size_scales(tb_idx)) + 1

        # compute scaled widths
        h_deltas = torch.multiply(rhw, rhs) * 0.5
        t_deltas = torch.multiply(rtw, rts) * 0.5

        # compute corners from base and width
        h_corner_1 = rhb + h_deltas
        h_corner_2 = rhb - h_deltas
        t_corner_1 = rtb + t_deltas
        t_corner_2 = rtb - t_deltas

        # determine upper and lower corners
        # (nb, dim)
        h_upper = torch.maximum(h_corner_1, h_corner_2)
        h_lower = torch.minimum(h_corner_1, h_corner_2)
        t_upper = torch.maximum(t_corner_1, t_corner_2)
        t_lower = torch.minimum(t_corner_1, t_corner_2)

        # assemble boxes
        # (nb, dim, 2)
        r_h_boxes = torch.stack((h_upper, h_lower), dim=2)
        r_t_boxes = torch.stack((t_upper, t_lower), dim=2)

        # (nb, dim, 2, 2)
        return self.embedding_norm_fn(torch.stack((r_h_boxes, r_t_boxes), dim=2))

    def get_ent_embedding(self, h_idx, t_idx):
        """
        @:param h_idx: (nb,) Tensor
        @:return (nb_examples, dim, 2) -> 2 is h & t
        """
        h_base = self.entity_bases(h_idx)
        h_bump = self.entity_bumps(h_idx)

        t_base = self.entity_bases(t_idx)
        t_bump = self.entity_bumps(t_idx)

        return self.embedding_norm_fn(torch.stack((h_base + t_bump, t_base + h_bump), dim=2))  # 具体的实体表示做tanh

    def dist(self, entity_emb, boxes):
        """
        center-based distance(boxe)
        :param entity_emb: (nb_examples, dim, 2)
        :param boxes: (nb_examples, dim, 2, 2)
        :return: (nb, emb, 2)
        """
        ub = boxes[:, :, :, 0]  # (nb_examples, dim, 2)
        lb = boxes[:, :, :, 1]
        c = (lb + ub) / 2  # (nb_examples, dim, 2)
        w = ub - lb + 1
        k = 0.5 * (w - 1) * (w - (1 / w))

        # [nb, dim, 2] -> 2:h&t
        flag = torch.logical_and(torch.ge(entity_emb, lb), torch.le(entity_emb, ub))
        d = torch.where(flag,
                        torch.abs(entity_emb - c) / w,
                        torch.abs(entity_emb - c) * w - k)
        return d, flag
    
    def dist2(self, entity_emb, boxes):
        """
        inside-outside distance(query2box)
        """
        ub = boxes[:, :, :, 0]  # (nb_examples, dim, 2)
        lb = boxes[:, :, :, 1]
        c = (lb + ub) / 2  # (nb_examples, dim, 2)
        
        flag = torch.logical_and(torch.ge(entity_emb, lb), torch.le(entity_emb, ub))
        
        m = nn.ReLU()
        d = torch.where(flag,
                        self.dis_arg * (c - torch.minimum(ub, torch.maximum(lb, entity_emb))),
                        m(entity_emb-ub) + m(lb-entity_emb))
        return d
    
    def dist3(self, entity_emb, boxes):
        """
        violation-based distance
        :return: (nb,1,2)
        """
        ub = boxes[:, :, :, 0]  # (nb, dim, 2)
        lb = boxes[:, :, :, 1]
        
        flag = torch.logical_and(torch.ge(entity_emb, lb), torch.le(entity_emb, ub)) # 逐维度判断实体是否在box范围内，(nb,dim,2)
        flag = flag.sum(dim=1, keepdim=True) == self.dim # 实体所有维度都在box范围内，则视为实体 in box，(nb,1,2)
        
        diff = torch.abs(lb - entity_emb) # ld-ed
        d = torch.where(flag, torch.amax(diff,dim=1,keepdim=True), torch.amin(diff,dim=1,keepdim=True)) # (nb,1,2)

        return d
        

    def calc(self, entities, relations, order):
        """
        :param entities: (nb_examples, dim, 2)
        :param relations: (nb_examples, dim, 2, 2)
        :param order: normalization
        :return: (nb_examples): nb_examples include positive and negative
        """
        distance, flag = self.dist(entities, relations)
        return distance.norm(dim=1, p=order).sum(dim=1), \
            flag.cpu().data.numpy()
    
    def calc2(self, entities, relations, order):
        distance = self.dist2(entities, relations)
        return distance.norm(dim=1, p=order).sum(dim=1) # 默认取1-范数
    
    def calc3(self, entities, relations):
        distance = self.dist3(entities, relations)
        return torch.squeeze(distance,dim=1).sum(dim=1)

    def forward(self, data):
        """
        :param data: dict, 只有batch_h/t是(nb_examples,) tensor, 其余是np.array
        :return score: (nb_examples,) tensor
        """
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        batch_hb = data['batch_hb']
        batch_tb = data['batch_tb']

        entity_emb = self.get_ent_embedding(batch_h, batch_t)
        hb_idx, tb_idx = self.locate_idx(batch_r, batch_hb, batch_tb)
        rel_emb = self.get_rel_embedding(hb_idx, tb_idx)

        if self.distance == 0:
            score = self.calc(entity_emb, rel_emb, self.score_ord)[0]
        elif self.distance == 1:
            score = self.calc2(entity_emb, rel_emb, self.score_ord)
        elif self.distance == 2:
            score = self.calc3(entity_emb, rel_emb)
        return score

    def predict(self, data):
        """
        计算实体在所有box中的得分(用于选择test triple对应的最优排名)
        :param data: test batch
        :return: scores: (box_num, ent_tot)
        """
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']

        batch_hb = data['batch_hb']
        batch_tb = data['batch_tb']
        hb = batch_hb[0]
        tb = batch_tb[0]

        rid = batch_r[0]

        hb_num, tb_num = self.box_num[rid]
        entity_emb = self.get_ent_embedding(batch_h, batch_t)

        scores = []

        if hb == -1 and tb == -1:
            for i in range(hb_num):
                for j in range(tb_num):
                    hb_idx, tb_idx = self.locate_idx(batch_r, batch_hb + 1 + i, batch_tb + 1 + j)
                    rel_emb = self.get_rel_embedding(hb_idx, tb_idx)
                    tmp = self.calc(entity_emb, rel_emb, self.score_ord)[0].cpu().data.numpy()
                    scores.append(tmp)
            scores = np.array(scores)
        elif hb == -1:
            for i in range(hb_num):
                hb_idx, tb_idx = self.locate_idx(batch_r, batch_hb + 1 + i, batch_tb)
                rel_emb = self.get_rel_embedding(hb_idx, tb_idx)
                tmp = self.calc(entity_emb, rel_emb, self.score_ord)[0].cpu().data.numpy()
                scores.append(tmp)
            scores = np.array(scores)
        elif tb == -1:
            for i in range(tb_num):
                hb_idx, tb_idx = self.locate_idx(batch_r, batch_hb, batch_tb + 1 + i)
                rel_emb = self.get_rel_embedding(hb_idx, tb_idx)
                tmp = self.calc(entity_emb, rel_emb, self.score_ord)[0].cpu().data.numpy()
                scores.append(tmp)
            scores = np.array(scores)
        else:
            scores = self.forward(data).cpu().data.numpy().reshape(1, self.ent_tot)

        return scores

    def predict4(self, data):
        score = self.forward(data)
        return score.cpu().data.numpy()
