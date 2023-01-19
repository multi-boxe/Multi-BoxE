# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import sys
import datetime
import ctypes
import json
import numpy as np
from sklearn.metrics import roc_auc_score
import copy
from tqdm import tqdm


class MyTester(object):

    def __init__(self, model=None, data_loader=None, use_gpu=True):
        base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../release/Base.so"))
        self.lib = ctypes.cdll.LoadLibrary(base_file)
        self.lib.testHead.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        self.lib.testTail.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        self.lib.test_link_prediction.argtypes = [ctypes.c_int64]

        self.lib.getTestLinkMRR.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkMR.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit10.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit3.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit1.argtypes = [ctypes.c_int64]

        self.lib.getTestLinkMRR.restype = ctypes.c_float
        self.lib.getTestLinkMR.restype = ctypes.c_float
        self.lib.getTestLinkHit10.restype = ctypes.c_float
        self.lib.getTestLinkHit3.restype = ctypes.c_float
        self.lib.getTestLinkHit1.restype = ctypes.c_float

        self.model = model
        self.data_loader = data_loader
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.model.cuda()

    def set_model(self, model):
        self.model = model

    def set_data_loader(self, data_loader):
        self.data_loader = data_loader

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu
        if self.use_gpu and self.model != None:
            self.model.cuda()

    def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

    def test_one_step(self, data, flag, fid):
        """
        :param flag: 预测 h or t
        :param fid: 选择预测函数
        :return:
        """
        if fid:
            # 选取最优score的排名作为结果
            return self.model.predict4({
                'batch_h': self.to_var(data['batch_h'], self.use_gpu),
                'batch_t': self.to_var(data['batch_t'], self.use_gpu),
                'batch_r': data['batch_r'],

                'batch_hb': data['batch_hb'],
                'batch_tb': data['batch_tb'],
                'hid': data['hid'],
                'tid': data['tid'],
                'mode': data['mode']
            }, flag)
        else:
            # 所有triple都选最优box然后排名
            return self.model.predict({
                'batch_h': self.to_var(data['batch_h'], self.use_gpu),
                'batch_t': self.to_var(data['batch_t'], self.use_gpu),
                'batch_r': data['batch_r'],

                'batch_hb': data['batch_hb'],
                'batch_tb': data['batch_tb'],
                'hid': data['hid'],
                'tid': data['tid'],
                'mode': data['mode']
            }, flag)

    def run_link_prediction(self, type_constrain=False, fid=1):
        self.lib.initTest()
        self.data_loader.set_sampling_mode('link')
        if type_constrain:
            type_constrain = 1
        else:
            type_constrain = 0
        training_range = tqdm(self.data_loader)

        # 记录每个测试样例的排名
        h_rank = []
        t_rank = []
        test_res = []

        for index, [data_head, data_tail] in enumerate(training_range):
            # index是由enumerate生成的索引，对应着具体哪个测试样例
            # 0 -> testHead
            score, tmp = self.test_one_step(data_head, 0, fid)
            tmp1 = self.lib.testHead(score.__array_interface__["data"][0], index, type_constrain)

            score, _ = self.test_one_step(data_tail, 1, fid)
            tmp2 = self.lib.testTail(score.__array_interface__["data"][0], index, type_constrain)

            h_rank.append(tmp1)
            t_rank.append(tmp2)
            test_res.append(tmp)

        # 总结输出
        self.lib.test_link_prediction(type_constrain)

        mrr = self.lib.getTestLinkMRR(type_constrain)
        mr = self.lib.getTestLinkMR(type_constrain)
        hit10 = self.lib.getTestLinkHit10(type_constrain)
        hit3 = self.lib.getTestLinkHit3(type_constrain)
        hit1 = self.lib.getTestLinkHit1(type_constrain)
        print(hit10)
        return h_rank, t_rank, test_res

    # ==========================选最优排名=====================================

    def test_one_step2(self, data):
        return self.model.predict2({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': data['batch_r'],
            'batch_hb': data['batch_hb'],
            'batch_tb': data['batch_tb'],
            'mode': data['mode']
        })

    def print_rank(self, hrk, trk):
        test_total = self.lib.getTestTotal()

        hmmr = np.sum(1.0 / hrk) / test_total
        tmmr = np.sum(1.0 / trk) / test_total

        hmr = np.sum(hrk) / test_total
        tmr = np.sum(trk) / test_total

        hrk10 = len(np.where(hrk <= 10)[0]) / len(hrk)
        trk10 = len(np.where(trk <= 10)[0]) / len(trk)

        hrk3 = len(np.where(hrk <= 3)[0]) / len(hrk)
        trk3 = len(np.where(trk <= 3)[0]) / len(trk)

        hrk1 = len(np.where(hrk == 1)[0]) / len(hrk)
        trk1 = len(np.where(trk == 1)[0]) / len(trk)

        print("metric:\t\t\t MRR \t\t MR \t\t hit@10 \t hit@3  \t hit@1")
        print("h(filter):\t\t {:.6f} \t {:.6f} \t {:.6f} \t {:.6f} \t {:.6f}".format(hmmr, hmr, hrk10, hrk3, hrk1))
        print("t(filter):\t\t {:.6f} \t {:.6f} \t {:.6f} \t {:.6f} \t {:.6f}".format(tmmr, tmr, trk10, trk3, trk1))
        print("averaged(filter):\t {:.6f} \t {:.6f} \t {:.6f} \t {:.6f} \t {:.6f}".format((hmmr + tmmr) / 2,
                                                                                          (hmr + tmr) / 2,
                                                                                          (hrk10 + trk10) / 2,
                                                                                          (hrk3 + trk3) / 2,
                                                                                          (hrk1 + trk1) / 2))

    def run_link_prediction2(self, type_constrain=False):
        self.lib.initTest()
        self.data_loader.set_sampling_mode('link')
        if type_constrain:
            type_constrain = 1
        else:
            type_constrain = 0
        testing_range = tqdm(self.data_loader)

        hrk_mean = []
        hrk_best = []
        trk_mean = []
        trk_best = []

        # res_h = []
        # res_t = []

        for index, [data_head, data_tail] in enumerate(testing_range):
            hrk = []
            scores = self.test_one_step2(data_head)
            for sc in scores:
                hrk.append(self.lib.testHead(sc.__array_interface__["data"][0], index, type_constrain))
            hrk_mean.append(np.mean(hrk))
            hrk_best.append(np.min(hrk))

            # tmp1 = len(scores)
            # if tmp1 == 1:
            #     tmp2 = 0
            # else:
            #     tmp2 = np.argmin(scores[:, data_head['hid']].reshape([-1]))
            # tmp3 = np.argmin(hrk)
            # res_h.append(f'{tmp2}:{tmp1}:{tmp3}')

            trk = []
            scores = self.test_one_step2(data_tail)
            for sc in scores:
                trk.append(self.lib.testTail(sc.__array_interface__["data"][0], index, type_constrain))
            trk_mean.append(np.mean(trk))
            trk_best.append(np.min(trk))

            # tmp1 = len(scores)
            # if tmp1 == 1:
            #     tmp2 = 0
            # else:
            #     tmp2 = np.argmin(scores[:, data_tail['tid']].reshpe([-1]))
            # tmp3 = np.argmin(trk)
            # res_t.append(f'{tmp2}:{tmp1}:{tmp3}')

        print("==================Best Rank======================")
        self.print_rank(np.array(hrk_best), np.array(trk_best))
        print("==================Mean Rank======================")
        self.print_rank(np.array(hrk_mean), np.array(trk_mean))

        # return res_h, res_t

    # ================================================================================

    # ====================================Multi-BoxE============================================
    def test_one_step3(self, data):
        return self.model.predict({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': data['batch_r'],
            'batch_hb': data['batch_hb'],
            'batch_tb': data['batch_tb'],
            'mode': data['mode']
        })

    def run_link_prediction3(self, type_constrain=False):
        self.lib.initTest()
        self.data_loader.set_sampling_mode('link')
        if type_constrain:
            type_constrain = 1
        else:
            type_constrain = 0
        testing_range = tqdm(self.data_loader)

        hrk_best = []
        hsc_best = []
        trk_best = []
        tsc_best = []

        res1 = []  # (h, t, r, 最优得分对应的box, 最优排名对应的box)
        res2 = []

        for index, [data_head, data_tail] in enumerate(testing_range):
            rid = data_head['batch_r'][0]
            hb_num = self.model.box_num[rid][0]
            tb_num = self.model.box_num[rid][1]
            hid = data_head['hid'][0]
            tid = data_head['tid'][0]
            hb = data_head['batch_hb'][0]
            tb = data_head['batch_tb'][0]

            hrk = []
            scores = self.test_one_step3(data_head)
            tmp = scores[:, hid]  # scores of correct triple
            sc_min_idx = np.argmin(tmp)  # index of best correct score
            for sc in scores:
                hrk.append(self.lib.testHead(sc.__array_interface__["data"][0], index, type_constrain))
            rk_min_idx = np.argmin(hrk)  # index of best rank
            hrk_best.append(np.min(hrk))  # 最优的排名
            hsc_best.append(hrk[sc_min_idx])  # 最优得分对应的排名
            if hb == -1 and tb == -1:
                hb_sc_best = sc_min_idx // tb_num
                tb_sc_best = sc_min_idx % tb_num
                hb_rk_best = rk_min_idx // tb_num
                tb_rk_best = rk_min_idx % tb_num
            elif hb == -1:
                hb_sc_best = sc_min_idx % hb_num
                tb_sc_best = tb
                hb_rk_best = rk_min_idx % hb_num
                tb_rk_best = tb
            elif tb == -1:
                tb_sc_best = sc_min_idx % tb_num
                hb_sc_best = hb
                tb_rk_best = rk_min_idx % tb_num
                hb_rk_best = hb
            else:
                hb_sc_best = hb
                tb_sc_best = tb
                hb_rk_best = hb
                tb_rk_best = tb
            res1.append([hid, tid, rid, hb_sc_best, tb_sc_best, hb_rk_best, tb_rk_best])

            trk = []
            scores = self.test_one_step3(data_tail)
            tmp = scores[:, tid]
            sc_min_idx = np.argmin(tmp)
            for sc in scores:
                trk.append(self.lib.testTail(sc.__array_interface__["data"][0], index, type_constrain))
            rk_min_idx = np.argmin(trk)
            trk_best.append(np.min(trk))
            tsc_best.append(trk[sc_min_idx])
            if hb == -1 and tb == -1:
                hb_sc_best = sc_min_idx // tb_num
                tb_sc_best = sc_min_idx % tb_num
                hb_rk_best = rk_min_idx // tb_num
                tb_rk_best = rk_min_idx % tb_num
            elif hb == -1:
                hb_sc_best = sc_min_idx % hb_num
                tb_sc_best = tb
                hb_rk_best = rk_min_idx % hb_num
                tb_rk_best = tb
            elif tb == -1:
                tb_sc_best = sc_min_idx % tb_num
                hb_sc_best = hb
                tb_rk_best = rk_min_idx % tb_num
                hb_rk_best = hb
            else:
                hb_sc_best = hb
                tb_sc_best = tb
                hb_rk_best = hb
                tb_rk_best = tb
            res2.append([hid, tid, rid, hb_sc_best, tb_sc_best, hb_rk_best, tb_rk_best])

        print("==================Best Rank======================")
        self.print_rank(np.array(hrk_best), np.array(trk_best))
        print("==================Best Score Rank======================")
        self.print_rank(np.array(hsc_best), np.array(tsc_best))

        return res1, res2

    # ====================================Multi-BoxE2============================================
    def test_one_step4(self, data, hb, tb):
        if not isinstance(hb, np.ndarray):
            hb = np.array([hb], dtype=int)
        if not isinstance(tb, np.ndarray):
            tb = np.array([tb], dtype=int)
        return self.model.predict4({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': data['batch_r'],
            'batch_hb': hb,
            'batch_tb': tb,
            'mode': data['mode']
        })

    def pred4(self, data, flag, index, type_constrain):
        """
        :param data:
        :param flag:
        :return: 得分最优排名 排名最优排名 对应的box
        """
        rid = data['batch_r'][0]
        hb_num = self.model.box_num[rid][0]
        tb_num = self.model.box_num[rid][1]
        hid = data['hid'][0]
        tid = data['tid'][0]
        hb = data['batch_hb'][0]
        tb = data['batch_tb'][0]
        idx = tid if flag else hid
        if flag:
            test = self.lib.testTail
        else:
            test = self.lib.testHead

        # 确定哪个box
        hb_sc_best = hb
        tb_sc_best = tb
        hb_rk_best = hb
        tb_rk_best = tb

        minimum = np.inf  # 最优得分
        best_sc_rk = np.inf  # 最优得分对应的排名
        best_rk_rk = np.inf  # 最优排名对应的排名
        best_rk_sc = 0  # 最优排名对应的得分

        correct_sc = []  # correct triple 在不同box组合下的得分
        rk_list = []  # correct triple 在不同box组合下的排名

        if hb == -1 and tb == -1:
            for i in range(hb_num):
                for j in range(tb_num):
                    sc = self.test_one_step4(data, i, j)  # narray, (ent_num, )
                    tmp1 = sc[idx]
                    tmp2 = test(sc.__array_interface__["data"][0], index, type_constrain)
                    correct_sc.append(tmp1)
                    rk_list.append(tmp2)
                    if tmp1 < minimum:
                        # 最优得分对应的rank
                        minimum = tmp1
                        best_sc_rk = tmp2

            best_rk_rk = np.min(rk_list)
            # 计算出具体的box
            min_sc_idx = np.argmin(correct_sc)
            min_rk_idx = np.argmin(rk_list)
            hb_sc_best = min_sc_idx // tb_num
            tb_sc_best = min_sc_idx % tb_num
            hb_rk_best = min_rk_idx // tb_num
            tb_rk_best = min_rk_idx % tb_num
            best_rk_sc = correct_sc[min_rk_idx]
        elif hb == -1:
            for i in range(hb_num):
                sc = self.test_one_step4(data, i, tb)
                tmp1 = sc[idx]
                tmp2 = test(sc.__array_interface__["data"][0], index, type_constrain)
                correct_sc.append(tmp1)
                rk_list.append(tmp2)
                if tmp1 < minimum:
                    minimum = tmp1
                    best_sc_rk = tmp2
            best_rk_rk = np.min(rk_list)
            min_sc_idx = np.argmin(correct_sc)
            min_rk_idx = np.argmin(rk_list)
            hb_sc_best = min_sc_idx % hb_num
            tb_sc_best = tb
            hb_rk_best = min_rk_idx % hb_num
            tb_rk_best = tb
            best_rk_sc = correct_sc[min_rk_idx]
        elif tb == -1:
            for i in range(tb_num):
                sc = self.test_one_step4(data, hb, i)
                tmp1 = sc[idx]
                tmp2 = test(sc.__array_interface__["data"][0], index, type_constrain)
                correct_sc.append(tmp1)
                rk_list.append(tmp2)
                if tmp1 < minimum:
                    minimum = tmp1
                    best_sc_rk = tmp2
            best_rk_rk = np.min(rk_list)
            min_sc_idx = np.argmin(correct_sc)
            min_rk_idx = np.argmin(rk_list)
            hb_sc_best = hb
            tb_sc_best = min_sc_idx % tb_num
            hb_rk_best = hb
            tb_rk_best = min_rk_idx % tb_num
            best_rk_sc = correct_sc[min_rk_idx]
        else:
            sc = self.test_one_step4(data, hb, tb)
            tmp1 = sc[idx]
            tmp2 = test(sc.__array_interface__["data"][0], index, type_constrain)

            minimum = tmp1
            best_rk_sc = tmp1

            best_sc_rk = tmp2
            best_rk_rk = tmp2

        res = [hid, tid, rid, hb_sc_best, tb_sc_best, minimum, hb_rk_best, tb_rk_best, best_rk_sc]

        return best_sc_rk, best_rk_rk, res

    def run_link_prediction4(self, type_constrain=False):
        self.lib.initTest()
        self.data_loader.set_sampling_mode('link')
        if type_constrain:
            type_constrain = 1
        else:
            type_constrain = 0
        testing_range = tqdm(self.data_loader)

        hsc_best = []
        hrk_best = []

        tsc_best = []
        trk_best = []

        res1 = []
        res2 = []

        for index, [data_head, data_tail] in enumerate(testing_range):
            tmp1, tmp2, tmp3 = self.pred4(data_head, 0, index, type_constrain)
            hsc_best.append(tmp1)
            hrk_best.append(tmp2)
            res1.append(tmp3)

            tmp1, tmp2, tmp3 = self.pred4(data_tail, 1, index, type_constrain)
            tsc_best.append(tmp1)
            trk_best.append(tmp2)
            res2.append(tmp3)

        print("==================Best Rank======================")
        self.print_rank(np.array(hrk_best), np.array(trk_best))
        print("==================Best Score Rank======================")
        self.print_rank(np.array(hsc_best), np.array(tsc_best))

    # ====================================Multi-BoxE3============================================
    # 只选最优得分对应的排名
    def test_one_step5(self, hid, tid, rid, test_hb, test_tb):
        return self.model.predict4({
            'batch_h': self.to_var(np.array([hid]), self.use_gpu),
            'batch_t': self.to_var(np.array([tid]), self.use_gpu),
            'batch_r': np.array([rid]),
            'batch_hb': test_hb,
            'batch_tb': test_tb,
        })

    def find_best_sc_box(self, hid, tid, rid, hb, tb, hb_num, tb_num):
        if hb == -1 and tb == -1:
            minimum = np.inf
            test_hb = np.zeros(hb_num * tb_num, dtype=np.int64)
            test_tb = np.zeros(hb_num * tb_num, dtype=np.int64)
            for i in range(hb_num):
                test_hb[i * tb_num:(i + 1) * tb_num] = i
                for j in range(tb_num):
                    test_tb[i * tb_num + j] = j
            for i in range(hb_num):  # box数量比较多时，按批次来计算得分
                sc = self.test_one_step5(hid, tid, rid,
                                         test_hb[i * tb_num:(i + 1) * tb_num],
                                         test_tb[i * tb_num:(i + 1) * tb_num])
                tmp1 = np.min(sc)
                if tmp1 < minimum:
                    minimum = tmp1
                    best_hb = i
                    best_tb = np.argmin(sc)
        elif hb == -1:
            test_hb = np.arange(hb_num, dtype=np.int64)
            test_tb = np.ones(hb_num, dtype=np.int64) * tb
            sc = self.test_one_step5(hid, tid, rid, test_hb, test_tb)
            best_sc_idx = np.argmin(sc)
            best_hb = best_sc_idx % hb_num
            best_tb = tb
        elif tb == -1:
            test_hb = np.ones(tb_num, dtype=np.int64) * hb
            test_tb = np.arange(tb_num, dtype=np.int64)
            sc = self.test_one_step5(hid, tid, rid, test_hb, test_tb)
            best_sc_idx = np.argmin(sc)
            best_hb = hb
            best_tb = best_sc_idx % tb_num
        else:
            best_hb = hb
            best_tb = tb
        return best_hb, best_tb

    def run_link_prediction5(self, type_constrain=False):
        self.lib.initTest()
        self.data_loader.set_sampling_mode('link')
        if type_constrain:
            type_constrain = 1
        else:
            type_constrain = 0
        testing_range = tqdm(self.data_loader)

        hrk = []
        trk = []

        for index, [data_head, data_tail] in enumerate(testing_range):
            rid = data_head['batch_r'][0]
            hb_num = self.model.box_num[rid][0]
            tb_num = self.model.box_num[rid][1]
            hid = data_head['hid'][0]
            tid = data_head['tid'][0]
            hb = data_head['batch_hb'][0]
            tb = data_head['batch_tb'][0]

            best_hb, best_tb = self.find_best_sc_box(hid, tid, rid, hb, tb, hb_num, tb_num)

            sc = self.test_one_step4(data_head, best_hb, best_tb)
            hrk.append(self.lib.testHead(sc.__array_interface__["data"][0], index, type_constrain))

            sc = self.test_one_step4(data_tail, best_hb, best_tb)
            trk.append(self.lib.testTail(sc.__array_interface__["data"][0], index, type_constrain))

        print("==================Best Score Rank======================")
        self.print_rank(np.array(hrk), np.array(trk))

    # ====================================Multi-BoxE4============================================
    # 新的训练方式
    def get_htbox(self, data, rid, hb, tb, e2hb, e2tb, flag):
        batch_h = data["batch_h"]
        batch_t = data["batch_t"]

        if flag:
            # 预测头实体，尾实体固定，box都相同
            batch_tb = np.ones([len(batch_h)], dtype=int) * tb
            batch_hb = np.ones([len(batch_h)], dtype=int)
            for i in range(len(batch_h)):
                h = batch_h[i]
                if e2hb[h][rid] != -1:
                    batch_hb[i] = e2hb[h][rid]
                else:
                    batch_hb[i] = hb
        else:
            batch_hb = np.ones([len(batch_t)], dtype=int) * hb
            batch_tb = np.ones([len(batch_t)], dtype=int)
            for i in range(len(batch_t)):
                t = batch_t[i]
                if e2tb[t][rid] != -1:
                    batch_tb[i] = e2tb[t][rid]
                else:
                    batch_tb[i] = tb

        return batch_hb, batch_tb

    def run_link_prediction6(self, type_constrain=False, e2hb=None, e2tb=None):
        self.lib.initTest()
        self.data_loader.set_sampling_mode('link')
        if type_constrain:
            type_constrain = 1
        else:
            type_constrain = 0
        testing_range = tqdm(self.data_loader)

        hrk = []
        trk = []

        for index, [data_head, data_tail] in enumerate(testing_range):
            rid = data_head['batch_r'][0]
            hb_num = self.model.box_num[rid][0]
            tb_num = self.model.box_num[rid][1]
            hid = data_head['hid'][0]
            tid = data_head['tid'][0]
            hb = data_head['batch_hb'][0]
            tb = data_head['batch_tb'][0]

            best_hb, best_tb = self.find_best_sc_box(hid, tid, rid, hb, tb, hb_num, tb_num)

            hb, tb = self.get_htbox(data_head, rid, best_hb, best_tb, e2hb, e2tb, 1)
            sc = self.test_one_step4(data_head, hb, tb)
            hrk.append(self.lib.testHead(sc.__array_interface__["data"][0], index, type_constrain))

            hb, tb = self.get_htbox(data_tail, rid, best_hb, best_tb, e2hb, e2tb, 0)
            sc = self.test_one_step4(data_tail, hb, tb)
            trk.append(self.lib.testTail(sc.__array_interface__["data"][0], index, type_constrain))

        print("==================Best Score Rank======================")
        self.print_rank(np.array(hrk), np.array(trk))
