# coding:utf-8
import os
import ctypes
import numpy as np


class TestDataSampler(object):

    def __init__(self, data_total, data_sampler):
        self.data_total = data_total
        self.data_sampler = data_sampler
        self.total = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.total += 1
        if self.total > self.data_total:
            raise StopIteration()
        return self.data_sampler()

    def __len__(self):
        return self.data_total


class MyTestDataLoader(object):

    def __init__(self, in_path="./", test_file="test2id.txt", sampling_mode='link', type_constrain=True):
        base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../release/Base.so"))
        self.lib = ctypes.cdll.LoadLibrary(base_file)
        """for link prediction"""
        self.lib.getHeadBatch2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.getTailBatch2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        """set essential parameters"""
        self.in_path = in_path
        self.test_path = in_path + test_file
        self.sampling_mode = sampling_mode
        self.type_constrain = type_constrain
        self.read()

    def read(self):
        self.lib.setInPath(ctypes.create_string_buffer(self.in_path.encode(), len(self.in_path) * 2))
        self.lib.setTestPath(ctypes.create_string_buffer(self.test_path.encode(), len(self.test_path) * 2))
        self.lib.randReset()

        self.lib.importTestFiles()

        if self.type_constrain:
            self.lib.importTypeFiles()

        self.relTotal = self.lib.getRelationTotal()
        self.entTotal = self.lib.getEntityTotal()
        self.testTotal = self.lib.getTestTotal()

        self.test_h = np.zeros(self.entTotal, dtype=np.int64)
        self.test_t = np.zeros(self.entTotal, dtype=np.int64)  # 替换entTotal次
        self.test_r = np.zeros(1, dtype=np.int64)  # 关系都是固定的，只需要 1 就行
        self.test_h_addr = self.test_h.__array_interface__["data"][0]
        self.test_t_addr = self.test_t.__array_interface__["data"][0]
        self.test_r_addr = self.test_r.__array_interface__["data"][0]

        self.test_hb = np.zeros(1, dtype=np.int64)  # head/tail box 确定的情况才是 1
        self.test_tb = np.zeros(1, dtype=np.int64)
        self.test_hb_addr = self.test_hb.__array_interface__["data"][0]
        self.test_tb_addr = self.test_tb.__array_interface__["data"][0]

        self.hid = np.zeros(1, dtype=np.int64)
        self.tid = np.zeros(1, dtype=np.int64)
        self.hid_addr = self.hid.__array_interface__["data"][0]
        self.tid_addr = self.tid.__array_interface__["data"][0]

    def sampling_lp(self):
        """
        :return: 返回一个triple对应的所有头、尾替换
        """
        res = []
        self.lib.getHeadBatch2(self.test_h_addr, self.test_t_addr, self.test_r_addr, self.test_hb_addr,
                               self.test_tb_addr, self.hid_addr, self.tid_addr)
        # 每次针对一个测试triple，返回它的所有替换
        res.append({
            "batch_h": self.test_h.copy(),  # (ent_tot,)
            # 只取[:1]一个，因为只是替h，而r和t都是相同的
            "batch_t": self.test_t[:1].copy(),  # (1,)
            "batch_r": self.test_r[:1].copy(),  # (1,)

            "batch_hb": self.test_hb[:1].copy(),
            "batch_tb": self.test_tb[:1].copy(),
            "hid": self.hid[:1].copy(),
            "tid": self.tid[:1].copy(),
            "mode": "head_batch"
        })
        self.lib.getTailBatch2(self.test_h_addr, self.test_t_addr, self.test_r_addr, self.test_hb_addr,
                               self.test_tb_addr, self.hid_addr, self.tid_addr)
        res.append({
            "batch_h": self.test_h[:1],
            "batch_t": self.test_t,
            "batch_r": self.test_r[:1],

            "batch_hb": self.test_hb[:1],
            "batch_tb": self.test_tb[:1],
            "hid": self.hid[:1],
            "tid": self.tid[:1],
            "mode": "tail_batch"
        })
        return res

    """interfaces to get essential parameters"""

    def get_ent_tot(self):
        return self.entTotal

    def get_rel_tot(self):
        return self.relTotal

    def get_triple_tot(self):
        return self.testTotal

    def set_sampling_mode(self, sampling_mode):
        self.sampling_mode = sampling_mode

    def __len__(self):
        return self.testTotal

    def __iter__(self):
        if self.sampling_mode == "link":
            self.lib.initTest()
            return TestDataSampler(self.testTotal, self.sampling_lp)
