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
import copy
from tqdm import tqdm


class MyTrainer(object):

    def __init__(self,
                 model=None,
                 data_loader=None,
                 train_times=1000,
                 alpha=0.5,
                 use_gpu=True,
                 opt_method="sgd",
                 save_steps=None,
                 checkpoint_dir=None,
                 train_method=0,
                 e2hb=None,
                 e2tb=None):

        self.work_threads = 8
        self.train_times = train_times

        self.opt_method = opt_method
        self.optimizer = None
        self.lr_decay = 0
        self.weight_decay = 0
        self.alpha = alpha

        self.model = model
        self.data_loader = data_loader
        self.use_gpu = use_gpu
        self.save_steps = save_steps
        self.checkpoint_dir = checkpoint_dir
        self.train_method = train_method
        self.e2hb = e2hb
        self.e2tb = e2tb

    def train_one_step(self, data):
        self.optimizer.zero_grad()
        loss = self.model({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': data['batch_r'],

            'batch_hb': data['batch_hb'],
            'batch_tb': data['batch_tb'],
            'mode': data['mode']
        })
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def run(self):
        if self.use_gpu:
            self.model.cuda()

        if self.optimizer != None:
            pass
        elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
            self.optimizer = optim.Adagrad(
                self.model.parameters(),
                lr=self.alpha,
                lr_decay=self.lr_decay,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
            self.optimizer = optim.Adadelta(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method == "Adam" or self.opt_method == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        print("Finish initializing...")

        training_range = tqdm(range(self.train_times))
        for epoch in training_range:
            res = 0.0
            for data in self.data_loader:
                # 每个data是一个TrainDataSampler
                if self.train_method:
                    data = self.func1(data, self.e2hb, self.e2tb)
                loss = self.train_one_step(data)
                res += loss
            training_range.set_description("Epoch %d | loss: %f" % (epoch, res))

            if self.save_steps and self.checkpoint_dir and (epoch + 1) % self.save_steps == 0:
                print("Epoch %d has finished, saving..." % epoch)
                self.model.save_checkpoint(os.path.join(self.checkpoint_dir + "-" + str(epoch) + ".ckpt"))

    def func1(self, data, e2hb, e2tb):
        batch_h = data["batch_h"]
        batch_t = data["batch_t"]
        batch_r = data["batch_r"]
        batch_hb = data["batch_hb"]
        batch_tb = data["batch_tb"]

        batch_size = self.data_loader.get_batch_size()

        for i in range(batch_size, len(batch_h)):
            h = batch_h[i]
            t = batch_t[i]
            r = batch_r[i]

            if e2hb[h][r] != -1:
                batch_hb[i] = e2hb[h][r]
            if e2tb[t][r] != -1:
                batch_tb[i] = e2tb[t][r]

        data["batch_hb"] = batch_hb
        data["batch_tb"] = batch_tb

        return data

    def set_model(self, model):
        self.model = model

    def to_var(self, x, use_gpu):
        # 把numpy数组转为tensor
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_lr_decay(self, lr_decay):
        self.lr_decay = lr_decay

    def set_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay

    def set_opt_method(self, opt_method):
        self.opt_method = opt_method

    def set_train_times(self, train_times):
        self.train_times = train_times

    def set_save_steps(self, save_steps, checkpoint_dir=None):
        self.save_steps = save_steps
        if not self.checkpoint_dir:
            self.set_checkpoint_dir(checkpoint_dir)

    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
