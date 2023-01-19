import torch
import torch.nn as nn
from .BaseModule import BaseModule


class BoxELoss(BaseModule):
    def __init__(self, gamma, w):
        """
        :param gamma: gamma loss margin
        :param w: hyper_parameter, corresponds to 1/k in RotatE paper
        """
        super(BoxELoss, self).__init__()
        # self.criterion = nn.LogSigmoid()

        self.eps = torch.finfo(torch.float32).tiny
        self.gamma = nn.Parameter(torch.Tensor([gamma]))
        self.gamma.requires_grad = False

        # 因为p_score是一个batch的正样本，所以负样本的个数就是neg_num * batch_size（错误）
        self.w = nn.Parameter(torch.Tensor([w]))
        self.w.requires_grad = False

    def forward(self, p_score, n_score):
        # (1, batch)
        s1 = - torch.log(torch.sigmoid(self.gamma - p_score) + self.eps)

        # (1, neg_num * batch)
        s2 = torch.sum(self.w * torch.log(torch.sigmoid(n_score - self.gamma) + self.eps), dim=1, keepdim=True)

        # s1 = - torch.sum(torch.log(torch.sigmoid(self.gamma - p_score) + self.eps))
        # s2 = self.w * torch.sum(torch.log(torch.sigmoid(n_score - self.gamma) + self.eps))
        return torch.sum(s1 - s2)

    def predict(self, p_score, n_score):
        score = self.forward(p_score, n_score)
        return score.cpu().data.numpy()
