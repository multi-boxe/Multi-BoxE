from .BaseModule import BaseModule


class NegativeSampling(BaseModule):

    def __init__(self, model=None, loss=None, batch_size=256, regul_rate=0.0, l3_regul_rate=0.0):
        super(NegativeSampling, self).__init__()
        self.model = model
        self.loss = loss
        self.batch_size = batch_size
        self.regul_rate = regul_rate
        self.l3_regul_rate = l3_regul_rate

    def _get_positive_score(self, score):
        positive_score = score[:self.batch_size]
        # 转为[-1,batch size]维，然后0,1维度互换
        positive_score = positive_score.view(-1, self.batch_size).permute(1, 0)  # (1, batch)
        return positive_score

    def _get_negative_score(self, score):
        negative_score = score[self.batch_size:]
        negative_score = negative_score.view(-1, self.batch_size).permute(1, 0)  # (1, neg_num * batch)
        return negative_score

    def forward(self, data):
        score = self.model(data)  # 一个batch的得分
        p_score = self._get_positive_score(score)
        n_score = self._get_negative_score(score)
        loss_res = self.loss(p_score, n_score)
        if self.regul_rate != 0:
            loss_res += self.regul_rate * self.model.regularization(data)
        if self.l3_regul_rate != 0:
            loss_res += self.l3_regul_rate * self.model.l3_regularization()
        return loss_res
