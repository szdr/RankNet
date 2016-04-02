from chainer import Chain
import chainer.functions as F
import chainer.links as L


class MLP(Chain):

    def __init__(self, n_in, n_hidden):
        super(MLP, self).__init__(
            l1=L.Linear(n_in, n_hidden),
            l2=L.Linear(n_hidden, n_hidden),
            l3=L.Linear(n_hidden, 1)
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


class RankNet(Chain):

    def __init__(self, predictor):
        super(RankNet, self).__init__(predictor=predictor)

    def __call__(self, x_i, x_j, t_i, t_j):
        s_i = self.predictor(x_i)
        s_j = self.predictor(x_j)
        s_diff = s_i - s_j
        if t_i.data > t_j.data:
            S_ij = 1
        elif t_i.data < t_j.data:
            S_ij = -1
        else:
            S_ij = 0
        self.loss = (1 - S_ij) * s_diff / 2. + F.log(1 + F.exp(-s_diff))
        return self.loss
