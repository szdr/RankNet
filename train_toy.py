import numpy as np
from chainer import Variable, optimizers
from sklearn.cross_validation import train_test_split
import net
import matplotlib.pyplot as plt
import seaborn as sns


# y ~ [1, n_rank] x ~ N(x|w * y, sigma)
def make_dataset(n_dim, n_rank, n_sample, sigma):
    ys = np.random.random_integers(n_rank, size=n_sample)
    w = np.random.randn(n_dim)
    X = [sigma * np.random.randn(n_dim) + w * y for y in ys]
    X = np.array(X).astype(np.float32)
    ys = np.reshape(np.array(ys), (-1, 1))
    return X, ys


def ndcg(y_true, y_score, k=100):
    y_true = y_true.ravel()
    y_score = y_score.ravel()
    y_true_sorted = sorted(y_true, reverse=True)
    ideal_dcg = 0
    for i in range(k):
        ideal_dcg += (2 ** y_true_sorted[i] - 1.) / np.log2(i + 2)
    dcg = 0
    argsort_indices = np.argsort(y_score)[::-1]
    for i in range(k):
        dcg += (2 ** y_true[argsort_indices[i]] - 1.) / np.log2(i + 2)
    ndcg = dcg / ideal_dcg
    return ndcg

if __name__ == '__main__':
    np.random.seed(0)
    n_dim = 50
    n_rank = 5
    n_sample = 1000
    sigma = 5.
    X, ys = make_dataset(n_dim, n_rank, n_sample, sigma)
    X_train, X_test, y_train, y_test = train_test_split(X, ys, test_size=0.33)

    n_iter = 2000
    n_hidden = 20
    loss_step = 50
    N_train = np.shape(X_train)[0]

    model = net.RankNet(net.MLP(n_dim, n_hidden))
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    N_train = np.shape(X_train)[0]
    train_ndcgs = []
    test_ndcgs = []
    for step in range(n_iter):
        i, j = np.random.randint(N_train, size=2)
        x_i = Variable(X_train[i].reshape(1, -1))
        x_j = Variable(X_train[j].reshape(1, -1))
        y_i = Variable(y_train[i])
        y_j = Variable(y_train[j])
        model.zerograds()
        loss = model(x_i, x_j, y_i, y_j)
        loss.backward()
        optimizer.update()
        if (step + 1) % loss_step == 0:
            train_score = model.predictor(Variable(X_train))
            test_score = model.predictor(Variable(X_test))
            train_ndcg = ndcg(y_train, train_score.data)
            test_ndcg = ndcg(y_test, test_score.data)
            train_ndcgs.append(train_ndcg)
            test_ndcgs.append(test_ndcg)
            print("step: {}".format(step + 1))
            print("NDCG@100 | train: {}, test: {}".format(
                train_ndcg, test_ndcg))

    sns.set_context("poster")
    plt.plot(train_ndcgs, label="Train")
    plt.plot(test_ndcgs, label="Test")
    xx = np.linspace(0, n_iter / loss_step, num=n_iter / loss_step + 1)
    labels = np.arange(loss_step, n_iter + 1, loss_step)
    plt.xticks(xx, labels, rotation=45)
    plt.legend(loc="best")
    plt.xlabel("step")
    plt.ylabel("NDCG@100")
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.show()
