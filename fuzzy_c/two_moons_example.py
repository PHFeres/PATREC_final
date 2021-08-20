import numpy as np
import math
from fcmeans import FCM
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
from sklearn import mixture
from scipy.stats import entropy

def log_likelihood(x_cluster):

    n = x_cluster.shape[0]

    my_mean = np.mean(x_cluster, axis=0)
    my_sd = np.var(x_cluster, axis=0)

    return (
        -(0.5 * n * math.log(2 * math.pi)) - (0.5 * n * 1)
    )


def default_bic(data, ks):

    bics = list()

    for k in ks:
        gmm = mixture.GaussianMixture(n_components=k, covariance_type='diag', random_state=1)
        gmm.fit(data)
        bics.append(gmm.bic(data))

    return bics


def make_merge(X, gmm, k):

    x_probs= gmm.predict_proba(X)

    entropy_dict = dict()
    for k_aux_1 in range(k):

        for k_aux_2 in range(k_aux_1 + 1, k):

            entropy_dict[(k_aux_1, k_aux_2)] = entropy(pk=x_probs[:, k_aux_1], qk=x_probs[:, k_aux_2])

    return min(entropy_dict, key=entropy_dict.get)


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def my_predict_gmm_hard(X, gmm, merged):

    x_probs = my_predict_gmm_soft(X, gmm, merged)

    return x_probs.argmax(axis=1)


def my_predict_gmm_soft(X, gmm, merged):
    x_probs = gmm.predict_proba(X)
    for merge_each in merged:
        x_probs[:, merge_each[0]] += x_probs[:, merge_each[1]]
    return x_probs


def my_plot_scatter(x_label):

    plt.figure()
    n_cluster = max(x_label)

    color_vec = get_cmap(n_cluster + 5)
    for label in range(n_cluster + 1):
        plot_aux = x_label == label
        plt.scatter(X[plot_aux, 0], X[plot_aux, 1], c=color_vec(label))


def total_entropy(X, gmm, merged):
    pass

if __name__ == '__main__':

    N = 300
    data = make_moons(n_samples=N, noise=0.1, random_state=1)
    X = data[0]
    Y = data[1]

    like_vec = list()

    bics = default_bic(data=X, ks=range(1, 20))
    plt.plot(bics)

    best_k = np.argmin(bics)
    print(best_k)

    gmm = mixture.GaussianMixture(n_components=best_k, covariance_type='diag', random_state=1)
    gmm.fit(X)

    x_labels = gmm.predict(X)
    my_plot_scatter(x_labels)
    plt.title("Mistura clássica")

    my_merge = make_merge(X, gmm, best_k)
    print(my_merge)

    x_labels2 = my_predict_gmm_hard(X, gmm, [my_merge])

    my_plot_scatter(x_labels2)

    plt.title("Mistura clássica - v1")

    # fcm = FCM(n_clusters=best_k)
    # fcm.fit(X)
    #
    # x_labels = fcm.hard_predict(X)
    # plt.figure()
    # plt.scatter(X[:,0], X[:,1], c=x_labels)
    # plt.title("Fuzzy c-means")
    # plt.scatter(fcm.centers[:, 0], fcm.centers[:, 1], c="k", marker="+")

    plt.show()