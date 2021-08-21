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


def make_merge(X, gmm, k, current_merge):

    x_probs = my_predict_gmm_soft(X, gmm, current_merge)

    entropy_dict = dict()
    for k_aux_1 in range(k):

        for k_aux_2 in range(k_aux_1 + 1, k):

            aux_tuple = (k_aux_1, k_aux_2)

            if aux_tuple not in current_merge:
                entropy_dict[aux_tuple] = entropy(pk=x_probs[:, k_aux_1], qk=x_probs[:, k_aux_2])

    return min(entropy_dict, key=entropy_dict.get)


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def my_predict_gmm_hard(X, gmm, merged):

    x_probs = my_predict_gmm_soft(X, gmm, merged)

    return x_probs.argmax(axis=1)


def my_inner_predict(X: np.ndarray, gmm: mixture.GaussianMixture):

    n_points = X.shape[0]
    response = np.zeros((n_points, gmm.n_components))
    N = X.shape[1]      # number of dimensions

    for component in range(gmm.n_components):
        my_mean = gmm.means_[component, :]
        cov_matrix = np.diag(gmm.covariances_[component, :])
        for point in range(n_points):
            x = X[point, :]
            aux = -0.5 * (
                np.dot(
                    np.dot(np.transpose(x - my_mean), np.linalg.inv(cov_matrix)),
                    x - my_mean
                )
            )

            response[point, component] = (
                1 / np.sqrt(
                    (2 * np.pi)**N * np.linalg.det(cov_matrix)
                ) * np.exp(aux)
            )

    return response


def my_predict_gmm_soft(X, gmm, merged) -> np.ndarray:
    # x_probs = gmm.predict_proba(X)
    x_probs = my_inner_predict(X, gmm)

    for merge_each in merged:
        x_probs[:, merge_each[0]] += x_probs[:, merge_each[1]]
        x_probs[:, merge_each[1]] = 0
    return x_probs


def my_plot_scatter(x_label):

    plt.figure()
    n_cluster = max(x_label)

    color_vec = get_cmap(n_cluster + 5)
    for label in range(n_cluster + 1):
        plot_aux = x_label == label
        plt.scatter(X[plot_aux, 0], X[plot_aux, 1], c=color_vec(label))


def total_entropy(X, gmm, merged):

    x_probs = my_predict_gmm_soft(X, gmm, merged)

    ent = 0
    for j in range(x_probs.shape[1]):

        if sum(x_probs[:, j]) > 0:

            ent += entropy(x_probs[:, j])

    return ent


def plot_contours(x_grid, y_grid, gmm, merged, best_k):

    contours = [np.zeros((len(y_grid), len(x_grid))) for _ in range(best_k)]

    for j, x in enumerate(x_grid):
        for i, y in enumerate(y_grid):

            x_probs = my_predict_gmm_soft(np.array([[x, y]]), gmm, merged)

            for aux_k, contour in enumerate(contours):

                contour[i, j] = x_probs[0, aux_k]

    color_vec = get_cmap(best_k + 5)
    for aux_k, contour in enumerate(contours):
        plt.contour(x_grid, y_grid, contour, levels=[0.9], colors=[color_vec(aux_k)])

if __name__ == '__main__':

    N = 300
    data = make_moons(n_samples=N, noise=0.05, random_state=1)
    X = data[0]
    Y = data[1]

    like_vec = list()

    bics = default_bic(data=X, ks=range(1, 20))
    # plt.plot(bics)

    best_k = np.argmin(bics)
    print(best_k)

    gmm = mixture.GaussianMixture(n_components=best_k, covariance_type='diag', random_state=1)
    gmm.fit(X)

    ent_list = list()
    merge_list = list()

    x_min = min(X[:, 0])
    y_min = min(X[:, 1])
    x_max = max(X[:, 0])
    y_max = max(X[:, 1])
    x_grid = np.arange(x_min, x_max, 0.1)
    y_grid = np.arange(y_min, y_max, 0.1)

    xx, yy = np.meshgrid(x_grid, y_grid, sparse=True)

    for aux in range(best_k):
        ent_list.append(total_entropy(X, gmm, merged=merge_list))

        x_labels = my_predict_gmm_hard(X, gmm, merge_list)
        my_plot_scatter(x_labels)

        plot_contours(x_grid, y_grid, gmm, merge_list, best_k)

        plt.title(f"Iteração {aux}")

        new_merge = make_merge(X, gmm, best_k, current_merge=merge_list)
        print(new_merge)

        if aux < (best_k - 1):

            merge_list.append(new_merge)

    plt.figure()
    plt.plot(ent_list)
    plt.title("Entropia para cada merge")
    print(ent_list)
    # fcm = FCM(n_clusters=best_k)
    # fcm.fit(X)
    #
    # x_labels = fcm.hard_predict(X)
    # plt.figure()
    # plt.scatter(X[:,0], X[:,1], c=x_labels)
    # plt.title("Fuzzy c-means")
    # plt.scatter(fcm.centers[:, 0], fcm.centers[:, 1], c="k", marker="+")

    plt.show()