import numpy as np
import math
from fcmeans import FCM
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
from sklearn import mixture
from scipy.stats import entropy


COV_TYPE = "diag"


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
        gmm = mixture.GaussianMixture(n_components=k, covariance_type=COV_TYPE, random_state=1)
        gmm.fit(data)
        bics.append(gmm.bic(data))

    return bics


def make_merge(X, gmm, k, current_merge):

    x_probs = my_predict_gmm_soft(X, gmm, current_merge)

    # hard_predict = x_probs.argmax(axis=1)
    if len(current_merge) > 0:
        absorbed = np.array(current_merge)[:, 1]

    else:
        absorbed = list()

    entropy_dict = dict()
    for k_aux_1 in range(k):

        for k_aux_2 in range(k_aux_1 + 1, k):

            aux_tuple = (k_aux_1, k_aux_2)

            if aux_tuple not in current_merge and k_aux_2 not in absorbed:

                # selected_lines = (hard_predict == k_aux_1) + (hard_predict == k_aux_2)
                #
                # entropy_dict[aux_tuple] = entropy(pk=x_probs[selected_lines, k_aux_1],
                #                                   qk=x_probs[selected_lines, k_aux_2])
                entropy_dict[aux_tuple] = entropy(pk=x_probs[:, k_aux_1],
                                                  qk=x_probs[:, k_aux_2])
                # entropy_dict[aux_tuple] = total_entropy(X=X, gmm=gmm, merged=merge_list + [aux_tuple])

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

        if COV_TYPE == "diag":
            cov_matrix = np.diag(gmm.covariances_[component, :])

        elif COV_TYPE == "full":
            cov_matrix = gmm.covariances_[component, :, :]

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
        # x_probs[:, merge_each[0]] = np.maximum(x_probs[:, merge_each[0]], x_probs[:, merge_each[1]])
        x_probs[:, merge_each[1]] = 0
    return x_probs


def my_plot_scatter(X, x_label):

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
        plt.contour(x_grid, y_grid, contour, levels=[0.6], colors=[color_vec(aux_k)])


def main_execution_gmm(X):

    like_vec = list()

    best_k = get_best_k_bic(X)

    print(best_k)
    gmm = mixture.GaussianMixture(n_components=best_k, covariance_type=COV_TYPE, random_state=1)
    gmm.fit(X)
    ent_list = list()
    merge_list = list()

    x_grid, y_grid = get_plot_limits(X)

    for aux in range(best_k):
        ent_list.append(total_entropy(X, gmm, merged=merge_list))

        x_labels = my_predict_gmm_hard(X, gmm, merge_list)
        my_plot_scatter(X, x_labels)

        plot_contours(x_grid, y_grid, gmm, merge_list, best_k)

        plt.title(f"Iteração {aux}")

        if aux < (best_k - 1):
            new_merge = make_merge(X, gmm, best_k, current_merge=merge_list)
            print(new_merge)
            merge_list.append(new_merge)
    plt.figure()
    plt.plot(ent_list)
    plt.title("Entropia para cada merge")
    print(ent_list)

    plt.show()


def get_plot_limits(X):
    x_min = min(X[:, 0])
    y_min = min(X[:, 1])
    x_max = max(X[:, 0])
    y_max = max(X[:, 1])
    x_grid = np.arange(x_min, x_max, 0.1)
    y_grid = np.arange(y_min, y_max, 0.1)
    return x_grid, y_grid


def get_best_k_bic(X):
    k_range = range(1, 20)
    bics = default_bic(data=X, ks=k_range)
    plt.figure()
    plt.plot(k_range, bics)
    plt.xticks(k_range)
    best_k = np.argmin(bics) + 1
    return best_k
