import matplotlib.pyplot as plt
import numpy as np
from fcmeans import FCM
from scipy.stats import entropy
from sklearn.datasets import make_moons

from fuzzy_c.my_lib import get_best_k_bic, my_plot_scatter, get_cmap, get_plot_limits


def my_predict_fuzzy_soft(X, fcm, merged):

    x_probs = fcm.soft_predict(X)

    for merge_each in merged:
        x_probs[:, merge_each[0]] += x_probs[:, merge_each[1]]
        # x_probs[:, merge_each[0]] = np.maximum(x_probs[:, merge_each[0]], x_probs[:, merge_each[1]])
        x_probs[:, merge_each[1]] = 0
    return x_probs

def make_merge_fuzzy(X, fcm, k, current_merge):

    x_probs = my_predict_fuzzy_soft(X, fcm, current_merge)

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

def plot_contours_fuzzy(x_grid, y_grid, fcm, merged, best_k):

    contours = [np.zeros((len(y_grid), len(x_grid))) for _ in range(best_k)]

    for j, x in enumerate(x_grid):
        for i, y in enumerate(y_grid):

            x_probs = fcm.soft_predict(np.array([[x, y]]))

            for aux_k, contour in enumerate(contours):

                contour[i, j] = x_probs[0, aux_k]

    color_vec = get_cmap(best_k + 5)
    for aux_k, contour in enumerate(contours):
        plt.contour(x_grid, y_grid, contour, levels=[0.5], colors=[color_vec(aux_k)])


if __name__ == '__main__':

    N = 300
    data = make_moons(n_samples=N, noise=0.05, random_state=1)
    X = data[0]
    Y = data[1]

    best_k = get_best_k_bic(X)

    fcm = FCM(n_clusters=best_k)
    fcm.fit(X)

    x_grid, y_grid = get_plot_limits(X)
    merge_list = list()

    fcm_labels = fcm.hard_predict(X)
    my_plot_scatter(X, fcm_labels)
    plot_contours_fuzzy(x_grid, y_grid, fcm, merged=merge_list, best_k=best_k)
    plt.title("Iteração 0")

    merge_list.append(
        make_merge_fuzzy(X, fcm, best_k, current_merge=merge_list)
    )

    print(merge_list)
    fcm_labels = fcm.hard_predict(X)
    my_plot_scatter(X, fcm_labels)
    plot_contours_fuzzy(x_grid, y_grid, fcm, merged=merge_list, best_k=best_k)
    plt.title("Iteração 1")

    plt.show()