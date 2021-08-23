import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from fuzzy_c.my_lib import main_execution_gmm

if __name__ == '__main__':

    N = 1000
    data = make_circles(n_samples=N, random_state=1, noise=0.05, factor=0.5)
    X = data[0]
    Y = data[1]

    main_execution_gmm(X)
