from sklearn.datasets import make_moons
from fuzzy_c.my_lib import main_execution_gmm

if __name__ == '__main__':

    N = 300
    data = make_moons(n_samples=N, noise=0.05, random_state=1)
    X = data[0]
    Y = data[1]

    main_execution_gmm(X)
