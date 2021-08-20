import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, mixture

# Create some random clustered data in 2 dimensions
n_clusters = 3
n_features = 2
n_samples = 500

data = datasets.make_blobs(n_samples=n_samples,
                           n_features=n_features,
                           centers=n_clusters)[0]

# Fit the data using a Gaussian mixture model and calculate the BIC for
# different numbers of clusters, ranging for 1 to 10
ks = np.arange(1, 11)
bics = []
for k in ks:
    gmm = mixture.GaussianMixture(n_components=k, covariance_type='diag')
    gmm.fit(data)
    bics.append(gmm.bic(data))

# Plot the data
fig, ax = plt.subplots()
ax.plot(ks, bics)
ax.set_xlabel(r'Number of clusters, $k$')
ax.set_ylabel('BIC')
ax.set_xticks(ks)
plt.show()