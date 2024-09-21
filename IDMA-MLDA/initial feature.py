import numpy as np


# example
drug_nodes = 463
drug_dim = 256
microbe_nodes = 234
microbe_dim = 256
low_limit = 0.0
high_limit = 1.0
drug_initial_feat = np.random.uniform(low=low_limit, high=high_limit, size=(drug_nodes, drug_dim))
microbe_initial_feat = np.random.uniform(low=low_limit, high=high_limit, size=(microbe_nodes, microbe_dim))
