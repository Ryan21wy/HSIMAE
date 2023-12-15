import numpy as np
from sklearn.decomposition import PCA


def split_pca(data_list, pca_data_list=None, n_components=16, whiten=True):
    split_data_list = []
    if pca_data_list is None:
        pca_data_list = []
    for data in data_list:
        n, c = data.shape
        data_s1 = data[:, :c // 2]
        data_s2 = data[:, c // 2:]
        split_data_list.append(data_s1)
        split_data_list.append(data_s2)

        pca_1 = PCA(n_components=n_components, whiten=whiten)
        pca_data_s1 = pca_1.fit_transform(data_s1)
        pca_data_list.append(pca_data_s1)

        pca_2 = PCA(n_components=n_components, whiten=whiten)
        pca_data_s2 = pca_2.fit_transform(data_s2)
        pca_data_list.append(pca_data_s2)
    return split_data_list, pca_data_list


def applyMSPCA(X, numComponents=64, step=4, whiten=True):
    newX = np.reshape(X, (-1, X.shape[2]))
    newX = (newX - newX.min()) / (np.max(newX) - np.min(newX))

    n_c_0 = numComponents // step
    pca = PCA(n_components=n_c_0, whiten=whiten)
    X_p = pca.fit_transform(newX)

    X_ps = [X_p]
    newX = [newX]
    for i in range(step - 1):
        n_c = n_c_0 // (2 ** (i + 1))
        newX, X_ps = split_pca(newX, X_ps, n_components=n_c, whiten=whiten)
    out = np.concatenate(X_ps, axis=-1)
    out = np.reshape(out, (X.shape[0], X.shape[1], -1))
    return out