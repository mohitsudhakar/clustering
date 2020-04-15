import numpy as np
import scipy as sc
from scipy.sparse.linalg import eigs
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
import networkx as nx

true_labels = np.array([1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,0,1,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0])

def k_means(X, k):
    return KMeans(n_clusters=k).fit(X).labels_

def spectral_clustering(A, k):
    d = np.sum(A, axis=0)
    d = np.array(d)
    if len(d.shape) == 2:
        d = d[0]
    D = np.diag(d)
    D = np.sqrt(np.linalg.inv(D))
    L = D.dot(A).dot(D)

    #  Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigs(L, k)
    X = eigenvectors.real
    # rows_norm = sc.linalg.norm(X, axis=1, ord=2)
    # Y = (X.T / rows_norm).T
    labels = k_means(X, k)
    return labels

if __name__ == '__main__':
    G = nx.karate_club_graph()
    adj = nx.to_numpy_matrix(G)
    pred_labels = spectral_clustering(adj, 2)
    f1 = f1_score(true_labels, pred_labels)
    if f1 < 0.5:
        # this is done since in binary it can go either way.
        true_labels = [1-x for x in true_labels]
        f1 = f1_score(true_labels, pred_labels)
    # print('true', true_labels)
    c1, c2 = [], []
    for i in range(0, len(pred_labels)):
        if pred_labels[i] == 0:
            c1.append(i+1)
        else:
            c2.append(i+1)
    print('Communities are C1 =', c1, ', C2 =', c2, '}')
    print("F1 score", f1)