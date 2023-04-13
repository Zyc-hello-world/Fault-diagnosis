"""
谱聚类
"""

import numpy as np
from sklearn.cluster import KMeans


def euclidDistance(x1, x2, sqrt_flag=False):
    res = np.sum((x1-x2)**2)
    if sqrt_flag:
        res = np.sqrt(res)
    return res


# In[3]:


def calDistanceMatrix(X):
    X = np.array(X)
    S = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            S[i][j] = 1.0 * euclidDistance(X[i], X[j])
            S[j][i] = S[i][j]
    return S


# # 由欧式距离得到邻接矩阵(反映了图中各结点之间的相似性)，使用KNN算法

# In[4]:


def myKNN(S, k, sigma=1.0):
    N = len(S)
    A = np.zeros((N,N))

    for i in range(N):
        dist_with_index = zip(S[i], range(N))
        dist_with_index = sorted(dist_with_index, key=lambda x:x[0])
        neighbours_id = [dist_with_index[m][1] for m in range(k+1)] # xi's k nearest neighbours

        for j in neighbours_id: # xj is xi's neighbour
            A[i][j] = np.exp(-S[i][j]/2/sigma/sigma)
            A[j][i] = A[i][j] # mutually

    return A


# # 标准化的拉普拉斯矩阵

# In[5]:


def calLaplacianMatrix(adjacentMatrix):
    """
    return :拉普拉斯矩阵
    """

    # compute the Degree Matrix: D=sum(A)
    degreeMatrix = np.sum(adjacentMatrix, axis=1)

    # compute the Laplacian Matrix: L=D-A
    laplacianMatrix = np.diag(degreeMatrix) - adjacentMatrix

    # normailze
    # L -> D^(-1/2) L D^(-1/2)
    sqrtDegreeMatrix = np.diag(1.0 / (degreeMatrix ** (0.5)))
    return np.dot(np.dot(sqrtDegreeMatrix, laplacianMatrix), sqrtDegreeMatrix)


# In[6]:


def spKmeans(H):
    """
    对特征向量H进行kmeans聚类
    return:标签列表
    """
    sp_kmeans = KMeans(n_clusters=2).fit(H)
    return sp_kmeans.labels_


def get_labels(fault_pair_data):
    """
    返回一对故障的预测标签
    """
    S = calDistanceMatrix(fault_pair_data)
    A = myKNN(S, 3)
    lacp = calLaplacianMatrix(A)
    lam, H = np.linalg.eig(lacp)
    H = H.real.astype(np.float32)
    labels = spKmeans(H)
    return labels