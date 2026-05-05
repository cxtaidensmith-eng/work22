import os

os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp
from collections import Counter
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import SpectralClustering
from scipy.spatial.distance import squareform
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from scipy.stats import mode


def feature_to_adj(feature, k_nearest_neighobrs=7, rm_common_neighbors=2):
    nbrs = NearestNeighbors(n_neighbors=k_nearest_neighobrs + 1, algorithm='ball_tree').fit(feature)
    adj_wave = nbrs.kneighbors_graph(feature)

    np_adj_wave = construct_symmetric_matrix(adj_wave.toarray())
    adj = sp.csc_matrix(np_adj_wave)

    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    adj = adj.toarray()

    b = np.nonzero(adj)
    rows = b[0]
    cols = b[1]
    dic = {}
    for row, col in zip(rows, cols):
        if row in dic.keys():
            dic[row].append(col)
        else:
            dic[row] = []
            dic[row].append(col)
    for row, col in zip(rows, cols):
        if len(set(dic[row]) & set(dic[col])) < rm_common_neighbors:
            adj[row][col] = 0
    adj = sp.csc_matrix(adj)
    adj.eliminate_zeros()

    adj_w = adj.toarray() + sp.eye(adj.shape[0])

    return adj_w


def construct_symmetric_matrix(original_matrix):
    """
        transform a matrix (n*n) to be symmetric
    :param np_matrix: <class 'numpy.ndarray'>
    :return: result_matrix: <class 'numpy.ndarray'>
    """
    result_matrix = np.zeros(original_matrix.shape, dtype=float)
    num = original_matrix.shape[0]
    for i in range(num):
        for j in range(num):
            if original_matrix[i][j] == 0:
                continue
            elif original_matrix[i][j] == 1:
                result_matrix[i][j] = 1
                result_matrix[j][i] = 1
            else:
                print("The value in the original matrix is illegal!")
    assert (result_matrix == result_matrix.T).all() == True

    if ~(np.sum(result_matrix, axis=1) > 1).all():
        print("There existing a outlier!")

    return result_matrix


def _pearson_correlation(data):
    # 计算每列的标准差
    std_devs = np.std(data, axis=0)

    # 加上一个极小的值以避免标准差为零的情况
    epsilon = 1e-10
    std_devs[std_devs == 0] = epsilon

    # 手动计算相关性矩阵
    corr_matrix = np.empty((data.shape[1], data.shape[1]))
    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                cov = np.cov(data[:, i], data[:, j])[0, 1]
                # corr_matrix[i, j] = cov / (std_devs[i] * std_devs[j])
                corr_value = cov / (std_devs[i] * std_devs[j])

                corr_matrix[i, j] = np.clip(corr_value, -1.0, 1.0)

    return corr_matrix


def _k_means(Feature, n_clusters, stabilization=True):
    if stabilization:
        n_runs = min(Feature.shape[0] // n_clusters, 100)
        all_labels = np.zeros((Feature.shape[0], n_runs))  # 用于存储每次聚类的结果

        # 多次运行聚类算法
        for i in range(n_runs):
            kmeans = KMeans(n_clusters=n_clusters, random_state=i)  # 每次使用不同的随机种子
            all_labels[:, i] = kmeans.fit_predict(Feature)
        # 统计每个点的模式簇
        final_labels = mode(all_labels, axis=1)[0].flatten()

        return final_labels

    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(Feature)
        labels = kmeans.labels_

        return labels


def hertergraph_kmean(DATASET_Dict, Feature_T, k=2, remove_self_loop=True, remove_repeat=True, base='correlation', distance='cosine', use='Spectral'):
    '''
    base = similarity correlation

    if base = similarity
        distance = euclidean  cosine
        use only support Kmeans

    use = Kmeans Hierarchical Spectral
    '''

    if base == 'similarity':

        if distance == 'euclidean':
            Feature_use = Feature_T
        elif distance == 'cosine':
            Feature_use = normalize(Feature_T, norm='l2')
        else:
            # default euclidean
            Feature_use = Feature_T

        labels = _k_means(Feature_use, k)

    elif base == 'correlation':

        corr_matrix = _pearson_correlation(Feature_T.T)

        if use == 'Kmeans':

            similarity_matrix = np.abs(corr_matrix)
            pca = PCA(n_components=k)  # 选取适当的降维维度
            reduced_features = pca.fit_transform(similarity_matrix)

            labels = _k_means(reduced_features, k)

        elif use == 'Hierarchical':

            distance_matrix = 1 - np.abs(corr_matrix)
            distance_array = squareform(distance_matrix, checks=False)

            Z = linkage(distance_array, method='average')
            # max_distance = 0.5  # 设定截断距离，调整以获取不同的簇数
            # labels = fcluster(Z, max_distance, criterion='distance')
            # # 方法 2: 直接设定期望簇数
            labels = fcluster(Z, k, criterion='maxclust')

        elif use == 'Spectral':
            similarity_matrix = np.abs(corr_matrix)

            threshold = 0.1  # 设定一个适当的阈值
            similarity_matrix[similarity_matrix < threshold] = threshold

            spectral = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=0)
            labels = spectral.fit_predict(similarity_matrix)

        else:
            # default Spectral

            similarity_matrix = np.abs(corr_matrix)

            threshold = 0.1  # 设定一个适当的阈值
            similarity_matrix[similarity_matrix < threshold] = threshold

            spectral = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=0)
            labels = spectral.fit_predict(similarity_matrix)

    else:
        # default similarity euclidean
        Feature_use = Feature_T
        labels = _k_means(Feature_use, k)


    # 创建一个字典来存储每个簇内的样本索引
    clusters = {i: np.where(labels == i)[0].tolist() for i in range(k)}

    # print("每个簇内的样本索引: ", clusters)

    kmeans_modal_Counter = []
    kmeans_modal = []
    for key, values in clusters.items():
        modal_name_ = []
        merged_dict = {}
        for i in values:
            modal_name = DATASET_Dict['Modal_Name'][[index_ for index_, list_ in enumerate(DATASET_Dict['Modal_Index']) if i in list_][0]]
            modal_name_.append(modal_name)

            if modal_name in merged_dict:
                merged_dict[modal_name].append(i)
            else:
                merged_dict[modal_name] = [i]

        edge_modal = create_modal_edges(list(merged_dict.keys()), remove_self_loop=remove_self_loop, remove_repeat=remove_repeat)
        kmeans_modal.append([merged_dict, edge_modal])
        kmeans_modal_Counter.append(dict(sorted(Counter(modal_name_).items(), key=lambda item: item[1], reverse=True)))

    for index, i in enumerate(kmeans_modal_Counter):
        print(index, i)
    
    return kmeans_modal


def create_modal_edges(modalities, remove_self_loop=True, remove_repeat=True):
    if len(modalities) == 1:
        return [[modalities[0], modalities[0]]]
    else:
        edges = [[modality1, modality2] for modality1 in modalities for modality2 in modalities]
        if remove_self_loop:
            edges = [edge for edge in edges if edge[0] != edge[1]]

        if remove_repeat:
            unique_pairs = set()
            for edge in edges:
                sorted_edge = tuple(sorted(edge))
                unique_pairs.add(sorted_edge)
            edges = [list(pair) for pair in unique_pairs]

        return edges
    

def split_hertergraph(DATASET_Dict, DATASET_DATA, CONFIG, K=2, remove_self_loop=True, remove_repeat=True):
    Split_hertergraph = []
    for Split_index in range(len(DATASET_DATA['Mask'])):
        DATA = DATASET_DATA['Feature']
        LABEL = DATASET_DATA['Label']
        Train_Mask = DATASET_DATA['Mask'][Split_index][0]

        Tranin_Data = DATA[Train_Mask]
        Train_Label = LABEL[Train_Mask]

        print('='*15 + f'Split:{Split_index}'+ '='*15)

        Feature_np = Tranin_Data.cpu().numpy()
        Label_np = Train_Label.cpu().numpy()

        Herter_graph_list = []
        for i in range(DATASET_Dict['Class_Num']):
            print('='*10 + f'label:{i}'+ '='*10)
            Herter_graph_list.append(hertergraph_kmean(DATASET_Dict, Feature_T=Feature_np[Label_np == i].T, k=K, remove_self_loop=remove_self_loop, remove_repeat=remove_repeat, base=CONFIG.G_base, distance=CONFIG.G_distance, use=CONFIG.G_use))  

        Split_hertergraph.append(Herter_graph_list)
        
    return Split_hertergraph
