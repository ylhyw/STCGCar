import numpy as np
import scipy.sparse as sp
import pandas as pd
import torch
from ot import dist
from torch_geometric.data.data import Data
import sklearn
import scanpy as sc
from scipy.spatial import distance
import networkx as nx
from kmeans import kmeans
from sklearn import metrics
import random

def refine_label(adata, radius=50, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values

    # calculate distance
    position = adata.obsm['spatial']
    distance = dist(position, position, metric='euclidean')

    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]

    return new_type

def preprocess(adata):
    sc.pp.filter_genes(adata, min_cells=3)
    # sc.pp.filter_cells(adata,min_genes=300)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)


def Transfer_pytorch_Data(adata,dim_reduction):
    G_df = adata.uns['Spatial_Net'].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)

    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])

    edgeList = np.nonzero(G)
    if not dim_reduction:
        if type(adata.X) == np.ndarray:
            data = Data(edge_index=torch.LongTensor(np.array(
                [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X))  # .todense()
        else:
            data = Data(edge_index=torch.LongTensor(np.array(
                [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X.todense()))  # .todense()

    else:
        if type(adata.obsm['X_pca']) == np.ndarray:
            data = Data(edge_index=torch.LongTensor(np.array(
                [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.obsm['X_pca'].copy()))  # .todense()
        else:
            data = Data(edge_index=torch.LongTensor(np.array(
                [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X.todense()))
    return data


from sklearn import neighbors
def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True):
    """\
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.

    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """

    assert (model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    if model == 'Radius':
        nbrs = neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it] * indices[it].shape[0], indices[it], distances[it])))

    if model == 'KNN':
        nbrs = neighbors.NearestNeighbors(n_neighbors=k_cutoff + 1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance'] > 0,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    if verbose:
        print('The graph contains %d edges, %d cells.' % (Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' % (Spatial_Net.shape[0] / adata.n_obs))

    adata.uns['Spatial_Net'] = Spatial_Net

def Stats_Spatial_Net(adata):
    import matplotlib.pyplot as plt
    Num_edge = adata.uns['Spatial_Net']['Cell1'].shape[0]
    Mean_edge = Num_edge/adata.shape[0]
    plot_df = pd.value_counts(pd.value_counts(adata.uns['Spatial_Net']['Cell1']))
    plot_df = plot_df/adata.shape[0]
    fig, ax = plt.subplots(figsize=[3,2])
    plt.ylabel('Percentage')
    plt.xlabel('')
    plt.title('Number of Neighbors (Mean=%.2f)'%Mean_edge)
    ax.bar(plot_df.index, plot_df)


def normalization(X):
    # data normalization
    X =np.array(X)
    print(X.shape)
    for i in range(len(X)):
        X[i] = X[i] / sum(X[i]) * 100000
    X = np.log2(X + 1)
    return X

def NE_dn(w, N, eps):
    w = w * N
    D = np.sum(np.abs(w), axis=1) + eps
    D = 1 / D
    D = np.diag(D)
    wn = np.dot(D, w)
    return wn

def dominateset(aff_matrix, NR_OF_KNN):
    thres = np.sort(aff_matrix)[:, -NR_OF_KNN]
    aff_matrix.T[aff_matrix.T < thres] = 0
    aff_matrix = (aff_matrix + aff_matrix.T) / 2
    return aff_matrix

def TransitionFields(W, N, eps):
    W = W * N
    W = NE_dn(W, N, eps)
    w = np.sqrt(np.sum(np.abs(W), axis=0) + eps)
    W = W / np.expand_dims(w, 0).repeat(N, 0)
    W = np.dot(W, W.T)
    return W

def getNeMatrix(W_in):
    N = len(W_in)
    K = min(20, N // 10)
    alpha = 0.9
    order = 3
    eps = 1e-20

    W0 = W_in * (1 - np.eye(N))
    W = NE_dn(W0, N, eps)
    W = (W + W.T) / 2

    DD = np.sum(np.abs(W0), axis=0)

    P = (dominateset(np.abs(W), min(K, N - 1))) * np.sign(W)
    P = P + np.eye(N) + np.diag(np.sum(np.abs(P.T), axis=0))

    P = TransitionFields(P, N, eps)

    D, U = np.linalg.eig(P)
    d = D - eps
    d = (1 - alpha) * d / (1 - alpha * d ** order)
    D = np.diag(d)
    W = np.dot(np.dot(U, D), U.T)
    W = (W * (1 - np.eye(N))) / (1 - np.diag(W))
    W = W.T

    D = np.diag(DD)
    W = np.dot(D, W)
    W[W < 0] = 0
    W = (W + W.T) / 2

    return W

def getGraph(X,K):
    # Construct cell graph
    co_matrix = np.corrcoef(X)
    X = normalization(X)
    in_matrix = np.corrcoef(X)
    NE_matrix = getNeMatrix(in_matrix)

    data = NE_matrix.reshape(-1)
    data = np.sort(data)
    data = data[:-int(len(data) * 0.02)]

    min_sh = data[0]
    max_sh = data[-1]

    delta = (max_sh - min_sh) / 100

    temp_cnt = []
    for i in range(20):
        s_sh = min_sh + delta * i
        e_sh = s_sh + delta
        temp_data = data[data > s_sh]
        temp_data = temp_data[temp_data < e_sh]
        temp_cnt.append([(s_sh + e_sh) / 2, len(temp_data)])

    candi_sh = -1
    for i in range(len(temp_cnt)):
        pear_sh, pear_cnt = temp_cnt[i]
        if 0 < i < len(temp_cnt) - 1:
            if pear_cnt < temp_cnt[i + 1][1] and pear_cnt < temp_cnt[i - 1][1]:
                candi_sh = pear_sh
                break
    if candi_sh < 0:
        for i in range(1, len(temp_cnt)):
            pear_sh, pear_cnt = temp_cnt[i]
            if pear_cnt * 2 < temp_cnt[i - 1][1]:
                candi_sh = pear_sh
    if candi_sh == -1:
        candi_sh = 0.3

    propor = len(NE_matrix[NE_matrix <= candi_sh]) / (len(NE_matrix) ** 2)
    propor = 1 - propor
    thres = np.sort(NE_matrix)[:, -int(len(NE_matrix) * propor)]
    co_matrix.T[NE_matrix.T <= thres] = 0

    up_K = np.sort(co_matrix)[:, -K]
    mat_K = np.zeros(co_matrix.shape)
    mat_K.T[co_matrix.T >= up_K] = 1
    mat_K = (mat_K+mat_K.T)/2
    mat_K[mat_K>=0.5] = 1
    W_NE = mat_K*co_matrix
    return mat_K,W_NE

def normalize_adj(adj):
    # Graph convolution
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj, loop=True):
    if loop:
        adj = adj + sp.eye(adj.shape[0])
    adj_normalized = normalize_adj(adj)
    return adj_normalized

def graph_construction(adj_coo, cell_N, params):
    adata_Adj = graph_computing(adj_coo, cell_N, params)
    graphdict = edgeList2edgeDict(adata_Adj, cell_N)
    adj_org = nx.adjacency_matrix(nx.from_dict_of_lists(graphdict))

    # Store original adjacency matrix (without diagonal entries) for later
    adj_m1 = adj_org
    adj_m1 = adj_m1 - sp.dia_matrix((adj_m1.diagonal()[np.newaxis, :], [0]), shape=adj_m1.shape)
    adj_m1.eliminate_zeros()

    # Some preprocessing
    adj_norm_m1 = preprocess_graph1(adj_m1)
    adj_label_m1 = adj_m1 + sp.eye(adj_m1.shape[0])
    adj_label_m1 = torch.FloatTensor(adj_label_m1.toarray())
    norm_m1 = adj_m1.shape[0] * adj_m1.shape[0] / float((adj_m1.shape[0] * adj_m1.shape[0] - adj_m1.sum()) * 2)

    graph_dict = {
        "adj_m1": adj_m1,
        "adj_org": adj_org,
        "adj_norm": adj_norm_m1,
        "adj_label": adj_label_m1,
        "norm_value": norm_m1
    }

    return graph_dict

def graph_computing(adj_coo, cell_num, params):
    edgeList = []
    for node_idx in range(cell_num):
        tmp = adj_coo[node_idx, :].reshape(1, -1)
        distMat = distance.cdist(tmp, adj_coo, params.knn_distanceType)
        res = distMat.argsort()[:params.k + 1]
        tmpdist = distMat[0, res[0][1:params.k + 1]]
        boundary = np.mean(tmpdist) + np.std(tmpdist)
        for j in np.arange(1, params.k + 1):
            if distMat[0, res[0][j]] <= boundary:
                weight = 1.0
            else:
                weight = 0.0
            edgeList.append((node_idx, res[0][j], weight))

    return edgeList

def edgeList2edgeDict(edgeList, nodesize):
    graphdict = {}
    tdict = {}
    for edge in edgeList:
        end1 = edge[0]
        end2 = edge[1]
        tdict[end1] = ""
        tdict[end2] = ""
        if end1 in graphdict:
            tmplist = graphdict[end1]
        else:
            tmplist = []
        tmplist.append(end2)
        graphdict[end1] = tmplist

    # check and get full matrix
    for i in range(nodesize):
        if i not in tdict:
            graphdict[i] = []

    return graphdict


def preprocess_graph(adj, layer, norm='sym', renorm=True):
    adj = sp.coo_matrix(adj)
    ident = sp.eye(adj.shape[0])
    if renorm:
        adj_ = adj + ident
    else:
        adj_ = adj

    rowsum = np.array(adj_.sum(1))

    if norm == 'sym':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        laplacian = ident - adj_normalized
    elif norm == 'left':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -1.).flatten())
        adj_normalized = degree_mat_inv_sqrt.dot(adj_).tocoo()
        laplacian = ident - adj_normalized

    reg = [1] * (layer)

    adjs = []
    for i in range(len(reg)):
        adjs.append(ident - (reg[i] * laplacian))

    return adjs

def preprocess_graph1(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def clustering(feature, true_labels, cluster_num):
    predict_labels,  initial, dis = kmeans(X=feature, num_clusters=cluster_num, distance="euclidean", device="cuda")
    # acc, nmi, ari, f1 = eva(true_labels, predict_labels.numpy(), show_details=False)
    ari = np.round(metrics.adjusted_rand_score(true_labels,predict_labels),5)
    nmi = np.round(metrics.normalized_mutual_info_score(true_labels,predict_labels),5)
    return  ari, nmi, predict_labels.numpy()

def setup_seed(seed):
    """
    setup random seed to fix the result
    Args:
        seed: random seed
    Returns: None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
