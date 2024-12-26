import argparse
import pandas as pd
import torch
from sklearn.decomposition import PCA
from utils import *
from model import *
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from s_dbw import S_Dbw
from sklearn.cluster import KMeans
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# parameter settings


if __name__ == '__main__':

    CH_list = []
    DB_list = []
    SC_list = []
    sdbw_list = []

    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=6, help='parameter k in spatial graph')
    parser.add_argument('--gnnlayers', type=int, default=4, help="Number of gnn layers")
    parser.add_argument('--linlayers', type=int, default=1, help="Number of hidden layers")
    parser.add_argument("--pca", type=int, default=300, help='parameter pca')
    parser.add_argument('--epochs', type=int, default=80, help='Number of epochs to train.')
    parser.add_argument('--dims', type=int, default=300, help='Number of units in hidden layer 1.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
    parser.add_argument('--alpha1', type=float, default=0.5, help='Loss balance parameter')
    parser.add_argument('--alpha2', type=float, default=0.8, help='Loss balance parameter')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--knn_distanceType', type=str, default='euclidean',
                        help='graph distance type: euclidean/cosine/correlation')
    parser.add_argument('--device', type=str, default='cuda:0', help='the training device')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()




    data_root = '/data/Adult_Mouse_Brain_FFPE'
    n_clusters = 20
    data_name = "Brain_FFPE"




    adata = sc.read_visium(data_root)
    adata.var_names_make_unique()
    Cal_Spatial_Net(adata, rad_cutoff=300)
    preprocess(adata)
    Stats_Spatial_Net(adata)

    if 'highly_variable' in adata.var.columns:
        adata_Vars = adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata
    data1 = Transfer_pytorch_Data(adata_Vars, dim_reduction=False)


    x = adata_Vars.X
    x = x.toarray()

    graph_adj = graph_construction(adata.obsm['spatial'],adata.shape[0],args)
    adjj = preprocess_adj(graph_adj['adj_label'].numpy())

    adj = adjj
    features = x
    best_adata = adata
    super_best_CH = 0
    i = -1

    for seed in range(10):
        setup_seed(seed)
        i += 1
        pca = PCA(n_components=args.pca)
        features = torch.FloatTensor(pca.fit_transform(features))

        adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        adj.eliminate_zeros()
        adj_tensor = torch.tensor(adj.todense(), dtype=torch.float32)

        adj_norm_s = preprocess_graph(adj, args.gnnlayers, norm='sym', renorm=True)
        sm_feature_s = sp.csr_matrix(features).toarray()
        for a in adj_norm_s:
            sm_feature_s = a.dot(sm_feature_s)
        sm_feature_s = torch.FloatTensor(sm_feature_s)

        rever_net = reversible_network([features.shape[1]])
        rever_net.to(device)
        optimizer_rever_net = optim.SGD(rever_net.parameters(), lr=args.lr, momentum=0.9, nesterov=True,weight_decay=0)
        rever_net.train()

        model = Encoder_net([features.shape[1]] + [args.dims])
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        model.train()

        # init
        best_DB = 0
        best_CH = 0
        best_SC = 0
        best_sdbw = 0

        # GPU
        if args.cuda:
            model.to(device)
            sm_feature_s = sm_feature_s.to(device)

        print('Start Training...')


        for epoch in tqdm(range(args.epochs)):

            if epoch % 100 == 0 and epoch != 0:
                optimizer.param_groups[0]['lr'] *= 0.75


            optimizer_rever_net.zero_grad()
            optimizer.zero_grad()

            aug_feature = rever_net(sm_feature_s, True)
            aug_feature = aug_feature.to(device)
            z1 = model(sm_feature_s)
            z2 = model(aug_feature)

            z11 = rever_net(z1, True)
            z22 = rever_net(z2, False)

            loss_1 = loss_cal(z1, z22)
            loss_2 = loss_cal(z2, z11)
            contra_loss = loss_1 + loss_2
            z_mat = torch.matmul(z1, z2.T)
            n1 = z1.shape[0]
            re_loss1 = F.mse_loss(z_mat, torch.eye(n1).to(device))
            z_mat1 = torch.matmul(z11, z22.T)
            n2 = z11.shape[0]
            re_loss2 = F.mse_loss(z_mat1, torch.eye(n2).to(device))
            or_pro_sim_matrix = z1 * z11
            re_or_sim_matrix = z22 * z1
            loss_semantc = F.mse_loss(or_pro_sim_matrix, re_or_sim_matrix)

            total_loss = contra_loss + args.alpha1 * loss_semantc + args.alpha2 * re_loss1 + (
                        1 - args.alpha2) * re_loss2

            total_loss.backward()
            optimizer_rever_net.step()
            optimizer.step()


            if epoch % 1 == 0:
                model.eval()
                z1 = model(sm_feature_s)
                z2  = model(aug_feature)
                hidden_emb = 0.5*z1 + 0.5*z2

                clusterings = KMeans(n_clusters=n_clusters,n_init=20)
                predict_labels = clusterings.fit_predict(hidden_emb.data.cpu().numpy())

                # label reverse
                adata.obs['nb'] = predict_labels
                cell_reps = pd.DataFrame(hidden_emb.data.cpu().numpy())
                cellss = np.array(adata.obs.index)
                cell_reps.index = cellss
                adata.obsm['nb'] = cell_reps.loc[adata.obs_names,].values


                predict_labels = list(map(int,predict_labels))

                adata.obs['nb'] = predict_labels
                X = adata.obsm['nb']
                y = adata.obs['nb']
                y = y.values.reshape(-1)



                DB = np.round(metrics.davies_bouldin_score(X,y),5)
                CH = np.round(metrics.calinski_harabasz_score(X,y),5)
                SC = np.round(metrics.silhouette_score(X,y),5)
                S_dbw = np.round(S_Dbw(X,y),5)


                if CH >= best_CH:
                    best_CH = CH
                    best_DB = DB
                    best_SC = SC
                    best_sdbw = S_dbw
                    best_embed = hidden_emb
                    adata.obs['domain_best'] = list(map(str,predict_labels))
                    cells_reps = pd.DataFrame(hidden_emb.data.cpu().numpy())
                    cellss = np.array(adata.obs.index)
                    cells_reps.index = cellss
                    adata.obsm['domain_best'] = cells_reps.loc[adata.obs_names,].values

            tqdm.write('best_CH: {}, best_DB: {}, best_SC: {}, best_sdbw: {},lr:{}'.format(best_CH, best_DB,best_SC,best_sdbw,optimizer.param_groups[0]['lr']))
        CH_list.append(best_CH)
        DB_list.append(best_DB)
        SC_list.append(best_SC)
        sdbw_list.append(best_sdbw)

        if best_CH>=super_best_CH:
            super_best_CH = best_CH
            best_adata = adata

    CH_avg = np.array(CH_list)
    DB_avg = np.array(DB_list)
    SC_avg = np.array(SC_list)
    sdbw_avg = np.array(sdbw_list)
    # save

    print("avg_CH:{},avg_DB:{},avg_SC:{},avg_sdbw:{}".format(CH_avg.mean(), DB_avg.mean(), SC_avg.mean(), sdbw_avg.mean()))
    plt.rcParams["figure.figsize"] = (3, 3)
    sc.pl.spatial(best_adata, img_key="hires", color='domain_best', s=10, show=False, title='STCGCar')
    plt.savefig(f'./result/{data_name}_clustering.jpg', bbox_inches='tight', dpi=600)






