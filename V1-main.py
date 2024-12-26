import argparse
import pandas as pd
import torch
from sklearn.decomposition import PCA
from utils import *
from model import *
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# parameter settings


if __name__ == '__main__':

    ARI_list = []
    NMI_list = []

    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=6, help='parameter k in spatial graph')
    parser.add_argument('--gnnlayers', type=int, default=5, help="Number of gnn layers")
    parser.add_argument('--linlayers', type=int, default=1, help="Number of hidden layers")
    parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
    parser.add_argument('--dims', type=int, default=300, help='Number of units in hidden layer 1.')
    parser.add_argument("--pca", type=int, default=300, help='parameter pca')
    parser.add_argument('--lr', type=float, default=1e-5, help='Initial learning rate.')
    parser.add_argument('--alpha1', type=float, default=0.5, help='Loss balance parameter')
    parser.add_argument('--alpha2', type=float, default=0.8,
                        help='Loss balance parameter')
    parser.add_argument('--no-cuda', action='store_true', default=False,help='Disables CUDA training.')
    parser.add_argument('--knn_distanceType', type=str, default='euclidean',
                        help='graph distance type: euclidean/cosine/correlation')
    parser.add_argument('--device', type=str, default='cuda:1', help='the training device')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    data_root = '/data/V1_Breast_Cancer_Block_A_Section_1'


    data_name = 'V1_Breast_Cancer_Block_A_Section_1'


    n_clusters = 20


    adata = sc.read_visium(data_root)
    adata.var_names_make_unique()

    df_meta = pd.read_csv(data_root + f'/metadata.tsv', sep='\t')
    result = pd.DataFrame(df_meta['fine_annot_type'])
    adata.obs['ground_truth'] = result['fine_annot_type'].values

    adata = adata[~pd.isnull(adata.obs['ground_truth'])]

    Cal_Spatial_Net(adata, rad_cutoff=300)
    preprocess(adata)
    Stats_Spatial_Net(adata)


    if 'highly_variable' in adata.var.columns:
        adata_Vars = adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata
    data1 = Transfer_pytorch_Data(adata_Vars, dim_reduction=False)
    data1 = data1.to(device)

    y = adata.obs['ground_truth']
    y = y.values

    x = data1.x
    x = x.to(torch.float32)
    x = x.cpu().numpy()



    G_df = adata.uns['Spatial_Net'].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)

    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])
    adj = G


    i = -1
    ari_list = []
    nmi_list = []
    super_best_ari = 0
    best_adata = adata
    true_labels = y
    features = x

    for seed in range(10):
        setup_seed(seed)
        i += 1
        pca = PCA(n_components=args.pca)
        features = torch.FloatTensor(pca.fit_transform(features))

        adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis,:],[0]),shape=adj.shape)
        adj.eliminate_zeros()
        adj_tensor = torch.tensor(adj.todense(),dtype=torch.float32)

        adj_norm_s = preprocess_graph(adj, args.gnnlayers, norm='sym', renorm=True)
        sm_feature_s = sp.csr_matrix(features).toarray()
        for a in adj_norm_s:
            sm_feature_s = a.dot(sm_feature_s)
        sm_feature_s = torch.FloatTensor(sm_feature_s)

        rever_net = reversible_network([features.shape[1]])
        rever_net.to(device)
        optimizer_rever_net = optim.Adam(rever_net.parameters(),lr=args.lr)
        rever_net.train()

        model = Encoder_net([features.shape[1]]+[args.dims])
        optimizer = optim.Adam(model.parameters(),lr=args.lr)
        model.train()

        best_ari = 0
        best_nmi = 0

        if args.cuda:
            model.to(device)
            sm_feature_s = sm_feature_s.to(device)


        print("Start Training...")
        for epoch in tqdm(range(args.epochs)):

            if epoch % 100 == 0 and epoch!=0:
                optimizer.param_groups[0]['lr'] *= 0.75
                optimizer_rever_net.param_groups[0]['lr'] *= 0.75

            optimizer_rever_net.zero_grad()
            optimizer.zero_grad()

            aug_feature = rever_net(sm_feature_s,True)
            aug_feature = aug_feature.to(device)
            z1 = model(sm_feature_s)
            z2 = model(aug_feature)

            z11 = rever_net(z1,True)
            z22 = rever_net(z2,False)

            loss_1 = loss_cal(z1,z22)
            loss_2 = loss_cal(z2,z11)
            contra_loss = loss_1 + loss_2
            z_mat = torch.matmul(z1,z2.T)
            n1 = z1.shape[0]
            re_loss1 = F.mse_loss(z_mat,torch.eye(n1).to(device))
            z_mat1 = torch.matmul(z11,z22.T)
            n2 = z11.shape[0]
            re_loss2 = F.mse_loss(z_mat1,torch.eye(n2).to(device))
            or_pro_sim_matrix = z1 * z11
            re_or_sim_matrix = z22 * z1
            loss_semantc = F.mse_loss(or_pro_sim_matrix,re_or_sim_matrix)

            total_loss = contra_loss + args.alpha1 * loss_semantc + args.alpha2 * re_loss1 + (1-args.alpha2) * re_loss2

            total_loss.backward()
            optimizer_rever_net.step()
            optimizer.step()

            if epoch % 1 ==0:
                model.eval()
                z1 = model(sm_feature_s)
                z2 = model(aug_feature)
                hidden_emb = (z1 + z2) / 2
                ari, nmi, pre_labels = clustering(hidden_emb,true_labels,n_clusters)
                adata.obs['nb_hid'] = pre_labels
                pre_labels = list(map(int, pre_labels))


                if ari > best_ari:

                    best_ari = ari
                    best_nmi = nmi
                    best_emb = hidden_emb
                    adata.obs['domain_best'] = list(map(str, pre_labels))
                    cells_reps = pd.DataFrame(best_emb.data.cpu().numpy())
                    cells = np.array(adata.obs.index)
                    cells_reps.index = cells
                    adata.obsm['domain_best'] = cells_reps.loc[adata.obs_names,].values

            tqdm.write('best_nmi: {}, best_ari: {} , lr:{} ,loss:{},range:{}'.format(best_nmi, best_ari,optimizer.param_groups[0]['lr'],total_loss, i))

        ari_list.append(best_ari)
        nmi_list.append(best_nmi)

        if best_ari > super_best_ari:
            super_best_ari = best_ari
            best_adata = adata

    ARI_avg = np.array(ari_list)
    NMI_avg = np.array(nmi_list)

    print("avg_ari:{},avg_nmi:{}".format(ARI_avg.mean(), NMI_avg.mean()))
    plt.rcParams["figure.figsize"] = (3, 3)
    sc.pl.spatial(best_adata, img_key="hires", color='domain_best', s=10, show=False, title='STCGCar')
    plt.savefig(f'./result/{data_name}_clustering.jpg',bbox_inches='tight', dpi=600)













