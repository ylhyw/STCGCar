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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--k",type=int,default=10,help='parameter k in spatial graph')
    parser.add_argument('--knn_distanceType', type=str, default='euclidean',
                        help='graph distance type: euclidean/cosine/correlation')
    parser.add_argument("--pca",type=int,default=300,help='parameter pca')
    parser.add_argument("--dims", type=int, default=300, help='parameter hidden')
    parser.add_argument('--gnnlayers', type=int, default=3, help="Number of gnn layers")
    parser.add_argument('--lr', type=float, default=4e-5, help='Initial learning rate.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--alpha1', type=float, default=0.5, help='Loss balance parameter')
    parser.add_argument('--alpha2', type=float, default=0.8, help='Loss balance parameter')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()


    data_root = "./data/151672"
    save_root = "./result"
    n_cliuters = 5

    adata = sc.read_visium(data_root)
    adata.var_names_make_unique()


    df_meta = pd.read_csv(data_root+'/metadata.tsv',sep='\t')
    df_meta_layer = df_meta['layer_guess']
    adata.obs['ground_truth'] = df_meta_layer.values


    adata = adata[~pd.isnull(adata.obs['ground_truth'])]

    Cal_Spatial_Net(adata,rad_cutoff=150)
    preprocess(adata)
    Stats_Spatial_Net(adata)

    if 'highly_variable' in adata.var.columns:
        adata_Vars = adata[:,adata.var['highly_variable']]
    else:
        adata_Vars = adata
    data = Transfer_pytorch_Data(adata_Vars,dim_reduction=False)
    data = data.to(device)

    true_labels = adata.obs['ground_truth'].values
    features = data.x.to(torch.float32).cpu().numpy()

    graph_adj = graph_construction(adata_Vars.obsm['spatial'],adata_Vars.shape[0],args)
    adj = preprocess_adj(graph_adj['adj_label'].numpy())

    i = -1
    ari_list = []
    nmi_list = []
    super_best_ari = 0
    best_adata = adata
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
        optimizer_rever_net = optim.SGD(rever_net.parameters(),lr=args.lr,momentum=0.9,nesterov=True,weight_decay=0)
        rever_net.train()

        model = Encoder_net([features.shape[1]]+[args.dims])
        optimizer = optim.Adam(model.parameters(),lr=args.lr)
        model.train()

        best_ari = 0
        best_nmi = 0

        if args.cuda:
            model.to(device)
            sm_feature_s = sm_feature_s.to(device)
            # rever_net.to(device)

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
                ari, nmi, pre_labels = clustering(hidden_emb,true_labels,n_cliuters)
                adata.obs['nb_hid'] = pre_labels
                if 1:
                    new_type = refine_label(adata,30,key="nb_hid")
                    pre_labels = new_type
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
    plt.savefig(f'./result/clustering.jpg',bbox_inches='tight', dpi=600)




















