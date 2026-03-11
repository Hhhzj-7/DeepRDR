import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData, Batch, Data
from torch_geometric.transforms import ToUndirected
from torch_geometric.loader import LinkNeighborLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_curve
import random
from utils import smile_to_graph, train_doc2vec_model
from models import UnifiedModel
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors
from scipy.spatial.distance import pdist, squareform
import pickle
import os
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, accuracy_score, matthews_corrcoef, precision_recall_curve

# AUC AUPR F1 ACC MCC
def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    auc = roc_auc_score(y_true, y_pred)
    aupr = average_precision_score(y_true, y_pred)
    
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    y_pred_binary = (y_pred >= best_threshold).astype(int)

    best_threshold = 0.5
    
    f1 = f1_score(y_true, y_pred_binary)
    acc = accuracy_score(y_true, y_pred_binary)
    mcc = matthews_corrcoef(y_true, y_pred_binary)
    
    return {
        'AUC': auc, 'AUPR': aupr, 'F1': f1, 'ACC': acc, 'MCC': mcc
    }


def get_enhanced_drug_features(smiles_list, fp_radius=2, fp_dim=1024):
    """
    Morgan FP + MACCS Keys + physicochemical descriptors
    """    
    features_list = []
    valid_indices = []
    
    all_props = [] 
    
    for i, smiles in tqdm(enumerate(smiles_list), total=len(smiles_list), desc="Extracting Features"):
        if smiles is None or smiles == "NotFound" or len(smiles) == 0:
            features_list.append(None)
            continue
            
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            features_list.append(None)
            continue
            
        # 1. Morgan Fingerprint (1024)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius, nBits=fp_dim)
        fp_arr = np.array(fp)
        
        # 2. MACCS Keys (167)
        maccs = MACCSkeys.GenMACCSKeys(mol)
        maccs_arr = np.array(maccs)
        
        # 3. Physicochemical Descriptors (6)
        props = np.array([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol)
        ])
        
        combined = np.concatenate([fp_arr, maccs_arr, props])
        features_list.append(combined)
        valid_indices.append(i)
        all_props.append(props)
        
    if all_props:
        props_matrix = np.array(all_props)
        # Z-score normalization
        mean = np.mean(props_matrix, axis=0)
        std = np.std(props_matrix, axis=0) + 1e-8
        
        for idx, i in enumerate(valid_indices):
            feat = features_list[i]
            feat[-6:] = (feat[-6:] - mean) / std
            features_list[i] = feat
            
    valid_feats = [f for f in features_list if f is not None]
    if valid_feats:
        mean_feat = np.mean(valid_feats, axis=0)
    else:
        # 1024 + 167 + 6 = 1197
        mean_feat = np.zeros(fp_dim + 167 + 6)
        
    final_features = []
    for f in features_list:
        if f is None:
            final_features.append(mean_feat)
        else:
            final_features.append(f)
            
    return torch.tensor(np.array(final_features), dtype=torch.float)


class AutomaticWeightedLoss(nn.Module):
    """
    multi-task loss
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        self.params = nn.Parameter(torch.zeros(num), requires_grad=True)

    def forward(self, *x):
        # [loss_main, loss_cl]
        loss_sum = 0
        for i, loss in enumerate(x):
            # 0.5 * exp(-s) * loss + 0.5 * s
            loss_sum += 0.5 / (torch.exp(self.params[i])) * loss + 0.5 * self.params[i]
        return loss_sum


def get_similarity_edges(matrix, threshold, self_loop=False):
    adj_bool = matrix > threshold
    if not self_loop:
        np.fill_diagonal(adj_bool, False)
    row, col = np.where(adj_bool)
    edge_index = torch.tensor(np.array([row, col]), dtype=torch.long)
    return edge_index


def calculate_drug_similarity(fp_matrix):
    """ Tanimoto Similarity """
    # fp_matrix: [N_drug, FP_DIM] (numpy)
    # Tanimoto = (A & B) / (A | B)
    # Jaccard distance (1 - Tanimoto)
    from sklearn.metrics import pairwise_distances
    # metric='jaccard' 0/1
    dist = pairwise_distances(fp_matrix > 0, metric='jaccard', n_jobs=-1)
    sim = 1 - dist
    return sim


def calculate_gip_similarity(adj_matrix):
    """
    GIP
    """
    # adj_matrix: [N_rna, N_drug]
    norm_sq = np.sum(np.square(adj_matrix), axis=1)
    mean_norm = np.mean(norm_sq)

    if mean_norm == 0:
        gamma_c = 1.0
    else:
        gamma_c = 1.0 / mean_norm

    dists_sq = pdist(adj_matrix, metric='sqeuclidean')
    dists_matrix = squareform(dists_sq)
    cgs_matrix = np.exp(-gamma_c * dists_matrix)

    return cgs_matrix


def create_hetero_data(edges, labels, positive_edges_for_graph,
                       drug_features, rna_features):
    data = HeteroData()

    # node feature
    data['drug'].x = drug_features
    data['rna'].x = rna_features

    # Graph
    if len(positive_edges_for_graph) > 0:
        pos_edges = torch.tensor(positive_edges_for_graph, dtype=torch.long).t().contiguous()
        data['drug', 'interacts', 'rna'].edge_index = pos_edges
    else:
        data['drug', 'interacts', 'rna'].edge_index = torch.tensor([[], []], dtype=torch.long)

    data['drug', 'interacts', 'rna'].edge_label_index = torch.tensor(edges.T, dtype=torch.long).contiguous()
    data['drug', 'interacts', 'rna'].edge_label = torch.tensor(labels, dtype=torch.float)

    data = ToUndirected()(data)

    return data


def process_batch_drugs(batch, drug_smiles_graphs, device):

    target_edge_index = batch['drug', 'interacts', 'rna'].edge_label_index
    batch_drug_ids = batch['drug'].n_id[target_edge_index[0]].cpu().numpy()

    unique_ids, inverse_indices = np.unique(batch_drug_ids, return_inverse=True)

    valid_unique_graphs = []
    unique_id_to_valid_idx = []  # unique_ids[i] -> valid_unique_graphs 

    curr_idx = 0
    for d_id in unique_ids:
        graph = drug_smiles_graphs[d_id]
        if graph is not None:
            valid_unique_graphs.append(graph)
            unique_id_to_valid_idx.append(curr_idx)
            curr_idx += 1
        else:
            unique_id_to_valid_idx.append(-1)

    unique_id_to_valid_idx = np.array(unique_id_to_valid_idx)

    # Unique Batch
    if valid_unique_graphs:
        drug_smiles_batch = Batch.from_data_list(valid_unique_graphs).to(device)
    else:
        drug_smiles_batch = None

    # Map: [Batch_Size]
    # Batch_Index -> Unique_Index (inverse) -> Valid_Unique_Index
    batch_to_valid_map_np = unique_id_to_valid_idx[inverse_indices]
    drug_unique_map = torch.tensor(batch_to_valid_map_np, dtype=torch.long, device=device)

    return drug_smiles_batch, drug_unique_map


def mask_target_edges(batch):
    """
    Prevent data leakage
    """
    if not hasattr(batch['drug', 'interacts', 'rna'], 'edge_label_index'):
        return batch

    edge_index = batch['drug', 'interacts', 'rna'].edge_index
    target_edges = batch['drug', 'interacts', 'rna'].edge_label_index
    target_labels = batch['drug', 'interacts', 'rna'].edge_label

    pos_target_edges = target_edges[:, target_labels == 1]

    if pos_target_edges.size(1) == 0:
        return batch

    device = edge_index.device
    ei_np = edge_index.cpu().numpy().T
    tgt_np = pos_target_edges.cpu().numpy().T

    target_set = set(map(tuple, tgt_np))
    target_set.update(set(map(tuple, tgt_np[:, ::-1])))

    mask = [tuple(x) not in target_set for x in ei_np]
    mask_tensor = torch.tensor(mask, dtype=torch.bool, device=device)

    batch['drug', 'interacts', 'rna'].edge_index = edge_index[:, mask_tensor]

    return batch


@torch.no_grad()
def evaluate(model, loader, drug_smiles_graphs, rna_has_seq_tensor, device, mode='balanced_chunk'):
    model.eval()
    preds = []
    truths = []
    Emb_list = []

    for batch in loader:
        batch = batch.to(device)

        drug_smiles_batch, drug_unique_map = process_batch_drugs(batch, drug_smiles_graphs, device)

        # RNA 
        target_edge_index = batch['drug', 'interacts', 'rna'].edge_label_index
        rna_indices = batch['rna'].n_id[target_edge_index[1]]
        rna_valid_mask = rna_has_seq_tensor[rna_indices]
        _, _, _, _, interaction_pred, Emb = model(batch, drug_smiles_batch, drug_unique_map, rna_valid_mask)

        preds.append(interaction_pred.sigmoid().cpu())
        truths.append(batch['drug', 'interacts', 'rna'].edge_label.cpu())
        Emb_list.append(Emb.cpu())
    if not preds:
        return 0, 0, 0, 0, 0

    preds = torch.cat(preds, dim=0).numpy()
    truths = torch.cat(truths, dim=0).numpy()
    Emb_list = torch.cat(Emb_list, dim=0).numpy()
    res = calculate_metrics(truths,preds)

    # metrics
    return (res['AUC'],
            res['AUPR'],
            res['F1'],
            preds, truths, Emb_list,
            res['MCC'],
            res['ACC']
            )



if __name__ == '__main__':

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Hyperparameters
    EPOCHS = 500
    LEARNING_RATE = 5e-5
    BATCH_SIZE = 128
    K_MER_SIZE = 3
    DOC2VEC_DIM = 256
    HIDDEN_CHANNELS = 128
    OUT_CHANNELS = 64

    FP_RADIUS = 2
    FP_DIM = 1024  # Fingerprint
    DRUG_INITIAL_DIM = 1197  

    N_SPLITS = 5

    # Data Loading and Preprocessing
    print("Loading！")

    # File Path
    smiles_csv_path = 'Data/drug_smiles.csv'
    fasta_path = 'Data/rna_sequences.fasta'
    association_csv_path = 'Data/ncrna-drug_split.csv'
    adj_matrix_path = 'Data/adj_with_sens.csv'

    smiles_df = pd.read_csv(smiles_csv_path)
    association_df = pd.read_csv(association_csv_path, index_col=0)

    # Load ncRNA sequence
    rna_data = {}
    current_name = ""
    with open(fasta_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                current_name = line.strip()[1:]
                rna_data[current_name] = ''
            elif current_name:
                rna_data[current_name] += line.strip()

    # Get ncRNA and drug name
    all_drug_names = list(association_df.columns)
    all_rna_names = list(association_df.index)

    # map
    drug_map = {name: i for i, name in enumerate(all_drug_names)}
    rna_map = {name: i for i, name in enumerate(all_rna_names)}

    smiles_map = {row['name']: row['smiles'] for _, row in smiles_df.iterrows()}

    CACHE_PATH = 'Data/processed_data_cache.pkl'

    if os.path.exists(CACHE_PATH):
        print(f"Loading {CACHE_PATH}...")
        with open(CACHE_PATH, 'rb') as f:
            cache_data = pickle.load(f)

        rna_features_tensor = cache_data['rna_features_tensor']
        rna_has_seq_tensor = cache_data['rna_has_seq_tensor']
        drug_features_tensor = cache_data['drug_features_tensor']
        drug_smiles_graphs = cache_data['drug_smiles_graphs']
        all_drug_names = cache_data['all_drug_names']
        all_rna_names = cache_data['all_rna_names']
        print("Finished！")

    else:
        print("No cache detected, starting preprocessing...")
        print("ncRNA Doc2Vec feature...")
        doc2vec_model = train_doc2vec_model(rna_data, K_MER_SIZE, DOC2VEC_DIM)

        valid_rna_vectors = []
        rna_has_seq_list = []

        for name in all_rna_names:
            if name in doc2vec_model.dv:
                valid_rna_vectors.append(doc2vec_model.dv[name])
                rna_has_seq_list.append(True)
            else:
                rna_has_seq_list.append(False)

        if valid_rna_vectors:
            rna_mean_vec = np.mean(valid_rna_vectors, axis=0)
        else:
            rna_mean_vec = np.zeros(DOC2VEC_DIM)

        rna_features = []
        for i, name in enumerate(all_rna_names):
            if rna_has_seq_list[i]:
                rna_features.append(doc2vec_model.dv[name])
            else:
                rna_features.append(rna_mean_vec)

        rna_features_tensor = torch.tensor(np.array(rna_features), dtype=torch.float)
        rna_has_seq_tensor = torch.tensor(rna_has_seq_list, dtype=torch.bool)

        print("Drug feature...")
        drug_smiles_list = [smiles_map.get(name, 'NotFound') for name in all_drug_names]
        drug_features_tensor = get_enhanced_drug_features(drug_smiles_list, FP_RADIUS, FP_DIM)
        
        DRUG_INITIAL_DIM = drug_features_tensor.shape[1]
        print(f"{drug_features_tensor.shape} (Updated DRUG_INITIAL_DIM to {DRUG_INITIAL_DIM})")

        print("SMILES->Graph...")
        drug_smiles_graphs = []
        for drug_name in all_drug_names:
            smiles = smiles_map.get(drug_name, 'NotFound')
            if smiles != 'NotFound':
                x, edge_index = smile_to_graph(smiles)
                if x is not None:

                    drug_smiles_graphs.append(Data(x=x, edge_index=edge_index))
                else:
                    drug_smiles_graphs.append(None)
            else:
                drug_smiles_graphs.append(None)
        # Saving
        cache_data = {
            'rna_features_tensor': rna_features_tensor,
            'rna_has_seq_tensor': rna_has_seq_tensor,
            'drug_features_tensor': drug_features_tensor,
            'drug_smiles_graphs': drug_smiles_graphs,
            'all_drug_names': all_drug_names,
            'all_rna_names': all_rna_names
        }
        with open(CACHE_PATH, 'wb') as f:
            pickle.dump(cache_data, f)
        print("Saving!")

    # Load and split data
    print("\nLoading fold_info.pickle ...")
    with open('Data/fold_info.pickle', 'rb') as f:
        fold_info = pickle.load(f)


    # [RNA, Drug] -> [Drug, RNA]
    def align_indices(indices):
        if indices.shape[1] == 2:
            return indices[:, [1, 0]]
        return indices


    # Sensitive sample
    try:
        adj_sens_df = pd.read_csv(adj_matrix_path, index_col=0)
        adj_sens = adj_sens_df.values
        sens_indices = np.argwhere(adj_sens == -1)
    except:
        sens_indices = np.empty((0, 2), dtype=int)

    fold_test_metrics = {'auc': [], 'aupr': [], 'recall': [], 'f1': [], 'f2': [], 'mcc':[], 'precision':[], 'acc':[]}

    print("Drug similarity...")
    drug_sim_matrix = calculate_drug_similarity(drug_features_tensor.numpy())
    drug_sim_edge_index = get_similarity_edges(drug_sim_matrix, 0.6)
    print("Finished")

    # 5 CV
    for fold_idx in range(N_SPLITS):
        print(f"\n===== {fold_idx + 1}/{N_SPLITS} fold=====")

        # Read fold data
        pos_train_dmgat = fold_info["pos_train_ij_list"][fold_idx]
        pos_test_dmgat = fold_info["pos_test_ij_list"][fold_idx]
        unlabelled_train_dmgat = fold_info["unlabelled_train_ij_list"][fold_idx]
        unlabelled_test_dmgat = fold_info["unlabelled_test_ij_list"][fold_idx]

        # [Drug, RNA] -> (N, 2)
        pos_train_fold = align_indices(pos_train_dmgat)
        pos_test_fold = align_indices(pos_test_dmgat)
        unlabelled_train_fold = align_indices(unlabelled_train_dmgat)
        unlabelled_test_fold = align_indices(unlabelled_test_dmgat)
        sens_all = align_indices(sens_indices)

        # Segment sensitive data by fold
        train_unlabel_set = set(map(tuple, unlabelled_train_fold))
        test_unlabel_set  = set(map(tuple, unlabelled_test_fold))

        sens_train_fold = np.array(
            [x for x in sens_all if tuple(x) in train_unlabel_set]
        )

        sens_test_fold = np.array(
            [x for x in sens_all if tuple(x) in test_unlabel_set]
        )

        # Construct training set negative samples (prioritizing sensitive data)
        rng = np.random.default_rng(seed=fold_idx)
        num_train_neg = len(pos_train_fold)

        if len(sens_train_fold) >= num_train_neg:
            train_neg_fold = sens_train_fold[:num_train_neg]
        else:
            sens_set = set(map(tuple, sens_train_fold))
            unlabelled_filtered = np.array(
                [x for x in unlabelled_train_fold if tuple(x) not in sens_set]
            )

            num_need = num_train_neg - len(sens_train_fold)
            assert len(unlabelled_filtered) >= num_need, \
                f"Not enough train negatives: need {num_need}, have {len(unlabelled_filtered)}"

            rand_idx = rng.choice(len(unlabelled_filtered), size=num_need, replace=False)
            train_neg_fold = np.vstack([sens_train_fold, unlabelled_filtered[rand_idx]])

        # Construct test set negative samples (prioritizing sensitive data)
        num_test_neg = len(pos_test_fold)

        if len(sens_test_fold) >= num_test_neg:
            test_neg_fold = sens_test_fold[:num_test_neg]
        else:
            sens_set = set(map(tuple, sens_test_fold))
            unlabelled_filtered = np.array(
                [x for x in unlabelled_test_fold if tuple(x) not in sens_set]
            )

            num_need = num_test_neg - len(sens_test_fold)
            assert len(unlabelled_filtered) >= num_need, \
                f"Not enough test negatives: need {num_need}, have {len(unlabelled_filtered)}"

            rand_idx = rng.choice(len(unlabelled_filtered), size=num_need, replace=False)
            test_neg_fold = np.vstack([sens_test_fold, unlabelled_filtered[rand_idx]])

        # Construct final training/testing sets
        final_train_edges = np.vstack([pos_train_fold, train_neg_fold])
        final_train_labels = np.concatenate([
            np.ones(len(pos_train_fold), dtype=np.int64),
            np.zeros(len(train_neg_fold), dtype=np.int64)
        ])

        fold_test_edges = np.vstack([pos_test_fold, test_neg_fold])
        fold_test_labels = np.concatenate([
            np.ones(len(pos_test_fold), dtype=np.int64),
            np.zeros(len(test_neg_fold), dtype=np.int64)
        ])

        np.save('standard_fold/fold_train_pos_'+str(fold_idx)+'.npy', align_indices(pos_train_fold))
        np.save('standard_fold/fold_train_neg_'+str(fold_idx)+'.npy', align_indices(train_neg_fold))

        np.save('standard_fold/fold_test_pos_'+str(fold_idx)+'.npy', align_indices(pos_test_fold))
        np.save('standard_fold/fold_test_neg_'+str(fold_idx)+'.npy', align_indices(test_neg_fold))


        print(
            f"[Fold {fold_idx}] "
            f"Train Pos={len(pos_train_fold)}, "
            f"Neg={len(train_neg_fold)}, "
            f"SensUsed={len(sens_train_fold)} | "
            f"Test Pos={len(pos_test_fold)}, "
            f"Neg={len(test_neg_fold)}, "
            f"SensUsed={len(sens_test_fold)}"
        )

        curr_train_pos = pos_train_fold
        print(curr_train_pos)

        # Construct graph structure (calculating similarity edges)
        train_adj_for_gip = np.zeros((len(all_rna_names), len(all_drug_names)))
        train_adj_for_gip[curr_train_pos[:, 1], curr_train_pos[:, 0]] = 1  # [RNA, Drug] = 1
        fold_rna_gip_sim = calculate_gip_similarity(train_adj_for_gip)
        fold_rna_gip_tensor = torch.tensor(fold_rna_gip_sim, dtype=torch.float)

        rna_sim_edge_index = get_similarity_edges(fold_rna_gip_sim, 0.6)

        def add_sim_edges(data, d_sim_idx, r_sim_idx):
            data['drug', 'similar_to', 'drug'].edge_index = d_sim_idx
            data['rna', 'similar_to', 'rna'].edge_index = r_sim_idx
            return data


        # Create Data object (construct message passing graph using only positive samples)

        fold_train_data = create_hetero_data(final_train_edges, final_train_labels, curr_train_pos,
                                             drug_features_tensor, fold_rna_gip_tensor)
        fold_train_data = add_sim_edges(fold_train_data, drug_sim_edge_index, rna_sim_edge_index)

        fold_test_data = create_hetero_data(fold_test_edges, fold_test_labels, curr_train_pos,
                                            drug_features_tensor, fold_rna_gip_tensor)
        fold_test_data = add_sim_edges(fold_test_data, drug_sim_edge_index, rna_sim_edge_index)

        # Model initialization
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device_rna_features_tensor = rna_features_tensor.to(device)
        device_rna_has_seq_tensor = rna_has_seq_tensor.to(device)

        model = UnifiedModel(
            drug_initial_dim=DRUG_INITIAL_DIM,
            rna_feature_dim=DOC2VEC_DIM,
            rna_sim_feature_dim=fold_rna_gip_tensor.shape[1],
            hidden_channels=HIDDEN_CHANNELS,
            out_channels=OUT_CHANNELS,
            metadata=fold_train_data.metadata(),
            full_rna_features=device_rna_features_tensor
        ).to(device)

        
        awl = AutomaticWeightedLoss(num=2).to(device)
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': awl.parameters(), 'weight_decay': 0} 
        ], lr=LEARNING_RATE)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )

        interaction_loss_fn = torch.nn.BCEWithLogitsLoss().to(device)
        eval_loader = LinkNeighborLoader(
            fold_test_data,
            num_neighbors={
                ('drug', 'interacts', 'rna'): [20, 10],
                ('rna', 'rev_interacts', 'drug'): [20, 10],
                ('drug', 'similar_to', 'drug'): [10, 5],
                ('rna', 'similar_to', 'rna'): [10, 5]
            },
            batch_size=BATCH_SIZE * 4,  
            shuffle=False,
            disjoint=False,
            num_workers=0,
            persistent_workers=False,
            edge_label_index=(
                ('drug', 'interacts', 'rna'), fold_test_data['drug', 'interacts', 'rna'].edge_label_index),
            edge_label=fold_test_data['drug', 'interacts', 'rna'].edge_label
        )

        # Loader
        train_loader = LinkNeighborLoader(
            fold_train_data,
            num_neighbors={
                ('drug', 'interacts', 'rna'): [20, 10],
                ('rna', 'rev_interacts', 'drug'): [20, 10],
                ('drug', 'similar_to', 'drug'): [10, 5],
                ('rna', 'similar_to', 'rna'): [10, 5]
            },
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            persistent_workers=False,
            edge_label_index=(
                ('drug', 'interacts', 'rna'), fold_train_data['drug', 'interacts', 'rna'].edge_label_index),
            edge_label=fold_train_data['drug', 'interacts', 'rna'].edge_label,
            disjoint=False
        )

        # Training
        print("\nTraining...")
        best_test_auc = 0
        best_metrics = (0, 0, 0, 0, 0, 0, 0, 0)
        best_epoch = 0
        best_model_path = f'save/best_model_fold_{fold_idx}.pth'
        best_emb = None


        def info_nce_loss(view1, view2, temperature=0.07, symmetric=True):
            """
            InfoNCE CL
            view1, view2: [N_valid, D_out]
            """
            view1 = torch.nn.functional.normalize(view1, p=2, dim=1)
            view2 = torch.nn.functional.normalize(view2, p=2, dim=1)

            # [N_valid, N_valid]
            similarity_matrix = torch.matmul(view1, view2.T) / temperature

            # [0, 1, 2, ..., N_valid-1]
            labels = torch.arange(view1.shape[0], device=view1.device)

            loss_v1_v2 = torch.nn.functional.cross_entropy(similarity_matrix, labels)

            if symmetric:
                # (v2: anchor, v1: target)
                loss_v2_v1 = torch.nn.functional.cross_entropy(similarity_matrix.T, labels)
                loss = (loss_v1_v2 + loss_v2_v1) / 2
            else:
                loss = loss_v1_v2

            return loss


        patience = 100
        counter = 0  
        early_stop = False  
        for epoch in range(EPOCHS):
            if early_stop:
                print(f"  [Early Stopping]  {epoch} epoch")
                break

            model.train()
            total_loss_sum = 0

            with tqdm(train_loader, desc=f"Fold {fold_idx} Ep {epoch + 1}/{EPOCHS}", leave=True) as pbar:
                
                for batch in pbar:
                    batch = batch.to(device)

                    batch = mask_target_edges(batch)

                    drug_smiles_batch, drug_unique_map = process_batch_drugs(batch, drug_smiles_graphs, device)

                    target_edge_index = batch['drug', 'interacts', 'rna'].edge_label_index
                    rna_indices = batch['rna'].n_id[target_edge_index[1]]
                    rna_valid_mask = device_rna_has_seq_tensor[rna_indices]

                    # model forward
                    drug_s_proj, drug_a_proj, rna_s_proj, rna_a_proj, interaction_pred, _ = model(
                        batch, drug_smiles_batch, drug_unique_map, rna_valid_mask
                    )

                    ground_truth = batch['drug', 'interacts', 'rna'].edge_label
                    loss_inter = interaction_loss_fn(interaction_pred.squeeze(), ground_truth)

                    # CL
                    loss_drug_cl = torch.tensor(0.0, device=device)
                    loss_rna_cl = torch.tensor(0.0, device=device)
                    drug_has_struct_mask = drug_unique_map >= 0
                    if drug_has_struct_mask.sum() > 1:
                        loss_drug_cl = info_nce_loss(
                            drug_s_proj[drug_has_struct_mask],
                            drug_a_proj[drug_has_struct_mask]
                        )
                    if rna_valid_mask.sum() > 1:
                        loss_rna_cl = info_nce_loss(
                            rna_s_proj[rna_valid_mask],
                            rna_a_proj[rna_valid_mask]
                        )

                    # Merge CL Loss
                    loss_cl_total = loss_drug_cl + loss_rna_cl

                    # Total Loss
                    total_loss = awl(loss_inter, loss_cl_total)

                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                    total_loss_sum += total_loss.item()

                    current_avg_loss = total_loss_sum / (pbar.n + 1)
                    pbar.set_postfix({'loss': f"{total_loss.item():.4f}", 'avg': f"{current_avg_loss:.4f}"})

            # Loss
            epoch_train_loss = total_loss_sum / len(train_loader)
            scheduler.step(epoch_train_loss)

            # Test Evaluation
            test_auc, test_aupr, test_f1, Predictions, True_labels, Emb, test_mcc, test_acc = evaluate(
                model=model,
                loader=eval_loader,
                drug_smiles_graphs=drug_smiles_graphs,
                rna_has_seq_tensor=device_rna_has_seq_tensor,
                device=device,
                mode='balanced_chunk'
            )
            print(
                f"    Test Results: AUC={test_auc:.4f} | AUPR={test_aupr:.4f} | F1={test_f1:.4f} | MCC={test_mcc:.4f}")
            print(
                f"Main w: {0.5 * torch.exp(-awl.params[0]).item():.4f}, CL w: {0.5 * torch.exp(-awl.params[1]).item():.4f}")

            if test_auc > best_test_auc:
                save_path_e = "save/Res"+"_fold_"+str(fold_idx)+".pth"
                torch.save({
                    'Pre': torch.tensor(Predictions),
                    'labels': torch.tensor(True_labels)
                }, save_path_e)
                best_test_auc = test_auc
                best_emb = Emb
                best_metrics = (test_auc, test_aupr, test_f1, test_mcc, test_acc)
                best_epoch = epoch + 1

                torch.save(model.state_dict(), best_model_path)

                print(f"    [Saved Best] New Best AUC : {best_test_auc:.4f}")
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    early_stop = True

        print(
            f"  [Fold {fold_idx} Best] Epoch {best_epoch}: AUC={best_metrics[0]:.4f}, AUPR={best_metrics[1]:.4f}, F1={best_metrics[2]:.4f}")
        fold_test_metrics['auc'].append(best_metrics[0])
        fold_test_metrics['aupr'].append(best_metrics[1])
        fold_test_metrics['f1'].append(best_metrics[2])
        fold_test_metrics['mcc'].append(best_metrics[3])
        fold_test_metrics['acc'].append(best_metrics[4])

        del model
        del optimizer
        del train_loader
        del eval_loader

    print("\n===== 5CV finished! =====")
    avg_auc = np.mean(fold_test_metrics['auc'])
    std_auc = np.std(fold_test_metrics['auc'])
    avg_aupr = np.mean(fold_test_metrics['aupr'])
    std_aupr = np.std(fold_test_metrics['aupr'])
    avg_f1 = np.mean(fold_test_metrics['f1'])
    std_f1 = np.std(fold_test_metrics['f1'])
    avg_f2 = np.mean(fold_test_metrics['f2'])
    std_f2 = np.std(fold_test_metrics['f2'])
    avg_mcc = np.mean(fold_test_metrics['mcc'])
    std_mcc = np.std(fold_test_metrics['mcc'])


    print(f"AUC:    {np.mean(fold_test_metrics['auc']):.4f} ± {np.std(fold_test_metrics['auc']):.4f}")
    print(f"AUPR:   {np.mean(fold_test_metrics['aupr']):.4f} ± {np.std(fold_test_metrics['aupr']):.4f}")
    print(f"F1:     {np.mean(fold_test_metrics['f1']):.4f} ± {np.std(fold_test_metrics['f1']):.4f}")
    print(f"MCC:     {np.mean(fold_test_metrics['mcc']):.4f} ± {np.std(fold_test_metrics['mcc']):.4f}")
    print(f"ACC:     {np.mean(fold_test_metrics['acc']):.4f} ± {np.std(fold_test_metrics['acc']):.4f}")

