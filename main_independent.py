"""
Independent test 
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData, Batch, Data
from torch_geometric.transforms import ToUndirected
from tqdm import tqdm
import pickle
import os
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors
from rdkit import DataStructs
from scipy.spatial.distance import pdist, squareform

from utils import smile_to_graph
from models import UnifiedModel

from sklearn.metrics import (roc_auc_score, average_precision_score, 
                           precision_recall_curve, auc,
                           accuracy_score, precision_score, 
                           recall_score, f1_score, confusion_matrix,
                           classification_report, matthews_corrcoef)
import matplotlib.pyplot as plt

def calculate_metrics(labels, predictions, threshold=0.5):

    labels = np.array(labels)
    predictions = np.array(predictions)
    
    pred_labels = (predictions >= threshold).astype(int)
    
    try:
        auc_roc = roc_auc_score(labels, predictions)
    except ValueError as e:
        auc_roc = 0.5  # 默认值
    
    try:
        precision_curve, recall_curve, _ = precision_recall_curve(labels, predictions)
        aupr = auc(recall_curve, precision_curve)
    except Exception as e:
        aupr = 0.5  # 默认值
        precision_curve, recall_curve = None, None
    
    try:
        avg_precision = average_precision_score(labels, predictions)
    except Exception as e:
        avg_precision = 0.5
    
    accuracy = accuracy_score(labels, pred_labels)
    
    
    f1 = f1_score(labels, pred_labels, zero_division=0)
    
    cm = confusion_matrix(labels, pred_labels)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    mcc = matthews_corrcoef(labels, pred_labels)
    metrics = {
        'AUC-ROC': auc_roc,
        'AUPR': aupr,
        'Accuracy': accuracy,
        'F1-Score': f1,
        'MCC': mcc,
    }
    if precision_curve is not None and recall_curve is not None:
        metrics['PR Curve'] = (recall_curve, precision_curve)
    
    return metrics

def print_metrics(metrics, detailed=False):
    """print result"""
    
    print("\n" + "="*50)
    print("Metric")
    print("="*50)
    
    print(f"  AUC-ROC:      {metrics['AUC-ROC']:.4f}")
    print(f"  AUPR:         {metrics['AUPR']:.4f}")
    print(f"  Accuracy:     {metrics['Accuracy']:.4f}")

    print(f"  F1-Score:     {metrics['F1-Score']:.4f}")
    

# get drug feature
def get_enhanced_drug_features(smiles_list, fp_radius=2, fp_dim=1024):
    print("Morgan + MACCS + PhysChem...")
    
    features_list = []
    valid_indices = []
    all_props = [] 
    
    for i, smiles in tqdm(enumerate(smiles_list), total=len(smiles_list), desc="Extracting Features"):
        if smiles is None or smiles == "NotFound" or len(smiles) == 0 or pd.isna(smiles):
            features_list.append(None)
            continue
            
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            features_list.append(None)
            continue
            
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius, nBits=fp_dim)
        fp_arr = np.array(fp)
        maccs = MACCSkeys.GenMACCSKeys(mol)
        maccs_arr = np.array(maccs)
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
        mean_feat = np.zeros(fp_dim + 167 + 6)
        
    final_features = []
    for f in features_list:
        if f is None:
            final_features.append(mean_feat)
        else:
            final_features.append(f)
            
    return torch.tensor(np.array(final_features), dtype=torch.float)


def calculate_gip_similarity(adj_matrix):
    """GIP"""
    norm_sq = np.sum(np.square(adj_matrix), axis=1)
    mean_norm = np.mean(norm_sq[norm_sq > 0]) if np.any(norm_sq > 0) else 1.0
    gamma_c = 1.0 / mean_norm if mean_norm > 0 else 1.0
    dists_sq = pdist(adj_matrix, metric='sqeuclidean')
    dists_matrix = squareform(dists_sq)
    cgs_matrix = np.exp(-gamma_c * dists_matrix)
    return cgs_matrix


def get_similarity_edges(matrix, threshold, self_loop=False):
    # Convert the similarity matrix into edge indices
    adj_bool = matrix > threshold
    if not self_loop:
        np.fill_diagonal(adj_bool, False)
    row, col = np.where(adj_bool)
    edge_index = torch.tensor(np.array([row, col]), dtype=torch.long)
    return edge_index


def calculate_drug_similarity(fp_matrix):
    """Drug Tanimoto Similarity"""
    from sklearn.metrics import pairwise_distances
    dist = pairwise_distances(fp_matrix > 0, metric='jaccard', n_jobs=-1)
    sim = 1 - dist
    return sim



def build_full_hetero_graph(all_drug_features, all_drug_smiles_graphs, rna_gip_features,
                            association_df, device):
    """Construct a complete heterogeneous graph containing all drugs and RNAs"""
    
    g = HeteroData()
    
    # Drug node
    n_drugs = len(all_drug_features)
    g['drug'].num_nodes = n_drugs
    g['drug'].x = all_drug_features.to(device)
    
    # RNA node
    n_rnas = len(rna_gip_features)
    g['rna'].num_nodes = n_rnas
    g['rna'].x = rna_gip_features.to(device)
    
    # edge
    drug_names = list(association_df.columns)
    rna_names = list(association_df.index)
    
    inter_edges = []
    for r_idx, rna_name in enumerate(rna_names):
        for d_idx, drug_name in enumerate(drug_names):
            if association_df.loc[rna_name, drug_name] == 1:
                inter_edges.append([d_idx, r_idx])
    
    if inter_edges:
        edge_index = torch.tensor(inter_edges, dtype=torch.long).t()
        g['drug', 'interacts', 'rna'].edge_index = edge_index.to(device)
    else:
        g['drug', 'interacts', 'rna'].edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    
    # Drug similarity
    drug_sim_matrix = calculate_drug_similarity(all_drug_features.cpu().numpy())
    drug_sim_edges = get_similarity_edges(drug_sim_matrix, threshold=0.6, self_loop=False)
    g['drug', 'similar_to', 'drug'].edge_index = drug_sim_edges.to(device)
    
    # RNA similarity
    rna_gip_np = rna_gip_features.cpu().numpy()
    rna_sim_edges = get_similarity_edges(rna_gip_np, threshold=0.6, self_loop=False)
    g['rna', 'similar_to', 'rna'].edge_index = rna_sim_edges.to(device)
    
    g = ToUndirected()(g)
    
    return g


def predict_scores_batch(model, hetero_graph, drug_indices, rna_indices, 
                         drug_smiles_graphs, rna_has_seq_tensor, device, batch_size=512):
    model.eval()
    n_pairs = len(drug_indices)
    all_scores = []
    with torch.no_grad():
        for start_idx in range(0, n_pairs, batch_size):
            end_idx = min(start_idx + batch_size, n_pairs)
            batch_drug_idx = drug_indices[start_idx:end_idx]
            batch_rna_idx = rna_indices[start_idx:end_idx]
            batch_size_actual = len(batch_drug_idx)
            # edge_label_index
            edge_label_index = torch.tensor(
                [batch_drug_idx, batch_rna_idx], 
                dtype=torch.long, 
                device=device
            )
            
            test_graph = hetero_graph.clone()
            test_graph['drug', 'interacts', 'rna'].edge_label_index = edge_label_index
            test_graph['drug', 'interacts', 'rna'].edge_label = torch.ones(
                batch_size_actual, dtype=torch.float, device=device
            )
            
            # n_id
            test_graph['drug'].n_id = torch.arange(hetero_graph['drug'].num_nodes, device=device)
            test_graph['rna'].n_id = torch.arange(hetero_graph['rna'].num_nodes, device=device)
            
            # drug graph
            unique_drugs = list(set(batch_drug_idx))
            drug_graphs = [drug_smiles_graphs[i] for i in unique_drugs if drug_smiles_graphs[i] is not None]
            
            if drug_graphs:
                drug_smiles_batch = Batch.from_data_list([g.to(device) for g in drug_graphs]).to(device)
                drug_map_dict = {d: i for i, d in enumerate(unique_drugs) if drug_smiles_graphs[d] is not None}
                drug_unique_map = torch.tensor(
                    [drug_map_dict.get(d, -1) for d in batch_drug_idx],
                    dtype=torch.long,
                    device=device
                )
            else:
                drug_smiles_batch = None
                drug_unique_map = torch.full((batch_size_actual,), -1, dtype=torch.long, device=device)
            
            rna_valid_mask = rna_has_seq_tensor[batch_rna_idx]
            
            try:
                _, _, _, _, predictions, _ = model(
                    test_graph,
                    drug_smiles_batch,
                    drug_unique_map,
                    rna_valid_mask
                )
                scores = torch.sigmoid(predictions).cpu().numpy().flatten()
                all_scores.extend(scores)
            except Exception as e:
                all_scores.extend([0.0] * batch_size_actual)
    
    return np.array(all_scores)


def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Path
    new_drugs_csv = 'Data/DC_Drugs.csv'
    existing_smiles_csv = 'Data/drug_smiles.csv'
    association_csv = 'Data/ncrna-drug_split.csv'
    cache_path = 'Data/processed_data_cache.pkl'
    model_dir = 'save/'
    
    
    # Load the data and model
    print("\nLoad the data and model")
    
    # Load cache
    print(f"Load: {cache_path}")
    try:
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
    except:
        raise RuntimeError('run standard setting first!')    
    
    rna_features_tensor = cache_data['rna_features_tensor']
    rna_has_seq_tensor = cache_data['rna_has_seq_tensor']
    all_rna_names = cache_data['all_rna_names']
    existing_drug_features = cache_data['drug_features_tensor']
    existing_drug_graphs = cache_data['drug_smiles_graphs']
    existing_drug_names = cache_data['all_drug_names']
    
    print(f"Drug: {len(existing_drug_names)}")
    print(f"RNA: {len(all_rna_names)}")
    all_drug_graphs = existing_drug_graphs
    all_drug_features = existing_drug_features
    # get RNA GIP
    association_df = pd.read_csv(association_csv, index_col=0)
    
    print("RNA GIP...")
    adj_matrix = np.zeros((len(all_rna_names), len(existing_drug_names)))
    for rna_idx, rna_name in enumerate(all_rna_names):
        if rna_name in association_df.index:
            for drug_idx, drug_name in enumerate(existing_drug_names):
                if drug_name in association_df.columns:
                    adj_matrix[rna_idx, drug_idx] = association_df.loc[rna_name, drug_name]
    
    rna_gip_sim = calculate_gip_similarity(adj_matrix)
    rna_gip_tensor = torch.tensor(rna_gip_sim, dtype=torch.float)
    
    # Construct the complete heterogeneous graph
    hetero_graph = build_full_hetero_graph(
        all_drug_features, all_drug_graphs, rna_gip_tensor,
        association_df, device
    )
    
    # Load model
    print("\nLoad model...")
    HIDDEN_CHANNELS = 128
    OUT_CHANNELS = 64
    DOC2VEC_DIM = 256
    DRUG_INITIAL_DIM = all_drug_features.shape[1]
    
    models = []
    used_id_edge = []
    for fold_idx in range(5):
        model_path = os.path.join(model_dir, f'best_model_fold_{fold_idx}.pth')
        print(f"  Load model: {model_path}")
        
        model = UnifiedModel(
            drug_initial_dim=DRUG_INITIAL_DIM,
            rna_feature_dim=DOC2VEC_DIM,
            rna_sim_feature_dim=rna_gip_tensor.shape[1],
            hidden_channels=HIDDEN_CHANNELS,
            out_channels=OUT_CHANNELS,
            metadata=hetero_graph.metadata(),
            full_rna_features=rna_features_tensor.to(device)
        ).to(device)
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models.append(model)


        # Load dataset
        fold_train_pos = np.load('standard_fold/fold_train_pos_'+str(fold_idx)+'.npy')
        fold_train_neg = np.load('standard_fold/fold_train_neg_'+str(fold_idx)+'.npy')
        fold_test_pos = np.load('standard_fold/fold_test_pos_'+str(fold_idx)+'.npy')
        fold_test_neg = np.load('standard_fold/fold_test_neg_'+str(fold_idx)+'.npy')
        used_id_edge.append(fold_train_pos)
        used_id_edge.append(fold_train_neg)
        used_id_edge.append(fold_test_pos)
        used_id_edge.append(fold_test_neg)

    combined_id = np.vstack(used_id_edge) 
    print(f"Load {len(models)} model")



    # Load the positive samples
    test_data = pd.read_csv('Data/independent_data.csv')
    print(f"Load {len(test_data)} data")

    # Create a mapping from names to indices
    rna_name_to_index = {name: idx for idx, name in enumerate(all_rna_names)}  # 行索引
    drug_name_to_index = {name: idx for idx, name in enumerate(existing_drug_names)}  # 列索引

    all_rna_indices = []   
    all_drug_indices = []  
    all_labels = []

    used_pairs_set = set()
    if 'combined_id' in locals():
        for pair in combined_id:
            # [rna_idx, drug_idx]
            used_pairs_set.add((pair[0], pair[1]))

    # Processing the positive samples

    resistant_count = 0
    sensitive_count = 0

    for idx, row in test_data.iterrows():
        rna_name = row['Dataset_RNA_match']
        drug_name = row['Dataset_Drug_match']
        effect = row['Effect']
        
        if rna_name in rna_name_to_index and drug_name in drug_name_to_index:
            rna_idx = rna_name_to_index[rna_name]  
            drug_idx = drug_name_to_index[drug_name] 
            
            if (rna_idx, drug_idx) not in used_pairs_set:
                all_rna_indices.append(rna_idx)  
                all_drug_indices.append(drug_idx)
                
                # sensitive=0, resistant=1
                if effect == 'sensitive' or effect == 0:
                    all_labels.append(0)
                    sensitive_count += 1
                else:
                    all_labels.append(1)
                    resistant_count += 1
                
                used_pairs_set.add((rna_idx, drug_idx))

    print(f"Resistant (1): {resistant_count}, Sensitive (0): {sensitive_count}")

    try:
        adj_sens_df = pd.read_csv('Data/adj_with_sens.csv', index_col=0)
        print(f"Load: {adj_sens_df.shape[0]}(RNA) x {adj_sens_df.shape[1]}(druug)")
        
        # Get the values and indices of the matrix
        adj_sens = adj_sens_df.values
        rna_names = adj_sens_df.index.tolist()
        drug_names = adj_sens_df.columns.tolist()
        
        unlabel_indices = np.argwhere(adj_sens == 0)
        np.random.seed(1)
        np.random.shuffle(unlabel_indices)
        # 1:1
        target_neg_samples = resistant_count
        
        neg_count = sensitive_count
        
        for rna_pos, drug_pos in unlabel_indices:
            rna_name = rna_names[rna_pos]
            drug_name = drug_names[drug_pos]
            
            # Check
            if rna_name in rna_name_to_index and drug_name in drug_name_to_index:
                rna_idx = rna_name_to_index[rna_name]
                drug_idx = drug_name_to_index[drug_name]
                
                if (rna_idx, drug_idx) not in used_pairs_set:
                    all_rna_indices.append(rna_idx)
                    all_drug_indices.append(drug_idx)
                    all_labels.append(0)  
                    used_pairs_set.add((rna_idx, drug_idx))
                    neg_count += 1
                    
                    if neg_count >= target_neg_samples:
                        break
            else:
                print('error name')
        print(f"{neg_count} unlabel as neg")
        
    except FileNotFoundError:
        print("no adj_with_sens.csv")
    except Exception as e:
        print(f" {e}")

    all_rna_indices = np.array(all_rna_indices)
    all_drug_indices = np.array(all_drug_indices)
    all_labels = np.array(all_labels)


    print("\n" + "="*50)
    print("Dataset construction completed")
    print("="*50)
    print(f"All samples: {len(all_labels)}")
    print(f"Resistant (1): {sum(all_labels==1)}")
    print(f"Sensitive/Unlabel (0): {sum(all_labels==0)}")
    if sum(all_labels==1) > 0:
        print(f"(1:0): 1:{sum(all_labels==0)/sum(all_labels==1):.2f}")
    print("="*50)

    labels = all_labels
    
    
    n_drugs = len(existing_drug_names)
    n_rnas = len(all_rna_names)
    n_models = len(models)
    
    print(f"Drug: {n_drugs}")
    print(f"RNA: {n_rnas}")
    print(f"Model: {n_models}")

    # 存储所有模型的预测
    all_model_predictions = []
    
    for model_idx, model in enumerate(models):
        print(f"\nUse model {model_idx + 1}/{n_models} ...")
        
        scores = predict_scores_batch(
            model, hetero_graph, all_drug_indices, all_rna_indices,
            all_drug_graphs, rna_has_seq_tensor.to(device), device,
            batch_size=1
        )
        
        all_model_predictions.append(scores)
    
    # Average
    print("\nAvg Scores...")
    avg_scores = np.mean(all_model_predictions, axis=0)
    
    metrics_default = calculate_metrics(labels, avg_scores, threshold=0.5)

    results_list = []
    results_list = [calculate_metrics(labels, all_model_predictions[i], threshold=0.5) for i in range(5)]    # 计算平均值和标准差
    metrics_to_average = ['AUC-ROC', 'AUPR', 'MCC', 'F1-Score', 'Accuracy']

    summary_stats = {}
    for metric in metrics_to_average:
        values = [r[metric] for r in results_list]
        summary_stats[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
        }
    print(summary_stats)
    
if __name__ == '__main__':
    main()