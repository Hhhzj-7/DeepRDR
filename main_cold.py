"""
RNA cold start: 5-fold cross-validation based on RNA nodes, where the RNAs in the test set are completely unseen in the training set

Drug cold start: 5-fold cross-validation based on drug nodes, where the drugs in the test set are completely unseen in the training set

Dual cold start: 5-fold cross-validation for both RNAs and drugs

"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData, Batch, Data
from torch_geometric.loader import LinkNeighborLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import random
import pickle
import os
import argparse
from utils import smile_to_graph, train_doc2vec_model
from models import UnifiedModel
from rdkit import Chem
from rdkit.Chem import MACCSkeys, Descriptors, AllChem
from main import (
    AutomaticWeightedLoss,
    get_similarity_edges,
    calculate_drug_similarity,
    calculate_gip_similarity,
    create_hetero_data,
    process_batch_drugs,
    mask_target_edges
)

def normalize_name(name):
    if pd.isna(name):
        return ""
    return str(name).lower().strip().replace(" ", "").replace("-", "").replace("_", "")


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
        
        # 3. physicochemical descriptors
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


def info_nce_loss(view1, view2, temperature=0.07, symmetric=True):
    """
    InfoNCE CL
    view1, view2: [N_valid, D_out]
    """
    view1 = torch.nn.functional.normalize(view1, p=2, dim=1)
    view2 = torch.nn.functional.normalize(view2, p=2, dim=1)

    similarity_matrix = torch.matmul(view1, view2.T) / temperature
    labels = torch.arange(view1.shape[0], device=view1.device)

    loss_v1_v2 = torch.nn.functional.cross_entropy(similarity_matrix, labels)

    if symmetric:
        loss_v2_v1 = torch.nn.functional.cross_entropy(similarity_matrix.T, labels)
        loss = (loss_v1_v2 + loss_v2_v1) / 2
    else:
        loss = loss_v1_v2

    return loss


class ColdStartSplitter:

    def __init__(self, adj_with_sens_df, n_splits=5, seed=42):
        self.adj_matrix = adj_with_sens_df.values
        self.rna_names = list(adj_with_sens_df.index)
        self.drug_names = list(adj_with_sens_df.columns)
        self.n_rna = len(self.rna_names)
        self.n_drug = len(self.drug_names)
        self.n_splits = n_splits
        self.seed = seed

        np.random.seed(seed)
        random.seed(seed)

        self.resistant_indices = np.argwhere(self.adj_matrix == 1)  # (rna_idx, drug_idx)
        self.sensitive_indices = np.argwhere(self.adj_matrix == -1)
        self.unknown_indices = np.argwhere(self.adj_matrix == 0)

        print(f"  Resistant: {len(self.resistant_indices)}")
        print(f"  Sensitive: {len(self.sensitive_indices)}")
        print(f"  Unknown: {len(self.unknown_indices)}")
        print(f"  RNA: {self.n_rna}")
        print(f"  Drug: {self.n_drug}")

    def _sample_balanced_negative(self, positive_edges, used_negatives=None,
                                    valid_rnas=None, valid_drugs=None):
        if used_negatives is None:
            used_negatives = set()

        n_needed = len(positive_edges)
        selected = []

        pos_set = set(map(tuple, positive_edges))
        valid_rnas_set = set(valid_rnas) if valid_rnas is not None else None
        valid_drugs_set = set(valid_drugs) if valid_drugs is not None else None

        def is_valid_sample(rna_idx, drug_idx):
            """Check whether the samples satisfy the node constraints"""
            if valid_rnas_set is not None and rna_idx not in valid_rnas_set:
                return False
            if valid_drugs_set is not None and drug_idx not in valid_drugs_set:
                return False
            return True

        # Combine all potential negative sample sources (sensitive + unknown) and shuffle them
        all_candidates = np.vstack([self.sensitive_indices, self.unknown_indices])
        np.random.shuffle(all_candidates)

        n_from_sensitive = 0
        n_from_unknown = 0
        
        # Sampling
        for idx in all_candidates:
            if len(selected) >= n_needed:
                break
            rna_idx, drug_idx = idx
            key = tuple(idx)
            
            if key not in pos_set and key not in used_negatives and is_valid_sample(rna_idx, drug_idx):
                selected.append(idx)
                used_negatives.add(key)
                pass
        
        return np.array(selected) if selected else np.empty((0, 2), dtype=int), used_negatives

    # Cold start for ncRNA
    def create_rna_cold_start_splits(self):
        print("\nCold ncRNA...")

        # Randomly divide the RNA nodes into 5 folds
        rna_indices = np.arange(self.n_rna)
        np.random.shuffle(rna_indices)

        rna_folds = np.array_split(rna_indices, self.n_splits)

        splits = []
        for fold_idx in range(self.n_splits):
            test_rnas = set(rna_folds[fold_idx])
            train_rnas = set(rna_indices) - test_rnas

            test_pos = []
            train_pos = []

            for rna_idx, drug_idx in self.resistant_indices:
                if rna_idx in test_rnas:
                    test_pos.append([rna_idx, drug_idx])
                else:
                    train_pos.append([rna_idx, drug_idx])

            test_pos = np.array(test_pos) if test_pos else np.empty((0, 2), dtype=int)
            train_pos = np.array(train_pos) if train_pos else np.empty((0, 2), dtype=int)

            used_neg = set()
            test_neg, used_neg = self._sample_balanced_negative(
                test_pos, used_neg, valid_rnas=test_rnas
            )
            train_neg, used_neg = self._sample_balanced_negative(
                train_pos, used_neg, valid_rnas=train_rnas
            )

            if len(test_pos) > 0 and len(test_neg) > 0:
                test_edges = np.vstack([test_pos, test_neg])
                test_labels = np.concatenate([np.ones(len(test_pos)), np.zeros(len(test_neg))])
            else:
                test_edges = np.empty((0, 2), dtype=int)
                test_labels = np.array([])

            if len(train_pos) > 0 and len(train_neg) > 0:
                train_edges = np.vstack([train_pos, train_neg])
                train_labels = np.concatenate([np.ones(len(train_pos)), np.zeros(len(train_neg))])
            else:
                train_edges = np.empty((0, 2), dtype=int)
                train_labels = np.array([])

            splits.append({
                'train_edges': train_edges,
                'train_labels': train_labels,
                'test_edges': test_edges,
                'test_labels': test_labels,
                'train_pos': train_pos,
                'test_rnas': list(test_rnas),
                'train_rnas': list(train_rnas)
            })

            print(f"  Fold {fold_idx + 1}: Train={len(train_edges)} (Pos:{len(train_pos)}), "
                  f"Test={len(test_edges)} (Pos:{len(test_pos)})")

        return splits

    # cold start for drug
    def create_drug_cold_start_splits(self):
        print("\nCold start for drug...")

        # Randomly divide the drug nodes into 5 folds

        drug_indices = np.arange(self.n_drug)
        np.random.shuffle(drug_indices)

        drug_folds = np.array_split(drug_indices, self.n_splits)

        splits = []
        for fold_idx in range(self.n_splits):
            test_drugs = set(drug_folds[fold_idx])
            train_drugs = set(drug_indices) - test_drugs

            test_pos = []
            train_pos = []

            for rna_idx, drug_idx in self.resistant_indices:
                if drug_idx in test_drugs:
                    test_pos.append([rna_idx, drug_idx])
                else:
                    train_pos.append([rna_idx, drug_idx])

            test_pos = np.array(test_pos) if test_pos else np.empty((0, 2), dtype=int)
            train_pos = np.array(train_pos) if train_pos else np.empty((0, 2), dtype=int)


            used_neg = set()
            test_neg, used_neg = self._sample_balanced_negative(
                test_pos, used_neg, valid_drugs=test_drugs
            )
            train_neg, used_neg = self._sample_balanced_negative(
                train_pos, used_neg, valid_drugs=train_drugs
            )

            if len(test_pos) > 0 and len(test_neg) > 0:
                test_edges = np.vstack([test_pos, test_neg])
                test_labels = np.concatenate([np.ones(len(test_pos)), np.zeros(len(test_neg))])
            else:
                test_edges = np.empty((0, 2), dtype=int)
                test_labels = np.array([])

            if len(train_pos) > 0 and len(train_neg) > 0:
                train_edges = np.vstack([train_pos, train_neg])
                train_labels = np.concatenate([np.ones(len(train_pos)), np.zeros(len(train_neg))])
            else:
                train_edges = np.empty((0, 2), dtype=int)
                train_labels = np.array([])

            splits.append({
                'train_edges': train_edges,
                'train_labels': train_labels,
                'test_edges': test_edges,
                'test_labels': test_labels,
                'train_pos': train_pos,
                'test_drugs': list(test_drugs),
                'train_drugs': list(train_drugs)
            })

            print(f"  Fold {fold_idx + 1}: Train={len(train_edges)} (Pos:{len(train_pos)}), "
                  f"Test={len(test_edges)} (Pos:{len(test_pos)})")

        return splits

    def create_both_cold_start_splits(self):

        print("\nCold start for both...")

        # Randomly divide the RNA and Drug nodes into 5 folds each
        rna_indices = np.arange(self.n_rna)
        drug_indices = np.arange(self.n_drug)
        np.random.shuffle(rna_indices)
        np.random.shuffle(drug_indices)

        rna_folds = np.array_split(rna_indices, self.n_splits)
        drug_folds = np.array_split(drug_indices, self.n_splits)

        rna_to_fold = {}
        for fold_idx, fold_rnas in enumerate(rna_folds):
            for rna in fold_rnas:
                rna_to_fold[rna] = fold_idx

        drug_to_fold = {}
        for fold_idx, fold_drugs in enumerate(drug_folds):
            for drug in fold_drugs:
                drug_to_fold[drug] = fold_idx

        splits = []
        for fold_idx in range(self.n_splits):
            test_rnas = set(rna_folds[fold_idx])
            test_drugs = set(drug_folds[fold_idx])

            train_rnas = set(rna_indices) - test_rnas
            train_drugs = set(drug_indices) - test_drugs

            test_pos = []
            train_pos = []
            discarded = 0

            for rna_idx, drug_idx in self.resistant_indices:
                rna_in_test = rna_idx in test_rnas
                drug_in_test = drug_idx in test_drugs

                if rna_in_test and drug_in_test:
                    test_pos.append([rna_idx, drug_idx])
                elif not rna_in_test and not drug_in_test:
                    train_pos.append([rna_idx, drug_idx])
                else:
                    discarded += 1

            test_pos = np.array(test_pos) if test_pos else np.empty((0, 2), dtype=int)
            train_pos = np.array(train_pos) if train_pos else np.empty((0, 2), dtype=int)

            def sample_neg_with_constraint(n_needed, valid_rnas, valid_drugs, used_neg):
                selected = []
                valid_rnas_set = set(valid_rnas)
                valid_drugs_set = set(valid_drugs)

                all_candidates = np.vstack([self.sensitive_indices, self.unknown_indices])
                np.random.shuffle(all_candidates)

                n_from_sensitive = 0
                n_from_unknown = 0
                
                for idx in all_candidates:
                    if len(selected) >= n_needed:
                        break
                    rna_idx, drug_idx = idx
                    
                    if rna_idx in valid_rnas_set and drug_idx in valid_drugs_set:
                        key = tuple(idx)
                        if key not in used_neg:
                            selected.append(idx)
                            used_neg.add(key)
                
                return np.array(selected) if selected else np.empty((0, 2), dtype=int), used_neg

            used_neg = set()
            test_neg, used_neg = sample_neg_with_constraint(len(test_pos), test_rnas, test_drugs, used_neg)
            train_neg, used_neg = sample_neg_with_constraint(len(train_pos), train_rnas, train_drugs, used_neg)

            if len(test_pos) > 0 and len(test_neg) > 0:
                test_edges = np.vstack([test_pos, test_neg])
                test_labels = np.concatenate([np.ones(len(test_pos)), np.zeros(len(test_neg))])
            else:
                test_edges = np.empty((0, 2), dtype=int)
                test_labels = np.array([])

            if len(train_pos) > 0 and len(train_neg) > 0:
                train_edges = np.vstack([train_pos, train_neg])
                train_labels = np.concatenate([np.ones(len(train_pos)), np.zeros(len(train_neg))])
            else:
                train_edges = np.empty((0, 2), dtype=int)
                train_labels = np.array([])

            splits.append({
                'train_edges': train_edges,
                'train_labels': train_labels,
                'test_edges': test_edges,
                'test_labels': test_labels,
                'train_pos': train_pos,
                'test_rnas': list(test_rnas),
                'train_rnas': list(train_rnas),
                'test_drugs': list(test_drugs),
                'train_drugs': list(train_drugs),
                'discarded_pairs': discarded
            })

            print(f"  Fold {fold_idx + 1}: Train={len(train_edges)} (Pos:{len(train_pos)}), "
                  f"Test={len(test_edges)} (Pos:{len(test_pos)}), Discarded={discarded}")

        return splits


def save_cold_start_splits(splits, cold_start_type, output_dir='cold_start_splits'):
    os.makedirs(output_dir, exist_ok=True)

    # pickle
    pkl_path = os.path.join(output_dir, f'{cold_start_type}_splits.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(splits, f)
    print(f"\nSaveing: {pkl_path}")

    csv_path = os.path.join(output_dir, f'{cold_start_type}_all_folds.csv')
    all_data = []

    for fold_idx, split in enumerate(splits):
        for (rna_idx, drug_idx), label in zip(split['train_edges'], split['train_labels']):
            all_data.append({
                'fold': fold_idx + 1,
                'rna_idx': int(rna_idx),
                'drug_idx': int(drug_idx),
                'label': int(label),
                'split': 'train'
            })
        for (rna_idx, drug_idx), label in zip(split['test_edges'], split['test_labels']):
            all_data.append({
                'fold': fold_idx + 1,
                'rna_idx': int(rna_idx),
                'drug_idx': int(drug_idx),
                'label': int(label),
                'split': 'test'
            })

    df = pd.DataFrame(all_data)
    df.to_csv(csv_path, index=False)


@torch.no_grad()
def evaluate(model, loader, drug_smiles_graphs, rna_has_seq_tensor, device):
    model.eval()
    preds = []
    truths = []

    for batch in loader:
        batch = batch.to(device)
        drug_smiles_batch, drug_unique_map = process_batch_drugs(batch, drug_smiles_graphs, device)

        target_edge_index = batch['drug', 'interacts', 'rna'].edge_label_index
        rna_indices = batch['rna'].n_id[target_edge_index[1]]
        rna_valid_mask = rna_has_seq_tensor[rna_indices]

        _, _, _, _, interaction_pred, _ = model(batch, drug_smiles_batch, drug_unique_map, rna_valid_mask)

        preds.append(interaction_pred.sigmoid().cpu())
        truths.append(batch['drug', 'interacts', 'rna'].edge_label.cpu())

    if not preds:
        return {'auc': 0, 'aupr': 0}

    preds = torch.cat(preds, dim=0).numpy()
    truths = torch.cat(truths, dim=0).numpy()

    try:
        auc = roc_auc_score(truths, preds)
        aupr = average_precision_score(truths, preds)
    except:
        auc, aupr = 0.5, 0.5


    return {
        'auc': auc,
        'aupr': aupr,
    }


def add_sim_edges(data, d_sim_idx, r_sim_idx):
    data['drug', 'similar_to', 'drug'].edge_index = d_sim_idx
    data['rna', 'similar_to', 'rna'].edge_index = r_sim_idx
    return data


def create_data_loader(data, batch_size, shuffle=True):
    """create DataLoader"""
    return LinkNeighborLoader(
        data,
        num_neighbors={
            ('drug', 'interacts', 'rna'): [20, 10],
            ('rna', 'rev_interacts', 'drug'): [20, 10],
            ('drug', 'similar_to', 'drug'): [10, 5],
            ('rna', 'similar_to', 'rna'): [10, 5]
        },
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        edge_label_index=(('drug', 'interacts', 'rna'), data['drug', 'interacts', 'rna'].edge_label_index),
        edge_label=data['drug', 'interacts', 'rna'].edge_label,
        disjoint=False
    )


def train_one_fold(
    fold_idx,
    split_data,
    drug_features_tensor,
    rna_features_tensor,
    rna_has_seq_tensor,
    drug_smiles_graphs,
    drug_sim_matrix, 
    all_rna_names,
    all_drug_names,
    config,
    cold_start_type='rna'
):
    """one fold"""
    print(f"\n--- Fold {fold_idx + 1} ---")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_edges = split_data['train_edges']
    train_labels = split_data['train_labels']
    test_edges = split_data['test_edges']
    test_labels = split_data['test_labels']
    train_pos = split_data['train_pos']

    if len(train_edges) == 0 or len(test_edges) == 0:
        return None

    # (rna_idx, drug_idx) -> (drug_idx, rna_idx)
    train_edges_dr = train_edges[:, [1, 0]]  # (drug, rna)
    test_edges_dr = test_edges[:, [1, 0]]
    train_pos_dr = train_pos[:, [1, 0]] if len(train_pos) > 0 else np.empty((0, 2), dtype=int)

    print(f"  Training: {len(train_edges)}  (Pos:{int(train_labels.sum())})")
    print(f"  Val: {len(test_edges)}  (Pos:{int(test_labels.sum())})")

    # RNA GIP 
    train_adj_for_gip = np.zeros((len(all_rna_names), len(all_drug_names)))
    for drug_idx, rna_idx in train_pos_dr:
        train_adj_for_gip[int(rna_idx), int(drug_idx)] = 1

    rna_gip_sim = calculate_gip_similarity(train_adj_for_gip)

    # Process GIP features for cold-start nodes
    row_sums = np.sum(np.abs(rna_gip_sim), axis=1)
    zero_rows = row_sums < 1e-8  
    non_zero_rows = ~zero_rows

    if zero_rows.any() and non_zero_rows.any():
        mean_gip_row = np.mean(rna_gip_sim[non_zero_rows], axis=0)
        rna_gip_sim[zero_rows] = mean_gip_row

    rna_gip_tensor = torch.tensor(rna_gip_sim, dtype=torch.float)
    rna_sim_edge_index = get_similarity_edges(rna_gip_sim, 0.6)


    drug_sim_edge_index = get_similarity_edges(drug_sim_matrix, 0.6)

    # HeteroData
    train_data = create_hetero_data(train_edges_dr, train_labels, train_pos_dr,
                                     drug_features_tensor, rna_gip_tensor)
    train_data = add_sim_edges(train_data, drug_sim_edge_index, rna_sim_edge_index)

    test_data = create_hetero_data(test_edges_dr, test_labels, train_pos_dr,
                                    drug_features_tensor, rna_gip_tensor)
    test_data = add_sim_edges(test_data, drug_sim_edge_index, rna_sim_edge_index)

    # Initialize the model
    device_rna_features = rna_features_tensor.to(device)
    device_rna_has_seq = rna_has_seq_tensor.to(device)

    model = UnifiedModel(
        drug_initial_dim=config['drug_initial_dim'],
        rna_feature_dim=config['rna_feature_dim'],
        rna_sim_feature_dim=rna_gip_tensor.shape[1],
        hidden_channels=config['hidden_channels'],
        out_channels=config['out_channels'],
        metadata=train_data.metadata(),
        full_rna_features=device_rna_features
    ).to(device)

    awl = AutomaticWeightedLoss(num=2).to(device)

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': awl.parameters(), 'weight_decay': 0}
    ], lr=config['learning_rate'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # BCE Loss
    bce_loss_fn = nn.BCEWithLogitsLoss()

    # DataLoader
    train_loader = create_data_loader(train_data, config['batch_size'], shuffle=True)
    test_loader = create_data_loader(test_data, config['batch_size'] * 4, shuffle=False)

    best_test = 0
    best_metrics = None
    best_epoch = 0
    patience_counter = 0
    patience = config.get('patience', 50)

    for epoch in range(config['epochs']):
        model.train()
        total_loss_sum = 0

        for batch in train_loader:
            batch = batch.to(device)
            batch = mask_target_edges(batch)

            drug_smiles_batch, drug_unique_map = process_batch_drugs(batch, drug_smiles_graphs, device)

            target_edge_index = batch['drug', 'interacts', 'rna'].edge_label_index
            rna_indices = batch['rna'].n_id[target_edge_index[1]]
            # rna_valid_mask = rna_has_seq_tensor[rna_indices]
            rna_valid_mask = device_rna_has_seq[rna_indices]

            drug_s_proj, drug_a_proj, rna_s_proj, rna_a_proj, interaction_pred, _ = model(
                batch, drug_smiles_batch, drug_unique_map, rna_valid_mask
            )

            ground_truth = batch['drug', 'interacts', 'rna'].edge_label
            loss_inter = bce_loss_fn(interaction_pred.squeeze(), ground_truth)

            # CL loss
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

            loss_cl_total = loss_drug_cl + loss_rna_cl
            total_loss = awl(loss_inter, loss_cl_total)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_loss_sum += total_loss.item()

        epoch_loss = total_loss_sum / len(train_loader)
        scheduler.step(epoch_loss)

        test_metrics = evaluate(model, test_loader, drug_smiles_graphs, device_rna_has_seq, device)


        if test_metrics['auc'] > best_test:
            best_test = test_metrics['auc']
            print(f"  Epoch {epoch + 1}: Loss={epoch_loss:.4f}, "
                  f"AUC={test_metrics['auc']:.4f}")
            best_metrics = test_metrics.copy()
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), f'save_cold/best_model_cold_start_{cold_start_type}_fold{fold_idx}.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f" Early stop: Epoch {epoch + 1}")
                break
            

    print(f"  Best (Epoch {best_epoch}): AUC={best_metrics['auc']:.4f}, AUPR={best_metrics['aupr']:.4f}")

    return best_metrics

# run cold start
def run_cold_start_experiment(cold_start_type, config):

    print("=" * 60)
    print(f"Type: {cold_start_type}")
    print("=" * 60)

    # 加载数据
    CACHE_PATH = 'Data/processed_data_cache.pkl'

    print("Loding...")
    try:
        with open(CACHE_PATH, 'rb') as f:
            cache_data = pickle.load(f)
    except:
        raise RuntimeError('run standard setting first!')    
    

    rna_features_tensor = cache_data['rna_features_tensor']
    rna_has_seq_tensor = cache_data['rna_has_seq_tensor']
    drug_features_tensor = cache_data['drug_features_tensor']
    drug_smiles_graphs = cache_data['drug_smiles_graphs']
    all_drug_names = cache_data['all_drug_names']
    all_rna_names = cache_data['all_rna_names']


    if config.get('use_enhanced_features', False):

        smiles_df = pd.read_csv('Data/drug_smiles.csv')
        smiles_map = {row['name']: row['smiles'] for _, row in smiles_df.iterrows()}
        smiles_list = [smiles_map.get(name, None) for name in all_drug_names]
        
        enhanced_features = get_enhanced_drug_features(smiles_list)
        
        drug_features_tensor = enhanced_features
        config['drug_initial_dim'] = enhanced_features.shape[1]

    # Adj
    adj_with_sens_df = pd.read_csv('Data/adj_with_sens.csv', index_col=0)

    splitter = ColdStartSplitter(adj_with_sens_df, n_splits=5, seed=42)
  
    if cold_start_type == 'rna':
        splits = splitter.create_rna_cold_start_splits()
    elif cold_start_type == 'drug':
        splits = splitter.create_drug_cold_start_splits()
    elif cold_start_type == 'both':
        splits = splitter.create_both_cold_start_splits()

    # saving
    save_cold_start_splits(splits, cold_start_type, output_dir=f'cold_start_splits')

    drug_sim_matrix = calculate_drug_similarity(drug_features_tensor.numpy())

    # Train and Val
    all_metrics = []
    for fold_idx, split in enumerate(splits):
        metrics = train_one_fold(
            fold_idx=fold_idx,
            split_data=split,
            drug_features_tensor=drug_features_tensor,
            rna_features_tensor=rna_features_tensor,
            rna_has_seq_tensor=rna_has_seq_tensor,
            drug_smiles_graphs=drug_smiles_graphs,
            drug_sim_matrix=drug_sim_matrix,  
            all_rna_names=all_rna_names,
            all_drug_names=all_drug_names,
            config=config,
            cold_start_type=cold_start_type
        )
        if metrics is not None:
            all_metrics.append(metrics)

    # 汇总结果
    if all_metrics:
        print("\n" + "=" * 60)
        print(f"Results: {cold_start_type} ")
        print("=" * 60)

        avg_metrics = {}
        for key in ['auc', 'aupr']:
            values = [m[key] for m in all_metrics]
            avg_metrics[key] = (np.mean(values), np.std(values))
            print(f"{key.upper()}: {np.mean(values):.4f} ± {np.std(values):.4f}")

        # 保存结果
        results = {
            'cold_start_type': cold_start_type,
            'fold_metrics': all_metrics,
            'avg_metrics': avg_metrics,
            'config': config
        }

        results_path = f'save_cold/cold_start_results_{cold_start_type}.pkl'
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)

    return all_metrics


def merge_all_cold_start_files(output_dir='cold_start_splits'):

    all_splits = {}
    all_data = []

    for cs_type in ['rna', 'drug', 'both']:
        pkl_path = os.path.join(output_dir, f'{cs_type}_splits.pkl')
        if os.path.exists(pkl_path):
            print(f"Saving split: Type: {cs_type} ...")
            with open(pkl_path, 'rb') as f:
                splits = pickle.load(f)
            all_splits[cs_type] = splits

            for fold_idx, split in enumerate(splits):
                for (rna_idx, drug_idx), label in zip(split['train_edges'], split['train_labels']):
                    all_data.append({
                        'type': cs_type,
                        'fold': fold_idx + 1,
                        'rna_idx': int(rna_idx),
                        'drug_idx': int(drug_idx),
                        'label': int(label),
                        'split': 'train'
                    })
                for (rna_idx, drug_idx), label in zip(split['test_edges'], split['test_labels']):
                    all_data.append({
                        'type': cs_type,
                        'fold': fold_idx + 1,
                        'rna_idx': int(rna_idx),
                        'drug_idx': int(drug_idx),
                        'label': int(label),
                        'split': 'test'
                    })
        else:
            print(f"Skipping")

    if all_splits:
        # PKL
        unified_pkl = os.path.join(output_dir, 'all_cold_start_splits.pkl')
        with open(unified_pkl, 'wb') as f:
            pickle.dump(all_splits, f)

        # CSV
        unified_csv = os.path.join(output_dir, 'all_cold_start_splits.csv')
        df = pd.DataFrame(all_data)
        df.to_csv(unified_csv, index=False)


        for cs_type in all_splits:
            type_data = df[df['type'] == cs_type]

    else:
        print("Bad!")

    return all_splits


def main():
    parser = argparse.ArgumentParser(description='Cold start')
    parser.add_argument('--type', type=str, default='all',
                        choices=['rna', 'drug', 'both', 'all'],
                        help='Type: rna, drug, both, all')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Max')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--merge', action='store_true')
    parser.add_argument('--no-enhanced', action='store_true')
    args = parser.parse_args()

    # Seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'patience': args.patience,
        'hidden_channels': 128,
        'out_channels': 64,
        'drug_initial_dim': 1024,  
        'rna_feature_dim': 256,
        'use_enhanced_features': not args.no_enhanced
    }

    if args.merge:
        merge_all_cold_start_files()
        return

    # RUN
    if args.type == 'all':
        for cs_type in ['rna', 'drug', 'both']:
            run_cold_start_experiment(cs_type, config)
        merge_all_cold_start_files()
    else:
        run_cold_start_experiment(args.type, config)


if __name__ == '__main__':
    main()