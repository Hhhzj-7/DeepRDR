import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, to_hetero, global_mean_pool

# Drug GCN
class DrugStructureEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index)

        return global_mean_pool(x, batch)   # [batch_size, out_channels]



# GraphSAGE on Heterogeneous graph
class InteractionGNN(nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x


# Gated cross-context cross-entity attention mechanism
class GatedCrossAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # Query, Key, Value
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

        # Token-wise gated
        self.gate = nn.Linear(hidden_size, 1)

        # projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size)

    def attention(self, Q, K, V):
        """
        Q: [batch_size, hidden_size]
        K: [2, batch_size, hidden_size]
        V: [2, batch_size, hidden_size]
        """
        # Q: [batch_size, hidden_size] -> [batch_size, 1, hidden_size]
        Q = Q.unsqueeze(1)

        # K: [2, batch_size, hidden_size] -> [batch_size, 2, hidden_size]
        K = K.permute(1, 0, 2)

        # V: [2, batch_size, hidden_size] -> [batch_size, 2, hidden_size]
        V = V.permute(1, 0, 2)

        # Attention scores: [batch_size, 1, 2]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_size ** 0.5)
        probs = F.softmax(scores, dim=-1)

        # gated fusion: [batch_size, 1, hidden_size] -> [batch_size, hidden_size]
        out = torch.matmul(probs, V).squeeze(1)
        return out

    def forward(self, rna_seq_emb, rna_assoc_emb, drug_struct_emb, drug_assoc_emb):
        """
        Input: [batch_size, hidden_size]
        Output: [batch_size, hidden_size]
        """
        
        tokens = [rna_seq_emb, rna_assoc_emb, drug_struct_emb, drug_assoc_emb]

        # Step 1: Q, K, V
        Q = [self.q_proj(t) for t in tokens]
        K = [self.k_proj(t) for t in tokens]
        V = [self.v_proj(t) for t in tokens]

        # Step 2: cross attention
        # ncRNA tokens (intrinsic feature, resistance-relational feature) attend to Drug tokens (intrinsic feature, resistance-relational feature)
        drug_keys = torch.stack([K[2], K[3]])    # [2, batch, hidden]
        drug_values = torch.stack([V[2], V[3]])  # [2, batch, hidden]

        rna_seq_attn = self.attention(Q[0], drug_keys, drug_values)
        rna_assoc_attn = self.attention(Q[1], drug_keys, drug_values)

        # Drug tokens (intrinsic feature, resistance-relational feature) attend to RNA tokens (intrinsic feature, resistance-relational feature)
        rna_keys = torch.stack([K[0], K[1]])     # [2, batch, hidden]
        rna_values = torch.stack([V[0], V[1]])   # [2, batch, hidden]

        drug_struct_attn = self.attention(Q[2], rna_keys, rna_values)
        drug_assoc_attn = self.attention(Q[3], rna_keys, rna_values)

        # Step 3: token-wise gated
        attn_outputs = [rna_seq_attn, rna_assoc_attn, drug_struct_attn, drug_assoc_attn]
        gated_outputs = []

        for t, attn_out in zip(tokens, attn_outputs):
            g = torch.sigmoid(self.gate(t))  # [batch, 1]
            gated_outputs.append(attn_out * g)

        # Step 4: gated fusion
        fused = sum(gated_outputs)  # [batch, hidden_size]

        fused = self.out_proj(fused)
        fused = self.layer_norm(fused)

        return fused


# Main model
class UnifiedModel(torch.nn.Module):
    def __init__(self, drug_initial_dim, rna_feature_dim, rna_sim_feature_dim, hidden_channels, out_channels, metadata,
                 full_rna_features):
        super().__init__()

        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.drug_lin = nn.Linear(drug_initial_dim, hidden_channels)
        self.rna_lin = nn.Linear(rna_sim_feature_dim, hidden_channels)

        self.missing_drug_struct_emb = nn.Parameter(torch.randn(1, out_channels))
        self.missing_rna_seq_emb = nn.Parameter(torch.randn(1, out_channels))


        dnn_hidden_dim = rna_feature_dim * 5

        # RNA intrinsic feature
        self.rna_seq_dnn = nn.Sequential(
            nn.Linear(rna_feature_dim, dnn_hidden_dim),

            nn.LayerNorm(dnn_hidden_dim),

            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(dnn_hidden_dim, rna_feature_dim),

            nn.LayerNorm(rna_feature_dim),

            nn.ReLU(),
            nn.Linear(rna_feature_dim, self.out_channels)
        )

        self.register_buffer('full_rna_features', full_rna_features)

        # GraphSAGE on Heterogeneous graph
        self.interaction_gnn = InteractionGNN(hidden_channels, out_channels)
        self.interaction_gnn = to_hetero(self.interaction_gnn, metadata=metadata)

        # drug intrinsic feature
        self.drug_structure_encoder = DrugStructureEncoder(
            in_channels=78,
            hidden_channels=hidden_channels,
            out_channels=out_channels
        )

        proj_hidden_dim = hidden_channels * 2

        def build_deep_head(in_dim, hidden_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  
                nn.ELU(), 
                nn.Linear(hidden_dim, hidden_dim), 
                nn.LayerNorm(hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, out_dim)
            )

        # Projection Heads for Contrastive Learning
        self.drug_assoc_proj_head = build_deep_head(out_channels, proj_hidden_dim, out_channels)
        self.rna_assoc_proj_head = build_deep_head(out_channels, proj_hidden_dim, out_channels)
        self.drug_struct_proj_head = build_deep_head(out_channels, proj_hidden_dim, out_channels)
        self.rna_seq_proj_head = build_deep_head(out_channels, proj_hidden_dim, out_channels)

        self.cl_dropout = nn.Dropout(p=0.2)  

        # Gated cross-context cross-entity attention mechanism
        self.gated_cross_attention = GatedCrossAttention(out_channels)

    
        self.classifier = nn.Sequential(
            nn.Linear(out_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_channels, 1)
        )

    def get_association_embeddings(self, data):
        x_dict = {
            "drug": self.drug_lin(data["drug"].x),
            "rna": self.rna_lin(data["rna"].x),
        }
        return self.interaction_gnn(x_dict, data.edge_index_dict)

    def get_structure_embeddings(self, drug_smiles_batch, drug_unique_map):

        # 1. Unique Drug Embeddings
        if drug_smiles_batch is not None and hasattr(drug_smiles_batch, 'x'):
            # [N_unique, out_channels]
            unique_drug_emb = self.drug_structure_encoder(
                drug_smiles_batch.x,
                drug_smiles_batch.edge_index,
                drug_smiles_batch.batch
            )
        else:
            return None

        batch_size = drug_unique_map.size(0)
        # [Batch_Size, out_channels]
        out_emb = self.missing_drug_struct_emb.expand(batch_size, -1).clone()
        valid_indices_in_batch = drug_unique_map >= 0

        source_indices_in_unique = drug_unique_map[valid_indices_in_batch]
        out_emb[valid_indices_in_batch] = unique_drug_emb[source_indices_in_unique]

        return out_emb

    def forward(self, data, drug_smiles_batch, drug_unique_map, rna_valid_mask):

        # resistance-relational feature
        assoc_embs = self.get_association_embeddings(data)
        edge_label_index = data['drug', 'interacts', 'rna'].edge_label_index

        # [Batch_Size, Hidden]
        drug_assoc_emb_batch = assoc_embs['drug'][edge_label_index[0]]
        rna_assoc_emb_batch = assoc_embs['rna'][edge_label_index[1]]

        # drug intrinsic feature
        unique_drug_struct_emb = None
        if drug_smiles_batch is not None:
            unique_drug_struct_emb = self.drug_structure_encoder(
                drug_smiles_batch.x,
                drug_smiles_batch.edge_index,
                drug_smiles_batch.batch
            )

        batch_size = drug_assoc_emb_batch.size(0)

        drug_struct_emb_all = self.missing_drug_struct_emb.expand(batch_size, -1).clone()

        if unique_drug_struct_emb is not None:
            valid_indices = drug_unique_map >= 0
            source_indices = drug_unique_map[valid_indices]
            drug_struct_emb_all[valid_indices] = unique_drug_struct_emb[source_indices]

        # RNA intrinsic feature
        rna_seq_emb_all = self.missing_rna_seq_emb.expand(batch_size, -1).clone()

        if rna_valid_mask.any():
            edge_label_index = data['drug', 'interacts', 'rna'].edge_label_index
            rna_indices_in_batch = edge_label_index[1]
            valid_rna_subgraph_indices = rna_indices_in_batch[rna_valid_mask]
            batch_rna_global_indices = data['rna'].n_id[valid_rna_subgraph_indices]
            rna_doc2vec_batch = self.full_rna_features[batch_rna_global_indices]

            # DNN
            rna_seq_emb_valid = self.rna_seq_dnn(rna_doc2vec_batch)
            rna_seq_emb_all[rna_valid_mask] = rna_seq_emb_valid

        # CL
        # drug
        d_struct_aug = self.cl_dropout(drug_struct_emb_all)
        d_assoc_aug = self.cl_dropout(drug_assoc_emb_batch)

        drug_cl_proj_s = self.drug_struct_proj_head(d_struct_aug)  # [Batch, Out]
        drug_cl_proj_a = self.drug_assoc_proj_head(d_assoc_aug)  # [Batch, Out]

        # RNA
        r_seq_aug = self.cl_dropout(rna_seq_emb_all)
        r_assoc_aug = self.cl_dropout(rna_assoc_emb_batch)

        rna_cl_proj_seq = self.rna_seq_proj_head(r_seq_aug)  # [Batch, Out]
        rna_cl_proj_assoc = self.rna_assoc_proj_head(r_assoc_aug)  # [Batch, Out]

        # Gated cross-context cross-entity attention mechanism
        fused_embedding = self.gated_cross_attention(
            rna_seq_emb=rna_seq_emb_all,
            rna_assoc_emb=rna_assoc_emb_batch,
            drug_struct_emb=drug_struct_emb_all,
            drug_assoc_emb=drug_assoc_emb_batch
        )
        # Prediction
        interaction_pred = self.classifier(fused_embedding)

        return (
            drug_cl_proj_s, drug_cl_proj_a,
            rna_cl_proj_seq, rna_cl_proj_assoc,
            interaction_pred, fused_embedding
        )