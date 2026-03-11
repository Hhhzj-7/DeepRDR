import numpy as np
import torch
import networkx as nx
from rdkit import Chem
from gensim.models.doc2vec import TaggedDocument, Doc2Vec


def atom_features(atom):
    """Generate feature for an atom"""
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    """One-hot"""
    if x not in allowable_set:
        raise Exception(f"input {x} not in allowable set: {allowable_set}")
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    """SMILES -> 2D graph"""
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None, None

    features = [atom_features(atom) for atom in mol.GetAtoms()]

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

    if not edges:  
        return torch.FloatTensor(features), torch.LongTensor([]).reshape(2, 0)

    g = nx.Graph(edges).to_directed()
    edge_index = torch.LongTensor(list(g.edges)).t().contiguous()

    return torch.FloatTensor(features), edge_index



def k_mers(k, seq):
    if k > len(seq): return []
    return [seq[i:i + k] for i in range(len(seq) - k + 1)]


def train_doc2vec_model(rna_data, k_mer_size, vec_dim, epochs=40):
    """
    :param rna_data: {rna_name: sequence}
    :param k_mer_size: k-mer size
    :param vec_dim: Doc2Vec output dim
    :return: Doc2Vec model
    """
    print(f" {len(rna_data)} ncRNA sequence -> Doc2Vec...")
    tagged_docs = [TaggedDocument(k_mers(k_mer_size, seq), [name]) for name, seq in rna_data.items() if seq]
    model = Doc2Vec(tagged_docs, vector_size=vec_dim, window=5, min_count=1, workers=4, epochs=epochs)
    return model