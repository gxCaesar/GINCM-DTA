#import torch.utils.data as DATA
from transformers import AdamW, AutoModel, AutoTokenizer
import os
import pandas as pd
from torch_geometric.data import InMemoryDataset
from torch_geometric import data as DATA
#import rdkit
#from rdkit import Chem
#from rdkit.Chem import AllChem
import numpy as np
import torch
from sklearn.utils import shuffle
import torch
import random
from utils import *


def get_finetune(dataset):
    msa_path = 'data/' + dataset + '/aln'
    contac_path = 'data/' + dataset + '/pconsc4'
    finetune_csv = 'data/'+ 'covid' + '.csv'

    df_finetune_fold = pd.read_csv(finetune_csv)
    finetune_drugs, finetune_prot_keys, finetune_Y, finetune_pro = list(df_finetune_fold['compound_iso_smiles']), list(
        df_finetune_fold['target_key']), list(df_finetune_fold['affinity']), list(df_finetune_fold['target_sequence'])

    finetune_drugs, finetune_prot_keys, finetune_Y = np.asarray(finetune_drugs), np.asarray(finetune_prot_keys), np.asarray(finetune_Y)
    compound_iso_smiles = set(finetune_drugs)
    target_key = set(finetune_prot_keys)

    proteins = {}
    for i in range(df_finetune_fold.shape[0]):
        proteins[finetune_prot_keys[i]] = finetune_pro[i]
    smile_graph = {}
    for smile in compound_iso_smiles:
        g = smile_to_graph(smile)
        if g is None:
            continue
        smile_graph[smile] = g

    target_graph = {}
    for key in target_key:
        if not valid_target(key, dataset):  # ensure the contact and aln files exists
            continue
        g = target_to_graph(key, proteins[key], contac_path, msa_path)
        target_graph[key] = g

    # count the number of  proteins with aln and contact files
    print('effective drugs,effective prot:', len(smile_graph), len(target_graph))
    if len(smile_graph) == 0 or len(target_graph) == 0:
        raise Exception('no protein or drug, run the script for datasets preparation.')

    finetune_dataset = DTADataset(root='data', dataset=dataset + '_' + 'train', xd=finetune_drugs, target_key=finetune_prot_keys,
                               y=finetune_Y, smile_graph=smile_graph, target_graph=target_graph)
    return finetune_dataset



def get_test(dataset,):
    msa_path = 'data/' + dataset + '/aln'
    contac_path = 'data/' + dataset + '/pconsc4'
    test_csv = 'data/' + dataset  + '_' + 'test' + '.csv'

    df_test_fold = pd.read_csv(test_csv)
    test_drugs, test_prot_keys, test_Y, test_pro = list(df_test_fold['compound_iso_smiles']), list(
        df_test_fold['target_key']), list(df_test_fold['affinity']), list(df_test_fold['target_sequence'])

    test_drugs, test_prot_keys, test_Y = np.asarray(test_drugs), np.asarray(test_prot_keys), np.asarray(test_Y)
    compound_iso_smiles = set(test_drugs)
    target_key = set(test_prot_keys)

    proteins = {}
    for i in range(df_test_fold.shape[0]):
        proteins[test_prot_keys[i]] = test_pro[i]
    smile_graph = {}
    for smile in compound_iso_smiles:
        g = smile_to_graph(smile)
        smile_graph[smile] = g

    target_graph = {}
    for key in target_key:
        if not valid_target(key, dataset):  # ensure the contact and aln files exists
            continue
        g = target_to_graph(key, proteins[key], contac_path, msa_path)
        target_graph[key] = g

    # count the number of  proteins with aln and contact files
    print('effective drugs,effective prot:', len(smile_graph), len(target_graph))
    if len(smile_graph) == 0 or len(target_graph) == 0:
        raise Exception('no protein or drug, run the script for datasets preparation.')




    test_dataset = DTADataset(root='data', dataset=dataset + '_' + 'train', xd=test_drugs, target_key=test_prot_keys,
                               y=test_Y, smile_graph=smile_graph, target_graph=target_graph)
    return test_dataset

def get_train_valid(dataset,fold,):
    msa_path = 'data/' + dataset + '/aln'
    contac_path = 'data/' + dataset + '/pconsc4'
    train_csv = 'data/' + dataset + '_' + 'fold_' + str(fold) + '_' + 'train' + '.csv'

    df_train_fold = pd.read_csv(train_csv)
    train_drugs, train_prot_keys, train_Y,train_pro = list(df_train_fold['compound_iso_smiles']), list(
        df_train_fold['target_key']), list(df_train_fold['affinity']) , list(df_train_fold['target_sequence'])

    train_drugs, train_prot_keys, train_Y = np.asarray(train_drugs), np.asarray(train_prot_keys), np.asarray(train_Y)

    df_valid_fold = pd.read_csv('data/' + dataset + '_' + 'fold_' + str(fold) + '_' + 'valid' + '.csv')
    valid_drugs, valid_prot_keys, valid_Y,valid_pro = list(df_valid_fold['compound_iso_smiles']), list(
        df_valid_fold['target_key']), list(df_valid_fold['affinity']), list(df_valid_fold['target_sequence'])
    valid_drugs, valid_prot_keys, valid_Y = np.asarray(valid_drugs), np.asarray(valid_prot_keys), np.asarray(
        valid_Y)
    compound_iso_smiles = set.union(set(valid_drugs),set(train_drugs))
    target_key = set.union(set(valid_prot_keys),set(train_prot_keys))

    proteins = {}
    for i in range(df_train_fold.shape[0]):
        proteins[train_prot_keys[i]] = train_pro[i]
    for i in range(df_valid_fold.shape[0]):
        proteins[valid_prot_keys[i]] = valid_pro[i]

    # create smile graph
    smile_graph = {}
    for smile in compound_iso_smiles:
        g = smile_to_graph(smile)
        smile_graph[smile] = g


    # create target graph
    target_graph = {}
    for key in target_key:
        if not valid_target(key, dataset):  # ensure the contact and aln files exists
            continue
        g = target_to_graph(key, proteins[key], contac_path, msa_path)
        target_graph[key] = g

    # count the number of  proteins with aln and contact files
    print('effective drugs,effective prot:', len(smile_graph), len(target_graph))
    if len(smile_graph) == 0 or len(target_graph) == 0:
        raise Exception('no protein or drug, run the script for datasets preparation.')




    train_dataset = DTADataset(root='data', dataset=dataset + '_' + 'train', xd=train_drugs, target_key=train_prot_keys,
                               y=train_Y, smile_graph=smile_graph, target_graph=target_graph)


    valid_dataset = DTADataset(root='data', dataset=dataset + '_' + 'train', xd=valid_drugs,
                               target_key=valid_prot_keys, y=valid_Y, smile_graph=smile_graph,
                               target_graph=target_graph)

    return train_dataset, valid_dataset


class DTADataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis',
                 xd=None, y=None, transform=None,
                 pre_transform=None, smile_graph=None, target_key=None, target_graph=None):

        super(DTADataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.process(xd, target_key, y, smile_graph, target_graph)

    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '_data_mol.pt', self.dataset + '_data_pro.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, xd, target_key, y, smile_graph, target_graph):
        assert (len(xd) == len(target_key) and len(xd) == len(y)), 'The three lists must be the same length!'
        data_list_mol = []
        data_list_pro = []
        data_len = len(xd)
        for i in range(data_len):
            smiles = xd[i]
            tar_key = target_key[i]
            labels = y[i]
            # convert SMILES to molecular representation using rdkit
            if smile_graph.__contains__(smiles) is False:
                continue
            c_size, features, edge_index = smile_graph[smiles]
            target_size, target_features, target_edge_index = target_graph[tar_key]
            data_mol = DATA.Data(x=torch.Tensor(features),
                                    edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                    y=torch.FloatTensor([labels]))
            data_mol.__setitem__('c_size', torch.LongTensor([c_size]))

            data_pro = DATA.Data(x=torch.Tensor(target_features),
                                    edge_index=torch.LongTensor(target_edge_index).transpose(1, 0),
                                    y=torch.FloatTensor([labels]))
            data_pro.__setitem__('target_size', torch.LongTensor([target_size]))
            data_list_mol.append(data_mol)
            data_list_pro.append(data_pro)

        if self.pre_filter is not None:
            data_list_mol = [data for data in data_list_mol if self.pre_filter(data)]
            data_list_pro = [data for data in data_list_pro if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list_mol = [self.pre_transform(data) for data in data_list_mol]
            data_list_pro = [self.pre_transform(data) for data in data_list_pro]
        self.data_mol = data_list_mol
        self.data_pro = data_list_pro

    def __len__(self):
        return len(self.data_mol)

    def __getitem__(self, idx):
        return self.data_mol[idx], self.data_pro[idx]






device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


