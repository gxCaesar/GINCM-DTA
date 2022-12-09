import pandas as pd
import numpy as np
import os
import json,pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles

# from utils import *

datasets = ['davis','kiba']
for dataset in datasets:
    fpath = 'data/' + dataset + '/'
    train_fold = json.load(open(fpath + "folds/train_fold_setting1.txt"))
    #train_fold = [ee for e in train_fold for ee in e ]
    valid_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))
    folds = train_fold + [valid_fold]
    valid_ids = [5,4,3,2,1]
    valid_folds = [folds[vid] for vid in valid_ids]
    train_folds = []
    for i in range(5):
        temp = []
        for j in range(6):
            if j != valid_ids[i]:
                temp += folds[j]
        train_folds.append(temp)
    
    ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(fpath + "Y","rb"), encoding='latin1')
    drugs = []
    prots = []
    for d in ligands.keys():
        #lg = ligands[d]
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]),isomericSmiles=True)
        drugs.append(lg)
    for t in proteins.keys():
        prots.append(proteins[t])
    if dataset == 'davis':
        affinity = [-np.log10(y/1e9) for y in affinity]

    # protein pretraing presentation
    # generate_protein_pretraining_representation(dataset, prots)

    affinity = np.asarray(affinity)
    opts = ['train','valid']
    for i in range(5):
        train_fold = train_folds[i]
        valid_fold = valid_folds[i]
        for opt in opts:
            rows, cols = np.where(np.isnan(affinity)==False)  
            if opt=='train':
                rows,cols = rows[train_fold], cols[train_fold]
            elif opt=='valid':
                rows,cols = rows[valid_fold], cols[valid_fold]
                
            if i == 0:
                #generating standard data
                print('generating standard data')
                with open('data/' + dataset + '_' + opt + '.csv', 'w') as f:
                    f.write('compound_iso_smiles,target_sequence,affinity,protein_id\n')
                    for pair_ind in range(len(rows)):
                        ls = []
                        ls += [ drugs[rows[pair_ind]]  ]
                        ls += [ prots[cols[pair_ind]]  ]
                        ls += [ affinity[rows[pair_ind],cols[pair_ind]]  ]
                        ls += [ cols[pair_ind] ]
                        f.write(','.join(map(str,ls)) + '\n')
            
            print('generating 5-fold validation data')
            with open('data/' + dataset + '/' + dataset + '_'  + 'fold_' + str(i) +'_'+ opt + '.csv', 'w') as f:
                f.write('compound_iso_smiles,target_sequence,affinity,protein_id\n')
                for pair_ind in range(len(rows)):
                    ls = []
                    ls += [ drugs[rows[pair_ind]]  ]
                    ls += [ prots[cols[pair_ind]]  ]
                    ls += [ affinity[rows[pair_ind],cols[pair_ind]]  ]
                    ls += [ cols[pair_ind] ]
                    f.write(','.join(map(str,ls)) + '\n')    