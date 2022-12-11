import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv,global_mean_pool as gep
from torch.nn import Sequential, Linear, ReLU

class GINNet(torch.nn.Module):
    def __init__(self, n_output=1, num_features_pro=54, num_features_mol=82, output_dim=128, dropout=0.2):
        super(GINNet, self).__init__()

        print('GINNet Loaded')
        self.n_output = n_output
        nn1 = Sequential(Linear(num_features_mol, num_features_mol), ReLU(), Linear(num_features_mol, num_features_mol))
        self.mol_conv1 = GINConv(nn1)
        nn2 = Sequential(Linear(num_features_mol, num_features_mol*2), ReLU(), Linear(num_features_mol*2, num_features_mol*2))
        self.mol_conv2 = GINConv(nn2)
        nn3 = Sequential(Linear(num_features_mol*2, num_features_mol*4), ReLU(), Linear(num_features_mol*4, num_features_mol*4))
        self.mol_conv3 = GINConv(nn3)
        self.mol_fc_g1 = torch.nn.Linear(num_features_mol * 4, 1024)
        self.mol_fc_g2 = torch.nn.Linear(1024, output_dim)

        pro_nn1 = Sequential(Linear(num_features_pro, num_features_pro), ReLU(), Linear(num_features_pro, num_features_pro))
        pro_nn2 = Sequential(Linear(num_features_pro, num_features_pro*2), ReLU(), Linear(num_features_pro*2, num_features_pro*2))
        pro_nn3 = Sequential(Linear(num_features_pro*2, num_features_pro*4), ReLU(), Linear(num_features_pro*4, num_features_pro*4))
        self.pro_conv1 = GINConv(pro_nn1)
        self.pro_conv2 = GINConv(pro_nn2)
        self.pro_conv3 = GINConv(pro_nn3)
        self.pro_fc_g1 = torch.nn.Linear(num_features_pro * 4, 1024)
        self.pro_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # combined layers
        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data_mol, data_pro):
        # get graph input
        mol_x, mol_edge_index, mol_batch = data_mol.x, data_mol.edge_index, data_mol.batch
        # get protein input
        target_x, target_edge_index, target_batch = data_pro.x, data_pro.edge_index, data_pro.batch

        x = self.mol_conv1(mol_x, mol_edge_index)
        x = self.relu(x)


        x = self.mol_conv2(x, mol_edge_index)
        x = self.relu(x)


        x = self.mol_conv3(x, mol_edge_index)
        x = self.relu(x)
        x = gep(x, mol_batch)  # global pooling

        # flatten
        x = self.relu(self.mol_fc_g1(x))
        x = self.dropout(x)
        x = self.mol_fc_g2(x)
        x = self.dropout(x)

        xt = self.pro_conv1(target_x, target_edge_index)
        xt = self.relu(xt)


        xt = self.pro_conv2(xt, target_edge_index)
        xt = self.relu(xt)

        xt = self.pro_conv3(xt, target_edge_index)
        xt = self.relu(xt)

        xt = gep(xt, target_batch)  # global pooling

        # flatten
        xt = self.relu(self.pro_fc_g1(xt))
        xt = self.dropout(xt)
        xt = self.pro_fc_g2(xt)
        xt = self.dropout(xt)

        # concat
        xc = torch.cat((x, xt), 1)
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out



