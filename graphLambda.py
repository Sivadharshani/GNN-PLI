import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Dropout
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv, global_add_pool

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # GCN Representation
        self.gcn1 = GCNConv(373, 256, cached=False)
        self.bn_gcn1 = BatchNorm1d(256)
        self.gcn2 = GCNConv(256, 128, cached=False)
        self.bn_gcn2 = BatchNorm1d(128)
        self.gcn3 = GCNConv(128, 128, cached=False)
        self.bn_gcn3 = BatchNorm1d(128)

        # GAT Representation
        self.gat1 = GATConv(373, 256, heads=3, concat=True)
        self.bn_gat1 = BatchNorm1d(256 * 3)
        self.gat2 = GATConv(256 * 3, 128, heads=3, concat=True)
        self.bn_gat2 = BatchNorm1d(128 * 3)
        self.gat3 = GATConv(128 * 3, 128, heads=3, concat=True)
        self.bn_gat3 = BatchNorm1d(128 * 3)

        # GIN Representation
        fc_gin1 = Sequential(Linear(373, 256), ReLU(), Linear(256, 256))
        self.gin1 = GINConv(fc_gin1)
        self.bn_gin1 = BatchNorm1d(256)
        fc_gin2 = Sequential(Linear(256, 128), ReLU(), Linear(128, 128))
        self.gin2 = GINConv(fc_gin2)
        self.bn_gin2 = BatchNorm1d(128)
        fc_gin3 = Sequential(Linear(128, 64), ReLU(), Linear(64, 64))
        self.gin3 = GINConv(fc_gin3)
        self.bn_gin3 = BatchNorm1d(64)

        # GraphSAGE Representation
        self.sage1 = SAGEConv(373, 256)
        self.bn_sage1 = BatchNorm1d(256)
        self.sage2 = SAGEConv(256, 128)
        self.bn_sage2 = BatchNorm1d(128)
        self.sage3 = SAGEConv(128, 64)
        self.bn_sage3 = BatchNorm1d(64)

        # Fully connected layers for concatenating outputs
        self.fc1 = Linear(128 + 128 * 3 + 64 + 64, 256)
        self.dropout1 = Dropout(p=0.2)
        self.fc2 = Linear(256, 64)
        self.dropout2 = Dropout(p=0.2)
        self.fc3 = Linear(64, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch

        # GCN Representation
        gcn_x = F.relu(self.gcn1(x, edge_index))
        gcn_x = self.bn_gcn1(gcn_x)
        gcn_x = F.relu(self.gcn2(gcn_x, edge_index))
        gcn_x = self.bn_gcn2(gcn_x)
        gcn_x = F.relu(self.gcn3(gcn_x, edge_index))
        gcn_x = self.bn_gcn3(gcn_x)
        gcn_x = global_add_pool(gcn_x, batch)

        # GAT Representation
        gat_x = F.relu(self.gat1(x, edge_index))
        gat_x = self.bn_gat1(gat_x)
        gat_x = F.relu(self.gat2(gat_x, edge_index))
        gat_x = self.bn_gat2(gat_x)
        gat_x = F.relu(self.gat3(gat_x, edge_index))
        gat_x = self.bn_gat3(gat_x)
        gat_x = global_add_pool(gat_x, batch)

        # GIN Representation
        gin_x = F.relu(self.gin1(x, edge_index))
        gin_x = self.bn_gin1(gin_x)
        gin_x = F.relu(self.gin2(gin_x, edge_index))
        gin_x = self.bn_gin2(gin_x)
        gin_x = F.relu(self.gin3(gin_x, edge_index))
        gin_x = self.bn_gin3(gin_x)
        gin_x = global_add_pool(gin_x, batch)

        # GraphSAGE Representation
        sage_x = F.relu(self.sage1(x, edge_index))
        sage_x = self.bn_sage1(sage_x)
        sage_x = F.relu(self.sage2(sage_x, edge_index))
        sage_x = self.bn_sage2(sage_x)
        sage_x = F.relu(self.sage3(sage_x, edge_index))
        sage_x = self.bn_sage3(sage_x)
        sage_x = global_add_pool(sage_x, batch)

        # Concatenate outputs from all architectures
        out = torch.cat([gcn_x, gat_x, gin_x, sage_x], dim=1)
        out = F.relu(self.fc1(out))
        out = self.dropout1(out)
        out = F.relu(self.fc2(out))
        out = self.dropout2(out)
        out = self.fc3(out)

        return out
