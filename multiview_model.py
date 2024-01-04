import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# TODO：添加注意力机制
class MultiViewGNN(torch.nn.Module):
    def __init__(self, args):
        super(MultiViewGNN, self).__init__()
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.dropout_ratio = args.dropout_ratio
        self.conv1_v1 = GCNConv(self.num_features, self.nhid)
        self.conv2_v1 = GCNConv(self.nhid, 1)

        self.conv1_v2 = GCNConv(self.num_features, self.nhid)
        self.conv2_v2 = GCNConv(self.nhid, 1)

        self.conv1_v3 = GCNConv(self.num_features, self.nhid)
        self.conv2_v3 = GCNConv(self.nhid, 1)

    def forward(self, x, edge_index_v1, edge_index_v2, edge_index_v3, edge_weight_v1, edge_weight_v2, edge_weight_v3):
        # 视图1
        x_v1 = F.relu(self.conv1_v1(x, edge_index_v1, edge_weight_v1))
        # store the learned node embeddings
        features_v1 = x_v1
        x_v1 = F.dropout(x_v1, p=self.dropout_ratio, training=self.training)
        x_v1 = self.conv2_v1(x_v1, edge_index_v1)

        # 视图2
        x_v2 = F.relu(self.conv1_v2(x, edge_index_v2, edge_weight_v2))
        # store the learned node embeddings
        features_v2 = x_v2
        x_v2 = F.dropout(x_v2, p=self.dropout_ratio, training=self.training)
        x_v2 = self.conv2_v2(x_v2, edge_index_v2)

        # 视图3
        x_v3 = F.relu(self.conv1_v3(x, edge_index_v3, edge_weight_v3))
        # store the learned node embeddings
        features_v3 = x_v3
        x_v3 = F.dropout(x_v3, p=self.dropout_ratio, training=self.training)
        x_v3 = self.conv2_v3(x_v3, edge_index_v3)

        # 合并多视图
        x_multiview = x_v1 + x_v2 + x_v3  # 可以是加和、拼接等
        x = torch.flatten(x_multiview)
        features = features_v1 + features_v2 + features_v3
        return x, features