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
        self.attention_mechanism = SelfAttention(self.nhid)
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

        # 合并多视图 - 加和
        x_multiview = x_v1 + x_v2 + x_v3
        # 合并多视图 - 拼接
        # x_multiview = torch.cat((x_v1, x_v2, x_v3), dim=1)  # 注意dim=1表示沿特征的维度拼接
        # 合并多视图 - 自注意力
        # x_multiview = self.attention_mechanism(x_v1, x_v2, x_v3)
        x = torch.flatten(x_multiview)

        features = features_v1 + features_v2 + features_v3
        return x, features


class SelfAttention(torch.nn.Module):
    def __init__(self, nhid):
        super(SelfAttention, self).__init__()
        self.query = torch.nn.Linear(nhid, nhid)
        self.key = torch.nn.Linear(nhid, nhid)
        self.value = torch.nn.Linear(nhid, nhid)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x1, x2, x3):
        # 计算自注意力得分
        q1 = self.query(x1)
        k2 = self.key(x2)
        v3 = self.value(x3)
        attention_scores = torch.matmul(q1, k2.transpose(-2, -1))
        attention_scores = self.softmax(attention_scores)

        # 应用得分到value
        weighted_values = torch.matmul(attention_scores, v3)
        return weighted_values
