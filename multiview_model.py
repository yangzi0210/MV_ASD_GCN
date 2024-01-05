import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


# TODO：添加注意力机制
#  每个单视图添加注意力累加 / 多视图融合的时候添加注意力
#  https://blog.51cto.com/u_16099224/7193830
#  通道注意力 空间注意力 .....
class MultiViewGNN(torch.nn.Module):
    def __init__(self, args):
        super(MultiViewGNN, self).__init__()
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.dropout_ratio = args.dropout_ratio
        # 添加空间和通道注意力层
        self.spatial_attention = SpatialAttention(self.nhid)
        self.channel_attention = ChannelAttention(self.nhid)

        self.conv1_v1 = GCNConv(self.num_features, self.nhid)
        self.conv2_v1 = GCNConv(self.nhid, 1)

        self.conv1_v2 = GCNConv(self.num_features, self.nhid)
        self.conv2_v2 = GCNConv(self.nhid, 1)

        self.conv1_v3 = GCNConv(self.num_features, self.nhid)
        self.conv2_v3 = GCNConv(self.nhid, 1)

        self.lin1 = torch.nn.Linear(self.nhid, 1)

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
        # x_multiview = x_v1 + x_v2 + x_v3
        # 合并多视图 - 拼接
        # x_multiview = torch.cat((x_v1, x_v2, x_v3), dim=1)  # 注意dim=1表示沿特征的维度拼接
        # 合并多视图 - 自注意力
        # x_multiview = self.attention_mechanism(features_v1, features_v2, features_v3)
        # x_multiview = F.relu(self.lin1(x_multiview))
        # x = torch.flatten(x_multiview)
        # 在每个视图的特征上分别应用空间和通道注意力
        x_v1 = self.channel_attention(self.spatial_attention(features_v1))
        x_v2 = self.channel_attention(self.spatial_attention(features_v2))
        x_v3 = self.channel_attention(self.spatial_attention(features_v3))
        # 融合特征
        x_multiview = x_v1 + x_v2 + x_v3

        # 可以选择使用一个线性层进一步转换融合后的特征
        x_multiview = F.relu(self.lin1(x_multiview))

        # 将融合后的特征展平
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


class SpatialAttention(torch.nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.query = torch.nn.Linear(in_channels, in_channels)
        self.key = torch.nn.Linear(in_channels, in_channels)
        self.value = torch.nn.Linear(in_channels, in_channels)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attention_scores = torch.matmul(q, k.transpose(-2, -1))
        attention_scores = self.softmax(attention_scores)
        weighted_values = torch.matmul(attention_scores, v)
        return weighted_values


class ChannelAttention(torch.nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.fc1 = torch.nn.Linear(in_channels, in_channels // 8)
        self.fc2 = torch.nn.Linear(in_channels // 8, in_channels)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        maxpooled = torch.max(x, dim=-2, keepdim=True)[0]
        avgpooled = torch.mean(x, dim=-2, keepdim=True)
        mlp_max = F.relu(self.fc1(maxpooled.squeeze(-2)))
        mlp_avg = F.relu(self.fc1(avgpooled.squeeze(-2)))
        mlp = self.fc2(mlp_max + mlp_avg)
        scale = self.softmax(mlp).unsqueeze(-2)
        weighted_values = x * scale
        return weighted_values


class MultiViewAttention(torch.nn.Module):
    def __init__(self, nhid, nviews):
        super(MultiViewAttention, self).__init__()
        self.attention_fc = torch.nn.Linear(nhid * nviews, nviews)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x_views):
        # x_views is a list of features from all views [x_v1, x_v2, ..., x_vn]
        x_concat = torch.cat(x_views, dim=1)  # Concatenate the features
        attention_weights = self.softmax(self.attention_fc(x_concat))  # Learn attention weights
        weighted_features = torch.einsum('bn,bnm->bm', attention_weights,
                                         x_concat.view(x_concat.shape[0], -1, x_concat.shape[1] // len(x_views)))

        return weighted_features

# class MultiViewGNN(torch.nn.Module):
#     def __init__(self, args):
#         super(MultiViewGNN, self).__init__()
#         # ...[保留原来的初始化代码]...
#
#         self.num_views = 3  # 如果有三个视图
#         self.multiview_attention = MultiViewAttention(self.nhid, self.num_views)
#
#     def forward(self, x, edge_index_v1, edge_index_v2, edge_index_v3, edge_weight_v1, edge_weight_v2, edge_weight_v3):
#         # ...[保留原来的前向传播代码]...
#
#         # 获得每个视图的特征
#         features_v1 = F.relu(self.conv1_v1(x, edge_index_v1, edge_weight_v1))
#         features_v2 = F.relu(self.conv1_v2(x, edge_index_v2, edge_weight_v2))
#         features_v3 = F.relu(self.conv1_v3(x, edge_index_v3, edge_weight_v3))
#
#         # 注意力融合
#         x_multiview = self.multiview_attention([features_v1, features_v2, features_v3])
#
#         # 可以选择使用一个线性层进一步转换融合后的特征
#         x_multiview = F.relu(self.lin1(x_multiview))
#
#         # 将融合后的特征展平
#         x = torch.flatten(x_multiview)
#
#         return x