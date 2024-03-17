import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


#  添加注意力机制后的多视图代码
#  每个单视图添加注意力累加 / 多视图融合的时候添加注意力
#  https://blog.51cto.com/u_16099224/7193830
#  通道注意力 空间注意力 .....

class MultiViewGNN(torch.nn.Module):
    def __init__(self, args):
        super(MultiViewGNN, self).__init__()
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.dropout_ratio = args.dropout_ratio
        # self.attention = torch.nn.Parameter(torch.ones(3, 1, requires_grad=True))  # 增加一个注意力参数 可以在训练中被优化
        self.attention_layer = AttentionLayer(num_views=3)

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

        # And in the forward method, you would use the attention layer:
        x_multiview = self.attention_layer(x_v1, x_v2, x_v3)
        # 将融合后的特征展平
        x = torch.flatten(x_multiview)

        features = features_v1 + features_v2 + features_v3

        return x, features


class AttentionLayer(torch.nn.Module):
    def __init__(self, num_views=3):
        super(AttentionLayer, self).__init__()
        self.attention_weights = torch.nn.Parameter(torch.ones(num_views, 1, requires_grad=True))
        torch.nn.init.xavier_uniform_(self.attention_weights.data)

    def forward(self, *views):
        # Calculate attention scores
        attention_scores = F.softmax(self.attention_weights, dim=0)
        # Apply attention scores to each view
        combined_view = attention_scores[0] * views[0] + attention_scores[1] * views[2] + attention_scores[2] * views[2]
        # Sum across views to get a single representation
        return combined_view


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
