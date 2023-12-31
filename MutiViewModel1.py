import torch
import torch.nn as nn
import torch.nn.functional as F


# 使用某种融合策略将不同图的节点表示整合起来。这可以通过各种方法完成，例如：
# 直接平均每个视图的特征。
# 加权平均，根据每个视图的预测性能来分配权重。
# 拼接（concatenate）每个视图的特征，然后通过一个或多个全连接层进行处理。
# 使用注意力机制来动态地分配不同视图特征的权重。

class MultiViewAttentionFusion(nn.Module):
    def __init__(self, num_views, feature_dim):
        super(MultiViewAttentionFusion, self).__init__()
        self.num_views = num_views
        self.attention_weights = nn.Parameter(torch.Tensor(1, num_views))
        self.feature_dim = feature_dim
        self.reset_parameters()

    def reset_parameters(self):
        # 初始化注意力权重
        nn.init.constant_(self.attention_weights, 1.0 / self.num_views)

    def forward(self, view_features):
        # view_features: 列表，包含每个视图的特征，每个元素的形状为 (batch_size, feature_dim)
        assert len(view_features) == self.num_views

        # 将特征堆叠起来以便广播，形状为 (batch_size, num_views, feature_dim)
        stacked_features = torch.stack(view_features, dim=1)

        # 计算注意力分数，使用softmax进行归一化
        attention_scores = F.softmax(self.attention_weights, dim=-1)

        # 应用注意力分数（广播机制）
        weighted_features = stacked_features * attention_scores.unsqueeze(-1)

        # 沿着视图的维度求和来融合特征
        fused_features = torch.sum(weighted_features, dim=1)

        return fused_features


# 假设的特征维度和视图数量
feature_dim = 128
num_views = 3

# 创建多视图注意力融合模块
attention_fusion = MultiViewAttentionFusion(num_views=num_views, feature_dim=feature_dim)

# 假设的每个视图的特征
view1_features = torch.rand(10, feature_dim)  # 10个样本，每个样本128维特征
view2_features = torch.rand(10, feature_dim)
view3_features = torch.rand(10, feature_dim)

# 融合特征
fused_features = attention_fusion([view1_features, view2_features, view3_features])

print(fused_features.shape)  # 应该打印出 (10, 128)，表示融合后的特征


# 在这个简单的例子中，MultiViewAttentionFusion类是一个PyTorch模块，它为每个视图学习一个注意力权重，并使用这些权重来融合不同视图的特征。
#
# 注意力权重通过softmax函数进行归一化，确保它们的和为1。然后，使用广播机制将每个视图的特征乘以对应的注意力权重，并对结果进行求和以获得最终的融合特征。
#
# 请注意，实际应用中可能需要更复杂的注意力机制，比如依赖于输入特征的动态注意力权重，或者使用多头注意力等。您可能需要根据具体任务和数据集调整和优化这个基本框架。

# ---------------------------------------------------------------------------------------------------------------------------------------

# 如果您有三个全连接图，每个图有871个节点，每个节点有一个378维的特征向量，且每个图的边权重不同，
#
# 您可以使用图注意力网络（Graph Attention Network, GAT）的变体来处理这个问题。

# 由于您提到图是全连接的，这意味着每个节点都与其他所有节点相连，但由于边权重不同，这将为每个视图提供独特的信息。

# 以下是一个使用PyTorch实现的简单例子，它将利用注意力机制来融合这三个视图的信息：


# 定义一个图注意力层
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# 定义一个多视图融合模块
class MultiViewGAT(nn.Module):
    def __init__(self, num_views, in_features, out_features, dropout, alpha, nheads):
        """ nheads 表示每个视图的注意力头数量 """
        super(MultiViewGAT, self).__init__()

        # 注意力层列表
        self.attention_layers = nn.ModuleList()
        for _ in range(num_views):
            attention_heads = [GraphAttentionLayer(in_features, out_features, dropout=dropout, alpha=alpha, concat=True)
                               for _ in range(nheads)]
            self.attention_layers.append(nn.ModuleList(attention_heads))

        # 输出层
        self.out_att = GraphAttentionLayer(out_features * nheads, out_features, dropout=dropout, alpha=alpha,
                                           concat=False)

    def forward(self, x, adjs):
        """ x 是节点特征，adjs 是三个视图的邻接矩阵的列表 """
        x = F.dropout(x, 0.6, training=self.training)
        view_features = []
        for i, layers in enumerate(self.attention_layers):
            x_head = [att_layer(x, adjs[i]) for att_layer in layers]
            x_head = torch.cat(x_head, dim=1)
            view_features.append(x_head)

        x = torch.mean(torch.stack(view_features), dim=0)
        x = F.dropout(x, 0.6, training=self.training)
        x = self.out_att(x, torch.mean(torch.stack(adjs), dim=0))

        return F.log_softmax(x, dim=1)


# 参数设置
num_nodes = 871
num_features = 378
num_classes = 10  # 假设有10个类别
num_heads = 8  # 多头注意力的头数
num_views = 3  # 视图数量

# 创建模型
model = MultiViewGAT(num_views=num_views, in_features=num_features, out_features=8, dropout=0.6, alpha=0.2,
                     nheads=num_heads)

# 创建节点特征矩阵
node_features = torch.rand(num_nodes, num_features)

# 创建三个视图的全连接邻接矩阵，权重随机初始化
adjacency_matrices = [torch.rand(num_nodes, num_nodes) for _ in range(num_views)]

# 将邻接矩阵转换为概率分布，例如使用softmax
# 注意：这里假设边权重是正的，如果有负权重，可能需要其他方法处理
adjacency_matrices = [F.softmax(adj, dim=1) for adj in adjacency_matrices]

# 模拟标签数据，假设分类任务
labels = torch.randint(0, num_classes, (num_nodes,))

# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 将模型和数据发送到设备
model = model.to(device)
node_features = node_features.to(device)
adjacency_matrices = [adj.to(device) for adj in adjacency_matrices]
labels = labels.to(device)

# 设置损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

# 将模型设置为训练模式
model.train()

# 前向传播
optimizer.zero_grad()
output = model(node_features, adjacency_matrices)
loss = criterion(output, labels)
loss.backward()
optimizer.step()

print(f"Loss: {loss.item()}")

# 在这个例子中，我们首先创建了一个表示节点特征的矩阵node_features，以及表示三个不同视图的全连接邻接矩阵adjacency_matrices。由于是全连接图，
#
# 邻接矩阵的每个元素都是一个权重值，我们将其通过softmax函数转换为概率分布，这样每个节点的连接强度都可以通过概率来表示。
#
# 接下来，我们初始化了一个多视图GAT模型model，该模型包含一个图注意力层列表，用于每个视图处理节点特征，并且有一个输出层来整合多视图的信息。
#
# 然后，我们模拟了一些标签数据labels，设置了损失函数和优化器，并进行了一个训练步骤，包括前向传播、计算损失、反向传播和优化步骤。
#
# 请注意，这只是一个简化的例子，您可能需要根据您的具体任务需求来调整网络结构、超参数和训练过程。特别是对于全连接图，您可能需要考虑邻接矩阵稀疏化或选择更合适的权重处理方法，以确保模型的计算效率和性能。
