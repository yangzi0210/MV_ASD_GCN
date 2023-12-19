import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from layers import HGPSLPool
from torch_geometric.nn import GCNConv, APPNP, ClusterGCNConv, ChebConv, GINConv, GATConv, GATv2Conv


# -------------  TODO:  正确合理修改 MLP 结构 ----------------------------------
# RNN Transformer
# 线性回归:
# 适用于线性关系的问题，尤其是当输入特征与目标变量之间存在线性关系时。
# 决策树回归:
# 通过构建树状结构来对输入空间进行划分，每个叶子节点对应一个目标值。
# 随机森林回归:
# 由多个决策树组成的集成模型，通过综合多个模型的预测来提高性能和鲁棒性。
# 支持向量机回归:
# 通过找到将特征映射到高维空间后能够分隔目标变量的超平面。
# K近邻回归:
# 通过找到与给定实例最近的 k 个实例，并取其目标变量的平均值来进行预测。
# 神经网络 (深度学习) 模型:
# 除了上面示例的 MLP，还有其他深度学习架构，如卷积神经网络 (CNN) 和循环神经网络 (RNN)，可以用于处理回归问题。
# 梯度提升机 (Gradient Boosting):
# 通过迭代地训练弱模型，并根据前一个模型的误差来调整参数，从而逐步提高模型性能。

# Model of hierarchical graph pooling
class GPModel(torch.nn.Module):
    def __init__(self, args):
        super(GPModel, self).__init__()
        # parameters of hierarchical graph pooling
        self.args = args
        self.num_features = args.num_features
        self.pooling_ratio = args.pooling_ratio
        self.sample = True
        self.sparse = True
        self.sl = False
        self.lamb = 1.0

        # define the pooling layers
        self.pool1 = HGPSLPool(self.num_features, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)
        self.pool2 = HGPSLPool(self.num_features, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)
        self.pool3 = HGPSLPool(self.num_features, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # initialize edge weights
        edge_attr = None

        # hierarchical pooling
        x, edge_index, edge_attr, batch = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x, edge_index, edge_attr, batch = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x, edge_index, edge_attr, batch = self.pool3(x, edge_index, edge_attr, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # Fuse the above three pooling results
        x = F.relu(x1) + F.relu(x2) + F.relu(x3)

        # return the selected substructures
        return x


# Multilayer Perceptron
class MultilayerPerceptron(torch.nn.Module):
    def __init__(self, args):
        super(MultilayerPerceptron, self).__init__()
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.dropout_ratio = args.dropout_ratio

        self.lin1 = torch.nn.Linear(self.num_features, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, 1)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)

        # further learned features
        features = x
        # for training phase
        x = torch.flatten(self.lin3(x))
        return x, features


## TODO Transformer搭建
class TransformerModel(nn.Module):
    def __init__(self, args, num_classes=2):
        super(TransformerModel, self).__init__()
        args.num_layers = 1
        self.embedding_dim = args.nhid
        self.nhead = args.nhid // 2
        self.dim_feedforward = args.nhid * 2
        self.embedding = nn.Embedding(args.num_features, self.embedding_dim)

        # 使用 nn.Transformer 定义 Transformer 模型
        self.transformer = nn.Transformer(
            d_model=self.embedding_dim,
            nhead=self.nhead,
            num_encoder_layers=args.num_layers,
            num_decoder_layers=args.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=args.dropout_ratio
        )

        # 全连接层，用于输出二分类结果
        self.fc = nn.Linear(self.embedding_dim, num_classes)

    def forward(self, x):
        # 输入x的维度为 (batch_size, num_features)
        x = x.long()
        # 通过嵌入层
        vocab_size = max(378, torch.max(x).to("cuda").item() + 1)
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        x = self.embedding(x)

        # Transformer模型的输入要求为 (sequence_length, batch_size, embedding_dim)
        x = x.permute(1, 0, 2)

        # Transformer模型的前向传播
        x = self.transformer(x, x)
        # feature
        feature = x
        # 取Transformer模型的最后一个位置的输出
        x = x[-1, :, :]

        # 全连接层
        x = self.fc(x)

        # 二分类任务通常使用 sigmoid 激活函数输出概率
        x = torch.sigmoid(x)

        return x, feature


# CNN
class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, args):
        super(ConvolutionalNeuralNetwork, self).__init__()

        # 输入特征数，与MLP中的num_features相对应
        self.num_features = args.num_features

        # 第一个卷积层
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)

        # 第二个卷积层
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)

        # 池化层
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # 计算全连接层的输入特征数
        self.fc_input_size = 64 * (self.num_features // 4)

        # 第一个全连接层
        self.fc1 = nn.Linear(self.fc_input_size, args.nhid)

        # 第二个全连接层
        self.fc2 = nn.Linear(args.nhid, args.nhid // 2)

        # 第三个全连接层，输出层
        self.fc3 = nn.Linear(args.nhid // 2, 1)

        # Dropout层
        self.dropout = nn.Dropout(p=args.dropout_ratio)

    def forward(self, x):
        x = x.unsqueeze(1)

        # 卷积层1 + 激活函数 + 池化层
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        # 卷积层2 + 激活函数 + 池化层
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # 将特征展平
        x = x.view(x.size(0), -1)  # 保留batch_size，将其余维度展平

        # 全连接层1 + 激活函数
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # 全连接层2 + 激活函数
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        features = x
        # 输出层
        x = self.fc3(x)

        return x, features


# RNN
class RecurrentNeuralNetwork(nn.Module):
    def __init__(self, args):
        super(RecurrentNeuralNetwork, self).__init__()
        args.num_layers = 1  # TODO 调参
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.dropout_ratio = args.dropout_ratio
        self.num_layers = args.num_layers  # You can adjust the number of layers based on your requirements

        # Define an RNN layer (using LSTM as an example, but you can choose other RNN architectures)
        self.rnn = nn.LSTM(input_size=self.num_features,
                           hidden_size=self.nhid,
                           num_layers=self.num_layers,
                           batch_first=True,
                           dropout=self.dropout_ratio if self.dropout_ratio > 0 else 0)

        self.fc1 = nn.Linear(self.nhid, self.nhid // 2)
        self.fc2 = nn.Linear(self.nhid // 2, 1)

    def forward(self, x):
        # Remove the sequence dimension if it's 1
        if x.size(1) == 1:
            x = x.squeeze(1)
        # fix: RuntimeError: cudnn RNN backward can only be called in training mode
        self.train()
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.nhid).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.nhid).to(x.device)

        # Forward pass through the RNN layer
        out, _ = self.rnn(x.unsqueeze(1), (h0, c0))

        # print(feature.shape)
        # Take the output from the last time step
        out = out[:, -1, :]

        # Apply the final fully connected layer
        out = self.fc1(out)
        out = F.relu(out)
        feature = out
        out = self.fc2(out)

        return out, feature


# Model of graph convolutional Networks run on population graph
class GCN(torch.nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.dropout_ratio = args.dropout_ratio
        # define the gcn layers. As stated in the paper,
        # herein, we have employed GCNConv and ClusterGCN
        # feat： 修改GCN模型结构
        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = ChebConv(self.nhid, self.nhid // 2, 6)
        self.conv3 = ClusterGCNConv(self.nhid // 2, 1)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)

        x = self.conv2(x, edge_index, edge_weight)
        x = x.relu()
        # store the learned node embeddings
        features = x
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)

        x = self.conv3(x, edge_index)

        x = torch.flatten(x)
        return x, features
