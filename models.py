import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_mean_pool, global_max_pool
from layers import HGPSLPool
from torch_geometric.nn import GCNConv, APPNP, ClusterGCNConv, ChebConv, GraphSAGE, CuGraphSAGEConv, GATv2Conv,GINConv,GATConv,AGNNConv


# Unsupervised Model of hierarchical graph pooling
class GPModel(torch.nn.Module):
    def __init__(self, args):
        super(GPModel, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.num_classes = 1
        self.pooling_ratio = args.pooling_ratio
        self.sample = True
        self.sparse = True
        self.sl = False
        self.lamb = 1.0
        self.nhid = args.nhid
        # 原本方法替换GCN 还是原有的改 不要线性层
        self.pool1 = HGPSLPool(self.num_features, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)

        # 修改 GCN 层输出特征维度
        self.conv1 = GCNConv(self.num_features, self.nhid)  # 假设 num_features 是输入特征的维度

        self.conv2 = AGNNConv(self.nhid, self.num_features)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = None
        # x 14208 * 189
        # 应用第一个 pooling 层
        x, edge_index, edge_attr, batch = self.pool1(x, edge_index, edge_attr, batch)  # 768 * 189
        # 768 * 189
        # 应用 GCN 层
        x = self.conv1(x, edge_index)  # 768 * 378 # 618 * 378

        # 应用非线性激活函数
        x = F.relu(x)

        x = self.conv2(x, edge_index)  # 768 * 378

        x = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        # x = global_mean_pool(x, batch)
        # 应用线性层
        x = F.relu(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, args):
        super(Autoencoder, self).__init__()
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.dropout_ratio = args.dropout_ratio

        # 编码器
        self.encoder_lin1 = nn.Linear(self.num_features, self.nhid)
        self.encoder_lin2 = nn.Linear(self.nhid, self.nhid // 2)
        self.encoder_lin3 = nn.Linear(self.nhid // 2, self.nhid // 4)  # 假设编码到更小的维度

        # 解码器
        self.decoder_lin1 = nn.Linear(self.nhid // 4, self.nhid // 2)
        self.decoder_lin2 = nn.Linear(self.nhid // 2, self.nhid)
        self.decoder_lin3 = nn.Linear(self.nhid, self.num_features)

    def forward(self, x):
        # 编码过程
        x = F.relu(self.encoder_lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.encoder_lin2(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        features = x
        encoded = F.relu(self.encoder_lin3(x))

        # 解码过程
        x = F.relu(self.decoder_lin1(encoded))
        x = F.relu(self.decoder_lin2(x))
        reconstructed = self.decoder_lin3(x)

        # return x 应该是一个展平的 128 一维向量
        reconstructed = torch.mean(reconstructed, dim=1)
        # reconstructed 128 * 378
        # encoded 128 * 64
        return reconstructed, features


class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, args):
        super(ConvolutionalNeuralNetwork, self).__init__()

        # 输入特征数，与MLP中的num_features相对应
        self.num_features = args.num_features

        # 第一个卷积层
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)

        # 第二个卷积层
        # self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 128, kernel_size=3, stride=1, padding=1)
        # 池化层
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # 计算全连接层的输入特征数
        self.fc_input_size = 128 * (self.num_features // 4)

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


class CNNRegression(nn.Module):
    def __init__(self, args):
        super(CNNRegression, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels=args.num_features, out_channels=16, kernel_size=3, stride=1, padding=1)
        # 第一个池化层
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第二个卷积层
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # 第二个池化层
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层
        self.fc1 = nn.Linear(32 * 8 * 8, 100)
        self.fc2 = nn.Linear(100, 50)
        # 回归预测层
        self.predict = nn.Linear(50, 1)

    def forward(self, x):
        # x = x.view(-1, 1, 1, 378)
        x = F.relu(self.conv1(x.unsqueeze(1)))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 32 * 8 * 8)
        # Adjust based on the pool and conv layers
        # further learned features
        features = x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.predict(x)
        return torch.flatten(output), features


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


# Model of graph convolutional Networks run on population graph
class GCN(torch.nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.dropout_ratio = args.dropout_ratio
        # define the gcn layers. As stated in the paper,
        # herein, we have employed GCNConv and ClusterGCN
        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = ClusterGCNConv(self.nhid, 1)
        # self.conv2 = GATv2Conv(self.nhid, 1)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        # store the learned node embeddings
        features = x
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.conv2(x, edge_index)
        x = torch.flatten(x)
        return x, features
