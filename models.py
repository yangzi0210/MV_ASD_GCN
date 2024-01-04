import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer
from torch_geometric.nn import global_mean_pool, global_max_pool,global_add_pool,TopKPooling,SAGPooling,EdgePooling,ASAPooling,PANPooling,MemPooling
from layers import HGPSLPool
from torch_geometric.nn import GCNConv, APPNP, ClusterGCNConv, ChebConv, GraphSAGE


# -------------  TODO:  正确合理修改 MLP 结构 ----------------------------------
# 如果你的输入特征是一维的378长度的向量，那么除了多层感知机（MLP）之外，还有多种不同的方法可以用来提取特征。以下是一些常见的方法：
#
# 卷积神经网络（CNN）:
# 即使CNN通常用于处理图像数据，你也可以将一维向量重新塑形成一维序列或虚构的二维数据（比如一个宽度为378，高度为1的图像）
# 然后使用一维或二维卷积来提取局部特征。
#
# 递归神经网络（RNN）:
# 对于序列数据，RNN及其变体（如LSTM或GRU）可以有效地处理输入特征。你可以将378维向量视为序列，并通过RNN来提取时间或顺序相关的特征。
#
# 自编码器（Autoencoders）:
# 自编码器可以通过无监督学习来学习输入数据的有效表示。一个典型的自编码器包括一个编码器（将输入压缩成一个低维表示）和一个解码器（从低维表示重构输入）。
#
# 变分自编码器（Variational Autoencoders，VAEs）:
# VAE是自编码器的一种，它不仅学习数据的压缩表示，还学习数据的概率分布。这可以用于生成新的数据样本，或者作为特征提取器。
#
# 变换器（Transformers）:
# 虽然变换器模型通常用于处理文本数据，但它们的自注意力机制可以应用于任何类型的序列数据。你可以将378维向量视为序列，并使用变换器提取全局依赖关系。
#
# 残差网络(ResNet)


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
        # x: 14208 * 189 = 128 * 111 * 189
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # initialize edge weights
        edge_attr = None

        # hierarchical pooling
        # 0.05 时 x: 768 * 189 = num_nodes * pooling_ration * 189
        x, edge_index, edge_attr, batch = self.pool1(x, edge_index, edge_attr, batch)
        # x1 Tensor 128 * 378
        # 按照维数 1 （列）拼接
        # gmp gap: Tensor 128(batch) * 189
        # 图池化
        x1 = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        # gmp_x = gmp(x, batch)
        # gap_x = gap(x, batch)

        x, edge_index, edge_attr, batch = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)

        x, edge_index, edge_attr, batch = self.pool3(x, edge_index, edge_attr, batch)
        x3 = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        # Fuse the above three pooling results: x Tensor 128 * 378 = batch * 189(时间序列) * 2
        x = F.relu(x1) + F.relu(x2) + F.relu(x3)

        # return the selected substructures
        return x


# Multilayer Perceptron
class MultilayerPerceptron(torch.nn.Module):
    """
    这个函数接收一个参数args，它包含模型配置的属性。
    self.num_features是输入特征的维度。
    self.nhid是隐藏层的神经元数量。
    self.dropout_ratio是dropout操作的概率，用于防止过拟合。
    创建三个全连接层（self.lin1，self.lin2，和self.lin3），这些层通过线性变换将数据从一个空间映射到另一个空间。
    第一层self.lin1将输入特征映射到self.nhid个神经元。
    第二层self.lin2将第一隐藏层的输出再映射到self.nhid // 2 个神经元。
    第三层self.lin3将第二隐藏层的输出映射到一个单一的输出节点，用于二分类。
    """

    def __init__(self, args):
        super(MultilayerPerceptron, self).__init__()
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.dropout_ratio = args.dropout_ratio

        self.lin1 = torch.nn.Linear(self.num_features, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, 1)

    def forward(self, x):
        # x: 128 * 378
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        # 128 * 256
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)

        # further learned features
        # 128 * 128
        features = x
        # for training phase
        # x: 128 * 128 -> 128
        x = torch.flatten(self.lin3(x))
        return x, features


# VAE
# class VariationalAutoencoder(nn.Module):
#     def __init__(self, args):
#         super(VariationalAutoencoder, self).__init__()
#         self.num_features = args.num_features
#         self.nhid = args.nhid
#         self.dropout_ratio = args.dropout_ratio
#
#         # Encoder
#         self.fc1 = nn.Linear(self.num_features, self.nhid)
#         self.fc2_mean = nn.Linear(self.nhid, self.nhid // 2)
#         self.fc2_logvar = nn.Linear(self.nhid, self.nhid // 2)
#
#         # Decoder
#         self.fc3 = nn.Linear(self.nhid // 2, self.nhid)
#         self.fc4 = nn.Linear(self.nhid, self.num_features)
#
#         # Feature layer
#         self.feature_layer = nn.Linear(self.nhid // 2, self.nhid // 2)
#
#     def encode(self, x):
#         h1 = F.relu(self.fc1(x))
#         h1 = F.dropout(h1, p=self.dropout_ratio, training=self.training)
#         return self.fc2_mean(h1), self.fc2_logvar(h1)
#
#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std
#
#     def decode(self, z):
#         h3 = F.relu(self.fc3(z))
#         h3 = F.dropout(h3, p=self.dropout_ratio, training=self.training)
#         return torch.sigmoid(self.fc4(h3))
#
#     def forward(self, x):
#         mu, logvar = self.encode(x.view(-1, self.num_features))
#         z = self.reparameterize(mu, logvar)
#         features = self.feature_layer(z)  # Extracted features
#         reconstructed_x = self.decode(z)
#         x = torch.mean(reconstructed_x, dim=1)
#         return x, features
#
#
# # Define loss function for VAE
# def loss_function(recon_x, x, mu, logvar):
#     BCE = F.binary_cross_entropy(recon_x, x.view(-1, recon_x.size(1)), reduction='sum')
#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     return BCE + KLD


# Auto Encoder

# AE
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
        feature = x
        encoded = F.relu(self.encoder_lin3(x))

        # 解码过程
        x = F.relu(self.decoder_lin1(encoded))
        x = F.relu(self.decoder_lin2(x))
        reconstructed = self.decoder_lin3(x)

        # return x 应该是一个展平的 128 一维向量 for train
        reconstructed = torch.mean(reconstructed, dim=1)
        # reconstructed 128 * 378
        # encoded 128 * 64
        return reconstructed, feature


# ResNet
class ResidualBlock(nn.Module):
    def __init__(self, num_features, nhid):
        super(ResidualBlock, self).__init__()
        self.lin1 = nn.Linear(num_features, nhid)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.lin2 = nn.Linear(nhid, num_features)
        self.bn2 = nn.BatchNorm1d(num_features)

    def forward(self, x):
        identity = x
        out = self.lin1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.lin2(out)
        out = self.bn2(out)
        out += identity  # Add the input x to the output
        out = F.relu(out)  # Apply activation function
        return out


class ResNet(nn.Module):
    def __init__(self, args):
        super(ResNet, self).__init__()
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.dropout_ratio = args.dropout_ratio

        # Define the ResNet layers
        self.lin1 = nn.Linear(self.num_features, self.nhid)
        self.bn1 = nn.BatchNorm1d(self.nhid)
        self.res_block1 = ResidualBlock(self.nhid, self.nhid // 2)
        self.res_block2 = ResidualBlock(self.nhid, self.nhid // 2)
        self.lin2 = nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = nn.Linear(self.nhid // 2, 1)

    def forward(self, x):
        x = self.lin1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        # print(x.shape)
        # further learned features
        features = x
        # for training phase
        x = torch.flatten(self.lin3(x))
        return x, features


# Transformer
class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.dropout_ratio = args.dropout_ratio
        self.num_layers = 3
        self.embedding = nn.Linear(args.num_features,
                                   args.nhid)  # Assuming input_size is the same as embedding size for simplicity
        encoder_layer = nn.TransformerEncoderLayer(d_model=args.nhid, nhead=2, dropout=args.dropout_ratio)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.num_layers)
        self.fc1 = nn.Linear(args.nhid, args.nhid // 2)
        self.fc2 = nn.Linear(args.nhid // 2, 1)

    def forward(self, x):
        # 128 * 378
        # x: batch_size * seq_length * input_size
        # Assuming x is already batched and has shape (batch_size, seq_length, input_size)
        x = self.embedding(x)  # Embed the input
        # x = x.permute(1, 0, 2)  # Transformer expects seq_length, batch_size, input_size
        out = self.transformer_encoder(x)
        # out = out.permute(1, 0, 2)  # Convert back to batch_size, seq_length, input_size
        # out = F.relu(out[:, -1, :])  # Take the output of the last token
        out = self.fc1(out)
        out = F.relu(out)
        out = F.dropout(out, self.dropout_ratio, self.training)
        features = out
        out = self.fc2(out)
        return out, features


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
        args.num_layers = 3
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


class RNNModel(nn.Module):
    def __init__(self, args):
        super(RNNModel, self).__init__()
        args.num_layers = 3
        self.dropout_ratio = args.dropout_ratio
        self.rnn = nn.RNN(args.num_features, args.nhid, args.num_layers, batch_first=True, dropout=args.dropout_ratio)
        self.fc1 = nn.Linear(args.nhid, args.nhid // 2)
        self.fc2 = nn.Linear(args.nhid // 2, 1)

    def forward(self, x):
        # x: batch_size * input_size
        # fix: RuntimeError: cudnn RNN backward can only be called in training mode
        self.train()
        x, _ = self.rnn(x)
        # x: 128 * 378
        # x = F.relu(x[-1,:])  # Take the output of the last time step
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        features = x
        x = torch.flatten(self.fc2(x))
        return x, features


# Model of graph convolutional Networks run on population graph

class GCN(torch.nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.num_features = args.num_features  # 128
        self.nhid = args.nhid # 64
        self.dropout_ratio = args.dropout_ratio
        # define the gcn layers. As stated in the paper,
        # herein, we have employed GCNConv and ClusterGCN
        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = ClusterGCNConv(self.nhid, 1)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        features = x
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.conv2(x, edge_index)
        # store the learned node embeddings
        x = torch.flatten(x)
        return x, features


class ChebGCN(torch.nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.dropout_ratio = args.dropout_ratio
        # feat： 修改模型结构
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


class GraphSAGEGCN(torch.nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.dropout_ratio = args.dropout_ratio
        # feat： 修改模型结构
        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GraphSAGE(self.nhid, self.nhid * 2, 2, self.nhid // 2)
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
