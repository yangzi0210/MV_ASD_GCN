"""
Training procedures of MLP and GCN
"""
import os
import torch
import time
import pandas as pd
import numpy as np
import shutil
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from models import GPModel, MultilayerPerceptron, RecurrentNeuralNetwork, Autoencoder, RNNModel, Transformer

import torch.nn as nn


def graph_pooling(args):
    """
    Run graph pooling on the graph representation of brain imagings from raw ABIDE dataset.
    This is the first part, say unsupervised graph pooling, in our paper.
    :param args: args from the main.py
    :return: None. Pooling results are saved to args.data/ABIDE_downsample
                    with respect to different pooling ratios
    """
    torch.manual_seed(args.seed)
    # load data
    # 在PyTorch
    # Geometric中，当你加载一个图数据集如TUDataset时，如果你设置use_node_attr = True，
    # 这意味着你想要加载和使用节点属性。这些节点属性通常已经预定义在数据集中，作为节点的特征向量。
    # 当你加载数据集后，TUDataset类会自动处理这些节点属性，并将其转换为PyTorch张量。属性num_features通常不需要手动设置
    # 因为它是从数据集本身推断出来的。num_features代表每个节点的特征数量，即特征向量的维度。
    # 在你提供的代码中，虽然没有显式指定num_features的值，但当你加载ABIDE数据集后，TUDataset自动读取了节点属性，并确定了特征向量的维度是189。
    abide_dataset = TUDataset(args.data_dir, name='ABIDE', use_node_attr=True)
    args.num_classes = abide_dataset.num_classes
    args.num_features = abide_dataset.num_features

    # hierarchical graph pooling model
    gp = GPModel(args).to(args.device)

    # run graph pooling
    abide_loader = DataLoader(abide_dataset, batch_size=args.batch_size, shuffle=False)
    downsample = []
    label = []
    """
    data: x = [96681,189] edge_index=[2,5461170] y =[871] 
    downsample: list 128 * 378 -> 256 * 378 ... -> 871 * 378
    x：节点特征矩阵。在图数据中，x通常是一个二维的张量（Tensor），其大小为[num_nodes, num_node_features]
    这里num_nodes是图中节点的数量，num_node_features是每个节点特征的维度。每一行代表一个节点的特征向量。
    
    y：节点或图的标签。y可以是一个向量或一个张量，它包含每个节点或整个图的标签信息。
    在节点级别的任务中（如节点分类），y的大小通常是[num_nodes]，每个元素对应一个节点的标签。
    在图级别的任务中（如图分类），y的大小可能是[num_graphs]，每个元素对应一个图的标签。
    
    edge_index：边索引。edge_index是一个二维的张量，大小为[2, num_edges]，这里num_edges是图中边的数量。
    每一列代表一条边，其中第一行的元素是边的起始节点索引，第二行的元素是边的终止节点索引。
    例如，如果edge_index的第一列是[0, 2]，这表示节点0和节点2之间有一条边。
    edge_index用于描述图的结构，即节点间的连接关系。
    """
    for i, data in enumerate(abide_loader):
        data = data.to(args.device)
        downsample += gp(data).cpu().detach().numpy().tolist()
        label += data.y.cpu().detach().numpy().tolist()
        # print(downsample, 'downsample')
    downsample_df = pd.DataFrame(downsample)
    # store the label, in case of data samples shuffle
    downsample_df['label'] = label

    # store the selected substructures
    downsample_dir = os.path.join(args.data_dir, 'ABIDE_downsample')
    if not os.path.exists(downsample_dir):
        os.makedirs(downsample_dir)
    downsample_file = os.path.join(downsample_dir, 'ABIDE_pool_%.3f_.txt' % args.pooling_ratio)
    downsample_df.to_csv(downsample_file, index=False, header=False, sep='\t')

    del gp
    del data
    del abide_dataset
    del abide_loader
    del downsample_df
    torch.cuda.empty_cache()


def test_mlp(model, loader, args):
    """
    Test the MLP performance on loader (validation dataloader)
    :param model: an instance of MultilayerPerceptron
    :param loader: dataloader, specifically, the validation dataloader.
    :param args: args from the main.py
    :return: accuracy, loss on the validation set.
    """
    model.eval()
    correct = 0.0
    loss_test = 0.0
    for data_x, data_y in loader:
        data_x, data_y = data_x.to(args.device), data_y.to(args.device)
        out, _ = model(data_x)
        pred = (out > 0).long()
        correct += pred.eq(data_y).sum().item()
        loss_func = nn.BCEWithLogitsLoss()
        loss_test += loss_func(out, data_y.float().unsqueeze(1)).item()
    return correct / len(loader.dataset), loss_test


def train_mlp(model, train_loader, val_loader, optimizer, save_path, args):
    """
    Training progress of Multilayer Perceptron.
    As said in the paper, we have implemented a special training progress,
    i.e., we have run another 10-fold cross-validation on the given dataset,
    and select the best model by limiting the smallest number of training epochs,
    and choosing the one with highest validation accuracy. The test set is strictly hidden.
    :param model: an instance of MultilayerPerceptron
    :param train_loader: dataloader of training set
    :param val_loader: dataloader of test set
    :param optimizer: Adam, by default
    :param save_path: temporary working path of this progress
    :param args: args from main.py
    :return: best_epoch: name of the best model
    :return: min_loss, max_accuracy: loss and accuracy on the validation set of the best model
    """
    min_loss = 1e10
    max_acc = 0
    patience_cnt = 0
    val_loss_values = []
    val_acc_values = []
    best_epoch = 0

    t = time.time()
    model.train()
    for epoch in range(args.epochs):
        loss_train = 0.0
        correct = 0
        # data_x 128 * 378 data_y 128
        for i, (data_x, data_y) in enumerate(train_loader):
            optimizer.zero_grad()
            data_x, data_y = data_x.to(args.device), data_y.to(args.device)
            out, _ = model(data_x)
            loss_func = nn.BCEWithLogitsLoss()
            loss = loss_func(out, data_y.float().unsqueeze(1))  # 注意看情况加 .unsqueeze
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            pred = (out > 0).long()
            correct += pred.eq(data_y).sum().item()
        acc_train = correct / len(train_loader.dataset)
        acc_val, loss_val = test_mlp(model, val_loader, args)
        if args.verbose:
            print('\r', 'Epoch: {:04d}'.format(epoch + 1), 'loss_train: {:.6f}'.format(loss_train),
                  'acc_train: {:.6f}'.format(acc_train), 'time: {:.6f}s'.format(time.time() - t), end='', flush=True)

        val_loss_values.append(loss_val)
        val_acc_values.append(acc_val)
        # Skip logging the first args.least models
        if epoch < args.least:
            continue
        if val_loss_values[-1] <= min_loss:
            model_state = {'net': model.state_dict(), 'args': args}
            torch.save(model_state, os.path.join(save_path, '{}.pth'.format(epoch)))
            min_loss = val_loss_values[-1]
            max_acc = val_acc_values[-1]
            best_epoch = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt == args.patience:
            break

        # delete other models
        files = [f for f in os.listdir(save_path) if f.endswith('.pth')]
        for f in files:
            if f.startswith('num'):
                continue
            epoch_nb = int(f.split('.')[0])
            if epoch_nb != best_epoch:
                os.remove(os.path.join(save_path, f))
    if args.verbose:
        print('\nOptimization Finished! Total time elapsed: {:.6f}'.format(time.time() - t))

    return best_epoch, max_acc, min_loss


def extract(data, args, least_epochs=100):
    """
    Herein, we use the best MLP model to extract further learned features from the pooling results
    This is the one that connects second part(MLP) and third part(GCN or LR) in our paper.
    :param least_epochs: least number of training epochs. This is a rather important super parameter.
    :param data: Pooling results. shape: [number of subjects, dim of pooling features] = [871, 378]
    :param args: args from main.py
    :return: None. All the extracted further learned features are saved to /args.data/Further_Learned_Features/fold_%d
    """
    x = data[:, :-1]
    y = data[:, -1]
    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(x, y)

    for i in range(10):
        fold_dir = os.path.join(args.check_dir, 'MLP', 'fold_%d' % (i + 1))
        files = os.listdir(fold_dir)
        max_epoch = 0
        best_model = None

        for f in files:
            if f.endswith('.pth') and f.startswith('num_'):
                acc = float(f.split('_')[3])
                epoch_num = int(f.split('_')[-2])
                if epoch_num > max_epoch:
                    max_epoch = epoch_num
                    best_model = f

        assert best_model is not None, \
            'Cannot find the trained model. Maybe the least_epochs is too large.'

        # use the best MLP model to further extract features
        # from the downsampled brain imaging
        if args.verbose:
            print('extracting information with model {}'.format(fold_dir + '/' + best_model))

        checkpoint = torch.load(os.path.join(fold_dir, best_model))
        model_args = checkpoint['args']
        dataloader = DataLoader(dataset, batch_size=model_args.batch_size, shuffle=False)
        # change model here
        model = Transformer(model_args).to(model_args.device)
        # load model

        model.load_state_dict(checkpoint['net'], strict=False)

        model.eval()

        feature_matrix = []
        label = []
        correct = 0
        for data_x, data_y in dataloader:
            data_x, data_y = data_x.to(args.device), data_y.to(args.device)
            out, features = model(data_x)
            feature_matrix += features.cpu().detach().numpy().tolist()
            pred = (out > 0).long()
            correct += pred.eq(data_y).sum().item()
            label += data_y.cpu().detach().numpy().tolist()

        fold_feature_matrix = np.array(feature_matrix)

        features = pd.DataFrame(fold_feature_matrix)
        features['label'] = label

        # save the features to data_dir/MLP/fold_
        feature_path = os.path.join(args.data_dir, 'Further_Learned_Features', 'fold_%d' % (i + 1))
        if not os.path.exists(feature_path):
            os.makedirs(feature_path)
        features.to_csv(os.path.join(feature_path, 'features.txt'), header=False, index=False, sep='\t')

        # inherit the test indices from model path to the data path
        shutil.copyfile(os.path.join(fold_dir, 'test_indices.txt'),
                        os.path.join(feature_path, 'test_indices.txt'))

    print('Done!')
    print('Further Learned Features saved to features.txt')


def test_gcn(loader, model, args, test=True):
    """
    Test the GCN performance on loader. We have not use validation set in GCN.
    So, this is used to print the performance on test set
    :param loader: an instance of torch_geometric.data.Dataloader
    :param model: an instance of GCN
    :param args: args from main.py
    :return: accuracy, loss, predictions on test set
    """
    model.eval()
    correct = 0.0
    loss_test = 0.0
    output = []
    criterion = nn.BCEWithLogitsLoss()
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    ACC = 0.0
    PRE = 0.0
    RECALL = 0.0
    SC = 0.0
    F1_SCORE = 0.0
    for data in loader:
        data = data.to(args.device)
        out, _ = model(data.x, data.edge_index, data.edge_attr)
        output += out.cpu().detach().numpy().tolist()
        if test:
            pred = (out[data.test_mask] > 0).long()  # Predicted tensor
            length = data.test_mask.sum().item()
            correct += pred.eq(data.y[data.test_mask]).sum().item()
            loss_test += criterion(out[data.test_mask], data.y[data.test_mask].float()).item()
            # ---------------------- feat: 添加其他指标预测 ---------------------------------------
            # TP(True Positive，真阳性)：样本的真实类别是正类，并且模型预测的结果也是正类。
            # FP(False Positive，假阳性)：样本的真实类别是负类，但是模型将其预测成为正类。
            # TN(True Negative，真阴性)：样本的真实类别是负类，并且模型将其预测成为负类。
            # FN(False Negative，假阴性)：样本的真实类别是正类，但是模型将其预测成为负类。c
            # ACC = TP + TN / TP + TN + FP + FN
            # Precision: PRE = TP /TP + FP
            # RECALL = TP / TP + FN
            # Specificity: SC = TN / TN + FP
            # F1 SCORE = 2 * PRE * RECALL / PRE + RECALL
            pre_list = pred.cpu().numpy().tolist()
            true_list = data.y[data.test_mask].cpu().numpy().tolist()
            for true_label, predicted_label in zip(true_list, pre_list):
                if true_label == 0 and predicted_label == 0:
                    TP += 1  # True Positive
                elif true_label == 1 and predicted_label == 0:
                    FN += 1  # False Negative
                elif true_label == 0 and predicted_label == 1:
                    FP += 1  # False Positive
                elif true_label == 1 and predicted_label == 1:
                    TN += 1  # True Negative
            ACC = (TP + TN) / (TP + TN + FP + FN)
            PRE = TP / (TP + FP)
            RECALL = TP / (TP + FN)
            SC = TN / (TN + FP)
            F1_SCORE = 2 * PRE * RECALL / (PRE + RECALL)
            # 统一保留16位小数对齐 且转为 str 格式
            ACC = "{:.16f}".format(ACC)
            RECALL = "{:.16f}".format(RECALL)
            PRE = "{:.16f}".format(PRE)
            SC = "{:.16f}".format(SC)
            F1_SCORE = "{:.16f}".format(F1_SCORE)
            file = "index.txt"
            # 打开文件
            f = open(file, 'a', encoding='utf-8')
            idx = 'ACC: ' + ACC + ' RECALL: ' + RECALL + ' PRE: ' + PRE + ' SC: ' + SC + ' F1_SCORE: ' + F1_SCORE + '\n'
            f.write(idx)
            # -------------------------------------------------------------------------------------------
        else:
            pred = (out[data.val_mask] > 0).long()
            length = data.val_mask.sum().item()
            correct += pred.eq(data.y[data.val_mask]).sum().item()
            loss_test += criterion(out[data.val_mask], data.y[data.val_mask].float()).item()
            return correct / length, loss_test, output
    return correct / length, loss_test, output, ACC, RECALL, PRE, SC, F1_SCORE


def train_gcn(dataloader, model, optimizer, save_path, args):
    """
    Training phase of GCN. No validation set is used here.
    :param save_path: working path for this progress
    :param dataloader: dataloader of training set
    :param model: an instance of GCN
    :param optimizer: Adam, by default
    :param args: args from main.py
    :return: filename of the best model
    """
    min_loss = 1e10
    patience_cnt = 0
    loss_set = []
    acc_set = []
    best_epoch = 0
    num_epoch = 0

    t = time.time()
    model.train()
    for epoch in range(args.epochs):
        loss_train = 0.0
        correct = 0
        num_epoch += 1
        for i, data in enumerate(dataloader):
            optimizer.zero_grad()
            data = data.to(args.device)
            out, _ = model(data.x, data.edge_index, data.edge_attr)
            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(out[data.train_mask], data.y[data.train_mask].float())
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            pred = (out[data.train_mask] > 0).long()
            correct += pred.eq(data.y[data.train_mask]).sum().item()

        acc_train = correct / data.train_mask.sum().item()
        acc_val, loss_val, _ = test_gcn(dataloader, model, args, test=False)
        if args.verbose:
            print('\r', 'Epoch: {:06d}'.format(epoch + 1), 'loss_train: {:.6f}'.format(loss_train),
                  'acc_train: {:.6f}'.format(acc_train), 'loss_val: {:.6f}'.format(loss_val),
                  'acc_val: {:.6f}'.format(acc_val), 'time: {:.6f}s'.format(time.time() - t), flush=True, end='')

        loss_set.append(loss_val)
        acc_set.append(acc_val)
        if epoch < args.least:
            continue
        if loss_set[-1] < min_loss:
            model_state = {'net': model.state_dict(), 'args': args}
            torch.save(model_state, os.path.join(save_path, '{}.pth'.format(epoch)))
            min_loss = loss_set[-1]
            best_epoch = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt == args.patience:
            break

        files = [f for f in os.listdir(save_path) if f.endswith('.pth')]
        for f in files:
            if f.startswith('fold'):
                continue
            epoch_nb = int(f.split('.')[0])
            if epoch_nb != best_epoch:
                os.remove(os.path.join(save_path, f))

    if args.verbose:
        print('\nOptimization Finished! Total time elapsed: {:.6f}'.format(time.time() - t))

    return best_epoch
