import numpy as np
from sklearn.model_selection import KFold
from torch import nn
from torch_geometric.data import DataLoader, Data
from multiview_model import MultiViewGNN
from util import formatOutput, meanOfArr, str2float
from torch_geometric.data import Data
import time
import torch
import os
import pandas as pd
from datetime import datetime


def kfold_multiview_gcn(edge_age_index, edge_age_attr, edge_sex_index, edge_sex_attr, edge_site_index, edge_site_attr,
                        num_samples, args):
    """
    Training phase of GCN. Some parameters of args are locally set here.
    No validation is implemented in this section.
    :param num_samples:
    :param args:
    :return:
    """
    # ------- feat：修改模型 添加其他指标 F1-score Accuracy Precision Recall/Sensitivity 在其他指标上效果更好 ---------
    # locally set parameters
    args.num_features = args.nhid // 2  # output feature size of MLP 整数除法
    args.nhid = args.num_features // 2
    args.epochs = 100000  # maximum number of training epochs
    args.patience = 20000  # patience for early stop regarding the performance on val set
    args.weight_decay = 0.001
    args.least = 0  # least number of training epochs

    # load population graph
    # edge_index = torch.tensor(edge_index, dtype=torch.long)
    # edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    indices = np.arange(num_samples)
    kf = KFold(n_splits=10, shuffle=True, random_state=args.seed)

    # store the predictions
    result_df = pd.DataFrame([])
    test_result_acc = []
    test_result_loss = []
    # ACC, RECALL, PRE, SC, F1_SCORE
    result_acc = []
    result_recall = []
    result_pre = []
    result_sc = []
    result_f1_score = []
    for i, (train_idx, test_idx) in enumerate(kf.split(indices)):
        # Ready to read further learned features extracted by MLP on different folds
        fold_path = os.path.join(args.data_dir, 'Further_Learned_Features', 'fold_%d' % (i + 1))
        # working path of training gcn
        work_path = os.path.join(args.check_dir, 'GCN')

        np.random.shuffle(train_idx)
        # random assign val and test sets. No nested search.
        val_idx = train_idx[:len(train_idx) // 10]
        train_idx = train_idx[len(train_idx) // 10:]

        # Make sure the three datasets are independent
        assert len(set(list(train_idx) + list(test_idx) + list(val_idx))) == num_samples, \
            'Something wrong in the CV'

        if not os.path.exists(work_path):
            os.makedirs(work_path)

        print('Training Multiview GCN on the %d fold' % (i + 1))
        model = MultiViewGNN(args).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # Load 'further learned features'
        feature_path = os.path.join(fold_path, 'features.txt')
        assert os.path.exists(feature_path), \
            'No further learned features found!'
        content = pd.read_csv(feature_path, header=None, sep='\t')

        x = content.iloc[:, :-1].values
        y = content.iloc[:, -1].values

        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        # 创建视图1的图数据对象
        data_age = Data(x=x, edge_index=edge_age_index, edge_attr=edge_age_attr, y=y)

        # 创建视图2的图数据对象
        data_sex = Data(x=x, edge_index=edge_sex_index, edge_attr=edge_sex_attr, y=y)

        # 创建视图3的图数据对象
        data_site = Data(x=x, edge_index=edge_site_index, edge_attr=edge_site_attr, y=y)

        # form the mask from idx
        train_mask = np.zeros(num_samples)
        test_mask = np.zeros(num_samples)
        val_mask = np.zeros(num_samples)
        train_mask[train_idx] = 1
        test_mask[test_idx] = 1
        val_mask[val_idx] = 1

        # set the mask for dataset
        data_age.train_mask = torch.tensor(train_mask, dtype=torch.bool)
        data_age.test_mask = torch.tensor(test_mask, dtype=torch.bool)
        data_age.val_mask = torch.tensor(val_mask, dtype=torch.bool)
        data_sex.train_mask = torch.tensor(train_mask, dtype=torch.bool)
        data_sex.test_mask = torch.tensor(test_mask, dtype=torch.bool)
        data_sex.val_mask = torch.tensor(val_mask, dtype=torch.bool)
        data_site.train_mask = torch.tensor(train_mask, dtype=torch.bool)
        data_site.test_mask = torch.tensor(test_mask, dtype=torch.bool)
        data_site.val_mask = torch.tensor(val_mask, dtype=torch.bool)

        # assure the masks has no overlaps!
        # Necessary in experiments
        assert np.array_equal(train_mask + val_mask + test_mask, np.ones_like(train_mask)), \
            'Something wrong with the cross-validation!'

        # Batch-size is meaningless

        loader_age = DataLoader([data_age], batch_size=1)
        loader_sex = DataLoader([data_sex], batch_size=1)
        loader_site = DataLoader([data_site], batch_size=1)
        # dataloader
        # # 假设您有三个视图的数据，这里是它们的模拟初始化
        # adjacency_matrices = [edge_age_index, edge_sex_index, edge_site_index]
        # edge_weights = [edge_age_attr, edge_sex_attr, edge_site_attr]
        # features = x
        # labels = y

        # Model training
        best_model = train_multiview_gcn(loader_age, loader_sex, loader_site, model, optimizer, work_path, args)
        # Restore best model for test set
        checkpoint = torch.load(os.path.join(work_path, '{}.pth'.format(best_model)))
        model.load_state_dict(checkpoint['net'])
        test_acc, test_loss, test_out, ACC, RECALL, PRE, SC, F1_SCORE = test_multiview_gcn(loader_age, loader_sex,
                                                                                           loader_site, model, args)

        # Store the resluts
        result_df['fold_%d_' % (i + 1)] = test_out
        test_result_acc.append(test_acc)
        test_result_loss.append(test_loss)
        result_acc.append(ACC)
        result_recall.append(RECALL)
        result_pre.append(PRE)
        result_sc.append(SC)
        result_f1_score.append(F1_SCORE)

        acc_val, loss_val, _ = test_multiview_gcn(loader_age, loader_sex, loader_site, model, args, test=False)
        print('GCN {:0>2d} fold test set results, loss = {:.6f}, accuracy = {:.6f}'.format(i + 1, test_loss, test_acc))
        print('GCN {:0>2d} fold val set results, loss = {:.6f}, accuracy = {:.6f}'.format(i + 1, loss_val, acc_val))

        state = {'net': model.state_dict(), 'args': args}
        torch.save(state, os.path.join(work_path, 'fold_{:d}_test_{:.6f}_drop_{:.3f}_epoch_{:d}_.pth'
                                       .format(i + 1, test_acc, args.dropout_ratio, best_model)))

    # save the predictions to args.result_dir/Graph Convolutional Networks/GCN_pool_%.3f_seed_%d_.csv
    result_path = args.result_dir
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    # 获取当前日期
    now_time = datetime.now()
    formatted_time = now_time.strftime("%Y_%m_%d_%H_%M_%S")
    # ------------------------------------------ feat: 保存时增加当前具体时间 ------------------------------------------
    result_df.to_csv(os.path.join(result_path,
                                  'GCN_pool_%.3f_seed_%d_%s.csv' % (args.pooling_ratio, args.seed, formatted_time)),
                     index=False, header=True)
    # ------------------------------------------ feat: 保存每次的平均指标 ------------------------------------------
    file = "index.txt"
    # 打开文件
    f = open(file, 'a', encoding='utf-8')
    # 统一保留8位小数对齐 且转为 str 格式
    acc_mean = formatOutput(meanOfArr(str2float(result_acc)))
    recall_mean = formatOutput(meanOfArr(str2float(result_recall)))
    pre_mean = formatOutput(meanOfArr(str2float(result_pre)))
    sc_mean = formatOutput(meanOfArr(str2float(result_sc)))
    f1_score_mean = formatOutput(meanOfArr(str2float(result_f1_score)))
    f.write(
        'Mean Accuracy: ' + acc_mean + ' Mean Recall: ' + recall_mean + ' Mean Precision: ' + pre_mean +
        ' Mean Specificity: ' + sc_mean + ' Mean F1 Score: ' + f1_score_mean)
    # 关闭文件
    f.close()
    print('Mean Accuracy: %f' % (sum(test_result_acc) / len(test_result_acc)))


def train_multiview_gcn(dataloader1, dataloader2, dataloader3, model, optimizer, save_path, args):
    """
    Training phase of multiview_GCN. No validation set is used here.
    :param save_path: working path for this progress
    :param dataloader1: dataloader1 of training set
    :param dataloader2: dataloader2 of training set
    :param dataloader3: dataloader3 of training set
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
        for data1, data2, data3 in zip(dataloader1, dataloader2, dataloader3):
            optimizer.zero_grad()
            data1 = data1.to(args.device)
            data2 = data2.to(args.device)
            data3 = data3.to(args.device)
            out, _ = model(data1.x, data1.edge_index, data2.edge_index, data3.edge_index, data1.edge_attr,
                           data2.edge_attr, data3.edge_attr)
            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(out[data1.train_mask], data1.y[data1.train_mask].float())
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            pred = (out[data1.train_mask] > 0).long()
            correct += pred.eq(data1.y[data1.train_mask]).sum().item()

        acc_train = correct / data1.train_mask.sum().item()
        acc_val, loss_val, _ = test_multiview_gcn(dataloader1, dataloader2, dataloader3, model, args, test=False)
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


def test_multiview_gcn(dataloader1, dataloader2, dataloader3, model, args, test=True):
    """
    Test the multiview_GCN performance on loaders. We have not use validation set in GCN.
    So, this is used to print the performance on test set
    :param dataloader1: an instance of torch_geometric.data.Dataloader
    :param dataloader2: an instance of torch_geometric.data.Dataloader
    :param dataloader3: an instance of torch_geometric.data.Dataloader
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
    for data1, data2, data3 in zip(dataloader1, dataloader2, dataloader3):
        data1 = data1.to(args.device)
        data2 = data2.to(args.device)
        data3 = data3.to(args.device)
        out, _ = model(data1.x, data1.edge_index, data2.edge_index, data3.edge_index, data1.edge_attr,
                       data2.edge_attr, data3.edge_attr)
        output += out.cpu().detach().numpy().tolist()
        if test:
            pred = (out[data1.test_mask] > 0).long()  # Predicted tensor
            length = data1.test_mask.sum().item()
            correct += pred.eq(data1.y[data1.test_mask]).sum().item()
            loss_test += criterion(out[data1.test_mask], data1.y[data1.test_mask].float()).item()
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
            true_list = data1.y[data1.test_mask].cpu().numpy().tolist()
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
            PRE = TP / (TP + FP) if TP + FP != 0 else 0.0
            RECALL = TP / (TP + FN) if TP + FN != 0 else 0.0
            SC = (TN / (TN + FP)) if TN + FP != 0 else 0.0
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
            pred = (out[data1.val_mask] > 0).long()
            length = data1.val_mask.sum().item()
            correct += pred.eq(data1.y[data1.val_mask]).sum().item()
            loss_test += criterion(out[data1.val_mask], data1.y[data1.val_mask].float()).item()
            return correct / length, loss_test, output
    return correct / length, loss_test, output, ACC, RECALL, PRE, SC, F1_SCORE
