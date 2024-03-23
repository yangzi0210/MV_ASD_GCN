"""
Construct the graph representation of brain imaging and population graph
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

from util import calculate_similarity_matrix_euclidean


def brain_graph(logs, atlas, path, data_folder):
    if not os.path.exists(path):
        os.makedirs(path)
    # the global mean is not included in ho_labels.csv
    atlas.loc[-1] = [3455, 'Background']
    print(atlas.shape)
    # label the regions as right/left/global mean
    label = []
    for e in atlas['area'].values:
        if e.startswith('Left'):
            label.append(0)
        elif e.startswith('Right'):
            label.append(1)
        else:
            label.append(-1)

    atlas['label'] = label
    atlas.sort_values('index', inplace=True)
    atlas = atlas.reset_index().drop('level_0', axis=1)

    ###################
    # Adjacent matrix #
    ###################
    print('Processing the adjacent matrix...')
    # now the index in [0, 110]
    adj = np.zeros([111, 111])
    not_right = [i for i in range(111) if atlas['label'][i] != 1]
    not_left = [i for i in range(111) if atlas['label'][i] != 0]
    not_gb = [i for i in range(111) if atlas['label'][i] != -1]

    # Build the bipartite brain graph
    for idx in range(111):
        if atlas['label'][idx] == 0:
            adj[idx, not_left] = 1
        elif atlas['label'][idx] == 1:
            adj[idx, not_right] = 1
        elif atlas['label'][idx] == -1:
            adj[idx, not_gb] = 1

    # now form the sparse adj matrix
    # node id:[1, 111*871]
    node_ids = np.array_split(np.arange(1, 111 * 871 + 1), 871)
    adj_matrix = []
    for i in range(871):
        node_id = node_ids[i]
        for j in range(111):
            for k in range(111):
                if adj[j, k]:
                    adj_matrix.append([node_id[j], node_id[k]])

    # save sparse adj matrix
    pd.DataFrame(adj_matrix).to_csv(os.path.join(path, 'ABIDE_A.txt'), index=False, header=False)
    print('Done!')

    ###################
    # Graph indicator #
    ###################
    print('processing the graph indicator...')
    indicator = np.repeat(np.arange(1, 872), 111)
    pd.DataFrame(indicator).to_csv(os.path.join(path, 'ABIDE_graph_indicator.txt'), index=False, header=False)
    print('Done!')

    ###################
    #   Graph labels  #
    ###################
    print('processing the graph labels...')
    graph_labels = logs[['label']]
    graph_labels.to_csv(os.path.join(path, 'ABIDE_graph_labels.txt'), index=False, header=False)
    print('Done!')

    ###################
    # Node Attributes #
    ###################
    print('processing the node attributes...')
    # follow the order in log.csv
    files = logs['file_name']
    node_att = pd.DataFrame([])
    for file in files:
        file_path = os.path.join(data_folder, file)
        # data collected from different site
        # may have different time length (rows in the data file)
        # Here I simply cut them off according to
        # the shortest one, 78.
        ho_rois = pd.read_csv(file_path, sep='\t').iloc[:78, :].T
        node_att = pd.concat([node_att, ho_rois])
    # 96681 * 785 = 111 * 871 * 785
    node_att.to_csv(os.path.join(path, 'ABIDE_node_attributes.txt'), index=False, header=False)

    print('The shape of node attributes is (%d, %d)' % node_att.shape)
    print('Done!')

    ###################
    #   Node labels   #
    ###################
    print('processing the node labels...')
    # Make sure all the downloaded files have the same column (brian regions) order
    cols = list(pd.read_csv(file_path, sep='\t').columns.values)
    for file in files:
        file_path = os.path.join(data_folder, file)
        temp_cols = list(pd.read_csv(file_path, sep='\t').columns.values)
        assert cols == temp_cols, 'Inconsistent order of brain regions in ABIDE pcp!'

    node_label = np.arange(111)
    node_labels = np.tile(node_label, 871)
    pd.DataFrame(node_labels).to_csv(os.path.join(path, 'ABIDE_node_labels.txt'), index=False, header=False)
    print('Done!')


def population_graph(args):
    """
    Build the population graph. The nodes are connected if their cosine similarity is above 0.5
    in terms of phenotypic information: gender, site, age.
    :param args: args from main.py
    :return: adj, att: adjacency matrix and edge weights
    """

    # considering phenotypic information: gender, age and site

    cluster_att = ['SEX', 'SITE_ID']
    # get text information: sex, site
    logs = pd.read_csv(os.path.join(args.data_dir, 'phenotypic', 'log.csv'))
    # text_info = 871 * 2
    text_info = logs[cluster_att].values

    enc = OneHotEncoder()
    enc.fit(text_info)
    # text_feature: 871 * 22
    text_feature = enc.transform(text_info).toarray()

    # take ages into consideration
    ages = logs['AGE_AT_SCAN'].values
    # Normalization
    ages = (ages - min(ages)) / (max(ages) - min(ages))
    # 871 * 23
    cluster_features = np.c_[text_feature, ages]

    adj = []
    att = []
    # 871 * 871
    sim_matrix = cosine_similarity(cluster_features)

    ages_features = ages.reshape(871, 1)
    # 站点 性别
    sim_site_sex_matrix = cosine_similarity(text_feature)

    # 年龄的欧式距离作为相似度矩阵
    sim_ages_matrix = calculate_similarity_matrix_euclidean(ages_features)

    # 原方法
    for i in range(871):
        for j in range(871):
            if sim_matrix[i, j] > 0.5 and i > j:
                adj.append([i, j])
                att.append(sim_matrix[i, j])

    adj = np.array(adj).T
    att = np.array([att]).T

    if not os.path.exists(os.path.join(args.data_dir, 'population graph')):
        os.makedirs(os.path.join(args.data_dir, 'population graph'))

    pd.DataFrame(adj).to_csv(os.path.join(args.data_dir, 'population graph', 'ABIDE.adj'), index=False, header=False)
    pd.DataFrame(att).to_csv(os.path.join(args.data_dir, 'population graph', 'ABIDE.attr'), index=False, header=False)

    # 建立年龄 性别两个视图
    # 年龄
    adj_age = []
    att_age = []
    for i in range(871):
        for j in range(871):
            if sim_ages_matrix[i, j] > 0.5 and i > j:
                adj_age.append([i, j])
                att_age.append(sim_ages_matrix[i, j])

    adj_age = np.array(adj_age).T
    att_age = np.array([att_age]).T

    if not os.path.exists(os.path.join(args.data_dir, 'Multi_view_graph')):
        os.makedirs(os.path.join(args.data_dir, 'Multi_view_graph'))

    pd.DataFrame(adj_age).to_csv(os.path.join(args.data_dir, 'Multi_view_graph', 'ABIDE_age.adj'), index=False,
                                 header=False)
    pd.DataFrame(att_age).to_csv(os.path.join(args.data_dir, 'Multi_view_graph', 'ABIDE_age.attr'), index=False,
                                 header=False)
    # 性别
    adj_sex = []
    att_sex = []
    for i in range(871):
        for j in range(871):
            if sim_site_sex_matrix[i, j] > 0.5 and i > j:
                adj_sex.append([i, j])
                att_sex.append(sim_site_sex_matrix[i, j])

    adj_sex = np.array(adj_sex).T
    att_sex = np.array([att_sex]).T

    pd.DataFrame(adj_sex).to_csv(os.path.join(args.data_dir, 'Multi_view_graph', 'ABIDE_sex.adj'), index=False,
                                 header=False)
    pd.DataFrame(att_sex).to_csv(os.path.join(args.data_dir, 'Multi_view_graph', 'ABIDE_sex.attr'), index=False,
                                 header=False)

def multiview_graph(args):
    """
    Build the multiview population graph.
    The nodes are connected if their euclidean distances similarity >= 0.5
    in terms of phenotypic information: gender, site, age.
    :param args: args from main.py
    :return: adj, att: adjacency matrix and edge weights
    """
    # considering phenotypic information: gender, age and site

    # get text information: sex, site
    logs = pd.read_csv(os.path.join(args.data_dir, 'phenotypic', 'log.csv'))
    # text_info = 871 * 2
    site_info = logs['SITE_ID'].values.reshape(871, 1)

    sex_info = logs['SEX'].values.reshape(871, 1)

    # take ages into consideration
    ages = logs['AGE_AT_SCAN'].values
    # Normalization
    ages = (ages - min(ages)) / (max(ages) - min(ages))
    ages_features = ages.reshape(871, 1)
    # HANDEDNESS_CATEGORY HANDEDNESS_SCORES 不全 想一想处理
    # ndarray
    handedness_category = logs['HANDEDNESS_CATEGORY'].values
    # handedness_scores = logs['HANDEDNESS_SCORES'].values

    enc = OneHotEncoder()
    enc.fit(site_info)
    # site_feature: 871 * 22
    site_feature = enc.transform(site_info).toarray()

    # 上面这段代码是Python中使用机器学习库scikit-learn的一部分，用于对文本信息text_info进行独热编码（One-Hot Encoding）
    # 独热编码是一种将分类变量转换为机器学习模型可以理解的形式的技术。在这个过程中，每个唯一的类别值都会被转换为一个二进制向量，其中只有一个元素是1，其余都是0。
    # 下面是代码的详细解释：
    # enc = OneHotEncoder()
    # 这行代码创建了一个OneHotEncoder对象enc。OneHotEncoder是scikit-learn库中用于进行独热编码的一个类。
    # enc.fit(text_info)
    # 接下来，使用fit方法来“训练”独热编码器。这并不是训练一个预测模型的意义上的训练，而是指让编码器学习text_info中的所有类别。
    # 这样，编码器就能知道有多少个不同的类别，以及每个类别应该如何编码成独热向量。
    # text_feature = enc.transform(text_info).toarray()
    # 然后使用transform方法将text_info数据转换为独热编码格式。
    # transform方法会将每个类别值转换为一个独特的二进制向量。
    # toarray()方法是因为transform方法返回的是一个稀疏矩阵（为了节省内存，因为独热编码会产生很多0）。toarray()方法将这个稀疏矩阵转换为一个常规的NumPy数组，方便后续的处理。
    # 在这段代码中，text_info可能是一个包含类别数据的列表、数组或者pandas的DataFrame。最终，text_feature将包含text_info的独热编码表示，可以直接用于机器学习模型的训练和预测。

    # 871 * 23
    adj_age = []
    att_age = []
    adj_sex = []
    att_sex = []
    adj_site = []
    att_site = []
    # 871 * 871     欧式距离作为相似度矩阵
    # 性别
    sim_sex_matrix = calculate_similarity_matrix_euclidean(sex_info)
    # 站点
    sim_site_matrix = calculate_similarity_matrix_euclidean(site_feature)
    # 年龄
    sim_ages_matrix = calculate_similarity_matrix_euclidean(ages_features)

    for i in range(871):
        for j in range(871):
            if sim_ages_matrix[i, j] >= 0.5 and i > j:
                adj_age.append([i, j])
                att_age.append(sim_ages_matrix[i, j])

    adj_age = np.array(adj_age).T
    att_age = np.array([att_age]).T

    for i in range(871):
        for j in range(871):
            if sim_sex_matrix[i, j] >= 0.5 and i > j:
                adj_sex.append([i, j])
                att_sex.append(sim_sex_matrix[i, j])

    adj_sex = np.array(adj_sex).T
    att_sex = np.array([att_sex]).T

    for i in range(871):
        for j in range(871):
            if sim_site_matrix[i, j] >= 0.5 and i > j:
                adj_site.append([i, j])
                att_site.append(sim_site_matrix[i, j])

    adj_site = np.array(adj_site).T
    att_site = np.array([att_site]).T

    if not os.path.exists(os.path.join(args.data_dir, 'multiview_graph')):
        os.makedirs(os.path.join(args.data_dir, 'multiview_graph'))

    # 保存多视图图结构信息
    pd.DataFrame(adj_age).to_csv(os.path.join(args.data_dir, 'multiview_graph', 'ABIDE_age.adj'), index=False,
                                 header=False)
    pd.DataFrame(att_age).to_csv(os.path.join(args.data_dir, 'multiview_graph', 'ABIDE_age.attr'), index=False,
                                 header=False)
    pd.DataFrame(adj_sex).to_csv(os.path.join(args.data_dir, 'multiview_graph', 'ABIDE_sex.adj'), index=False,
                                 header=False)
    pd.DataFrame(att_sex).to_csv(os.path.join(args.data_dir, 'multiview_graph', 'ABIDE_sex.attr'), index=False,
                                 header=False)
    pd.DataFrame(adj_site).to_csv(os.path.join(args.data_dir, 'multiview_graph', 'ABIDE_site.adj'), index=False,
                                  header=False)
    pd.DataFrame(att_site).to_csv(os.path.join(args.data_dir, 'multiview_graph', 'ABIDE_site.attr'), index=False,
                                  header=False)

