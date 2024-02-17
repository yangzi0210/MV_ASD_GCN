import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import coo_matrix

G = nx.Graph()

# plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

if __name__ == '__main__':
    # 假设你的邻接矩阵和边权重保存在csv文件中
    # 请根据实际情况替换文件路径
    # 读取图结构数据集
    root_path = './data/multiview_graph'
    adj_age_path = os.path.join(root_path, 'ABIDE_age.adj')
    attr_age_path = os.path.join(root_path, 'ABIDE_age.attr')
    adj_sex_path = os.path.join(root_path, 'ABIDE_sex.adj')
    attr_sex_path = os.path.join(root_path, 'ABIDE_sex.attr')
    adj_site_path = os.path.join(root_path, 'ABIDE_site.adj')
    attr_site_path = os.path.join(root_path, 'ABIDE_site.attr')
    downsample_file = os.path.join('./data', 'ABIDE_downsample',
                                   'ABIDE_pool_{:.3f}_.txt'.format(0.050))
    downsample = pd.read_csv(downsample_file, header=None, sep='\t').values
    # ndarry
    edge_age_index = pd.read_csv(adj_age_path, header=None).values
    edge_age_attr = pd.read_csv(attr_age_path, header=None).values.reshape(-1)

    edge_sex_index = pd.read_csv(adj_sex_path, header=None).values
    edge_sex_attr = pd.read_csv(attr_sex_path, header=None).values.reshape(-1)

    edge_site_index = pd.read_csv(adj_site_path, header=None).values
    edge_site_attr = pd.read_csv(attr_site_path, header=None).values.reshape(-1)

    num_nodes = np.max(edge_age_index) + 1
    # 使用coo_matrix创建稀疏邻接矩阵
    adj_matrix = coo_matrix((edge_age_attr, (edge_age_index[0], edge_age_index[1])), shape=(num_nodes, num_nodes)).toarray()

    # 打印结果

    # 创建无向图
    G = nx.Graph()

    # 根据邻接矩阵添加边到图中
    # 邻接矩阵是方阵，行和列的大小相同
    # for i in range(871):
    #     for j in range(871):
    #         if sim_ages_matrix[i, j] >= 0.5 and i > j:
    #             adj_age.append([i, j])
    #             att_age.append(sim_ages_matrix[i, j])

    for i in range(16):
        for j in range(16):  # 避免重复添加边，只遍历上三角矩阵
            weight = float(format(adj_matrix[i][j], ".2f"))
            if weight != 0:  # 如果权重不为0，则添加边
                G.add_edge(i, j, weight=weight)

    # 绘制图
    pos = nx.spring_layout(G)  # 为图设置布局
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]

    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size=60)
    # 绘制边
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights)
    # 绘制节点标签
    nx.draw_networkx_labels(G, pos, font_size=6, font_family='sans-serif')
    # 绘制边权重标签
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # 显示图
    plt.axis('off')  # 不显示坐标轴
    plt.show()
