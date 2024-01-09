import os
import pandas as pd
import torch
from sklearn.metrics import euclidean_distances
from torch import Tensor
from typing import Optional, Union, Tuple
from torch_geometric.utils import scatter, softmax
from torch_geometric.utils.num_nodes import maybe_num_nodes


def calculate_similarity_matrix_euclidean(vectors):
    """
    计算向量集合的欧几里德距离相似度矩阵

    参数:
    vectors (numpy.ndarray): 形状为 (n_samples, n_features) 的二维NumPy数组，其中n_samples是向量的数量，n_features是每个向量的特征数。

    返回:
    similarity_matrix (numpy.ndarray): 形状为 (n_samples, n_samples) 的相似度矩阵。
    """
    # 计算欧几里德距离
    distances = euclidean_distances(vectors)

    # 将距离转换为相似度（取倒数）
    similarity_matrix = 1 / (1 + distances)

    return similarity_matrix


def read_dataset():
    # 读取图结构数据集
    root_path = './data/multiview_graph'
    adj_age_path = os.path.join(root_path, 'ABIDE_age.adj')
    attr_age_path = os.path.join(root_path, 'ABIDE_age.attr')
    adj_sex_path = os.path.join(root_path, 'ABIDE_sex.adj')
    attr_sex_path = os.path.join(root_path, 'ABIDE_sex.attr')
    adj_site_path = os.path.join(root_path, 'ABIDE_site.adj')
    attr_site_path = os.path.join(root_path, 'ABIDE_site.attr')

    edge_age_index = pd.read_csv(adj_age_path, header=None).values
    edge_age_attr = pd.read_csv(attr_age_path, header=None).values.reshape(-1)
    edge_age_index = torch.tensor(edge_age_index, dtype=torch.long)
    edge_age_attr = torch.tensor(edge_age_attr, dtype=torch.float)

    edge_sex_index = pd.read_csv(adj_sex_path, header=None).values
    edge_sex_attr = pd.read_csv(attr_sex_path, header=None).values.reshape(-1)
    edge_sex_index = torch.tensor(edge_sex_index, dtype=torch.long)
    edge_sex_attr = torch.tensor(edge_sex_attr, dtype=torch.float)

    edge_site_index = pd.read_csv(adj_site_path, header=None).values
    edge_site_attr = pd.read_csv(attr_site_path, header=None).values.reshape(-1)
    edge_site_index = torch.tensor(edge_site_index, dtype=torch.long)
    edge_site_attr = torch.tensor(edge_site_attr, dtype=torch.float)

    return edge_age_index, edge_age_attr, edge_sex_index, edge_sex_attr, edge_site_index, edge_site_attr


def str2float(arr):
    for i in range(len(arr)):
        arr[i] = float(arr[i])
    return arr


def formatOutput(output):
    return "{:.8f}".format(output)


def meanOfArr(arr):
    return sum(arr) / len(arr)


def cumsum(x: Tensor, dim: int = 0) -> Tensor:
    r"""Returns the cumulative sum of elements of :obj:`x`.
    In contrast to :meth:`torch.cumsum`, prepends the output with zero.

    Args:
        x (torch.Tensor): The input tensor.
        dim (int, optional): The dimension to do the operation over.
            (default: :obj:`0`)

    Example:
        >>> x = torch.tensor([2, 4, 1])
        >>> cumsum(x)
        tensor([0, 2, 6, 7])

    """
    size = x.size()[:dim] + (x.size(dim) + 1,) + x.size()[dim + 1:]
    out = x.new_empty(size)

    out.narrow(dim, 0, 1).zero_()
    torch.cumsum(x, dim=dim, out=out.narrow(dim, 1, x.size(dim)))

    return out


def topk(
        x: Tensor,
        ratio: Optional[Union[float, int]],
        batch: Tensor,
        min_score: Optional[float] = None,
        tol: float = 1e-7,
) -> Tensor:
    if min_score is not None:
        # Make sure that we do not drop all nodes in a graph.
        scores_max = scatter(x, batch, reduce='max')[batch] - tol
        scores_min = scores_max.clamp(max=min_score)

        perm = (x > scores_min).nonzero().view(-1)
        return perm

    if ratio is not None:
        num_nodes = scatter(batch.new_ones(x.size(0)), batch, reduce='sum')

        if ratio >= 1:
            k = num_nodes.new_full((num_nodes.size(0),), int(ratio))
        else:
            k = (float(ratio) * num_nodes.to(x.dtype)).ceil().to(torch.long)

        x, x_perm = torch.sort(x.view(-1), descending=True)
        batch = batch[x_perm]
        batch, batch_perm = torch.sort(batch, descending=False, stable=True)

        arange = torch.arange(x.size(0), dtype=torch.long, device=x.device)
        ptr = cumsum(num_nodes)
        batched_arange = arange - ptr[batch]
        mask = batched_arange < k[batch]

        return x_perm[batch_perm[mask]]


def filter_adj(
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
        node_index: Tensor,
        cluster_index: Optional[Tensor] = None,
        num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if cluster_index is None:
        cluster_index = torch.arange(node_index.size(0),
                                     device=node_index.device)

    mask = node_index.new_full((num_nodes,), -1)
    mask[node_index] = cluster_index

    row, col = edge_index[0], edge_index[1]
    row, col = mask[row], mask[col]
    mask = (row >= 0) & (col >= 0)
    row, col = row[mask], col[mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    return torch.stack([row, col], dim=0), edge_attr
