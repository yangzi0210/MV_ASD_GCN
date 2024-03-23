# ASD_DS_GAD

基于多视图、异常检测、图卷积网络等机器学习技术进行 ASD 患者的诊断分类模型

数据集为 ABIDE I

预处理方法为 CPAC

## Requirements

torch_geometric 等库直接安装可能会有问题 参考网上教程

```

torch >= 1.8.1+cu102

torch-cluster >= 1.5.9

torch-geometric >= 1.7.0

torch-scatter >= 2.0.7

torch-sparse >= 0.6.10

sklearn

nilearn

```

## 使用

```
# 多视图模型
$ python multiview_main.py
# 单视图模型
$ python main.py
# 可视化
$ python visualize.py
```

## 参考

```
@article{pan2021identifying,
  title={Identifying Autism Spectrum Disorder Based on Individual-Aware Down-Sampling and Multi-Modal Learning},
  author={Pan, Li and Liu, Jundong and Shi, Mingqin and Wong, Chi Wah and Chan, Kei Hang Katie},
  journal={arXiv preprint arXiv:2109.09129},
  year={2021}
}
```
