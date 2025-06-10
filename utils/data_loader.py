import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Data
import yaml


def load_data(config_path='config.yaml'):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 加载原始数据
    ratings = pd.read_csv(config['data_path']['ratings'])
    movies = pd.read_csv(config['data_path']['movies'])

    # 数据预处理
    ratings = ratings[['userId', 'movieId', 'rating']]
    ratings['userId'] = ratings['userId'].astype('category').cat.codes.values
    ratings['movieId'] = ratings['movieId'].astype('category').cat.codes.values

    # 划分训练测试集
    train, test = train_test_split(
        ratings,
        test_size=config['test_size'],
        random_state=config['seed']
    )

    # 构建图数据
    edge_index = []
    for _, row in train.iterrows():
        u = row['userId']
        m = row['movieId'] + config['n_users']  # 物品ID偏移

        # 无向图添加双向边
        edge_index.append([u, m])
        edge_index.append([m, u])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x = torch.ones((config['n_users'] + config['n_items'], config['embed_dim']))

    graph_data = Data(x=x, edge_index=edge_index)

    return graph_data, train, test, movies


def get_mappings(config_path='config.yaml'):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    ratings = pd.read_csv(config['data_path']['ratings'])
    movies = pd.read_csv(config['data_path']['movies'])

    user_to_idx = {u: i for i, u in enumerate(ratings['userId'].unique())}
    item_to_idx = {m: i for i, m in enumerate(ratings['movieId'].unique())}
    idx_to_item = {i: m for m, i in item_to_idx.items()}

    return user_to_idx, item_to_idx, idx_to_item, movies