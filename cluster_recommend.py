import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
from sklearn.cluster import KMeans
from utils.data_loader import load_data, get_mappings
import yaml
import joblib
from tqdm import tqdm


class GCN(torch.nn.Module):
    """Wrapper for GCN model to load weights"""

    def __init__(self, config):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(config['embed_dim'], 128)
        self.conv2 = GCNConv(128, config['embed_dim'])

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


def generate_recommendations():
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    # 加载模型和数据
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(config).to(device)
    model.load_state_dict(torch.load(config['model_path']['gcn']))
    model.eval()

    data, train, _, _ = load_data()
    data = data.to(device)

    with torch.no_grad():
        embeddings = model(data).cpu().numpy()

    user_embeddings = embeddings[:config['n_users']]
    item_embeddings = embeddings[config['n_users']:]

    # K-means聚类
    kmeans = KMeans(n_clusters=config['n_clusters'], random_state=config['seed'])
    user_clusters = kmeans.fit_predict(user_embeddings)
    joblib.dump(kmeans, config['model_path']['kmeans'])

    # 为每个聚类构建推荐
    user_to_idx, item_to_idx, idx_to_item, _ = get_mappings()

    # 确保 item_to_idx 的键是 int 类型
    item_to_idx = {int(k): v for k, v in item_to_idx.items()}

    cluster_items = {}
    for cluster in range(config['n_clusters']):
        cluster_users = np.where(user_clusters == cluster)[0]
        items = set()

        # 找出聚类中用户交互过的所有物品
        for u in cluster_users:
            user_id = list(user_to_idx.keys())[list(user_to_idx.values()).index(u)]
            interacted = train[train['userId'] == user_id]['movieId'].values
            items.update([int(m) for m in interacted])  # 确保 movieId 是 int 类型

        cluster_items[cluster] = list(items)

    # 生成推荐
    recommendations = {}
    for user_id in tqdm(user_to_idx.keys()):
        user_idx = user_to_idx[user_id]
        cluster = user_clusters[user_idx]
        candidates = [m for m in cluster_items[cluster] if m in item_to_idx]  # 过滤无效 ID

        if not candidates:
            continue

        # 计算用户与候选物品的相似度
        user_emb = user_embeddings[user_idx]
        item_indices = [item_to_idx[m] for m in candidates]  # 此时 m 已经是 int 类型
        item_embs = item_embeddings[item_indices]

        scores = user_emb @ item_embs.T
        top_indices = np.argsort(scores)[-config['top_k']:][::-1]

        recommendations[user_id] = [candidates[i] for i in top_indices]

    return recommendations, user_embeddings, item_embeddings


if __name__ == '__main__':
    recs, _, _ = generate_recommendations()
    print(f"Generated recommendations for {len(recs)} users")