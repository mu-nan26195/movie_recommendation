import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from utils.data_loader import load_data
import yaml
from tqdm import tqdm


class GCN(nn.Module):
    def __init__(self, config):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(config['embed_dim'], 128)
        self.conv2 = GCNConv(128, config['embed_dim'])

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def train():
    with open('config.yaml', 'r', encoding='utf-8-sig') as f:  # 处理 BOM
        config = yaml.safe_load(f)
        config['weight_decay'] = float(config['weight_decay'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data, _, _, _ = load_data()
    data = data.to(device)

    model = GCN(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['lr'],
                                 weight_decay=config['weight_decay'])

    model.train()
    for epoch in tqdm(range(config['n_epochs'])):
        optimizer.zero_grad()
        out = model(data)

        # 链接预测损失
        pos_out = (out[data.edge_index[0]] * out[data.edge_index[1]]).sum(dim=1)
        pos_loss = -torch.log(torch.sigmoid(pos_out) + 1e-15).mean()

        # 负采样
        neg_edge_index = torch.randint(0, config['n_users'] + config['n_items'],
                                       data.edge_index.size(),
                                       dtype=torch.long).to(device)
        neg_out = (out[neg_edge_index[0]] * out[neg_edge_index[1]]).sum(dim=1)
        neg_loss = -torch.log(1 - torch.sigmoid(neg_out) + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch + 1}, Loss: {loss.item():.4f}')

    # 保存模型
    torch.save(model.state_dict(), config['model_path']['gcn'])
    print("GCN training completed and model saved.")


if __name__ == '__main__':
    train()