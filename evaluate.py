import matplotlib

matplotlib.use('TkAgg')  # 设置后端
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, ndcg_score
from utils.data_loader import load_data
from cluster_recommend import generate_recommendations
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate():
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    _, train, test, _ = load_data()
    recommendations, _, _ = generate_recommendations()

    precisions = []
    recalls = []
    f1_scores = []
    ndcgs = []

    for user_id in test['userId'].unique():
        if user_id not in recommendations:
            continue

        actual = test[test['userId'] == user_id]['movieId'].values
        pred = recommendations[user_id]

        if len(actual) == 0 or len(pred) == 0:
            continue

        y_true = np.isin(pred, actual).astype(int)
        y_score = np.ones(len(pred))

        precisions.append(precision_score(y_true, [1] * len(y_true), zero_division=0))
        recalls.append(recall_score(y_true, [1] * len(y_true), zero_division=0))
        f1_scores.append(f1_score(y_true, [1] * len(y_true), zero_division=0))
        ndcgs.append(ndcg_score([y_true], [y_score]))

    metrics = {
        'Precision': np.mean(precisions),
        'Recall': np.mean(recalls),
        'F1': np.mean(f1_scores),
        'NDCG': np.mean(ndcgs)
    }

    pd.DataFrame([metrics]).to_csv('results.csv', index=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
    plt.title('Recommendation Performance Metrics')
    plt.ylabel('Score')
    plt.savefig('metrics.png')
    plt.show()  # 如果仍然报错，可以注释掉这一行

    return metrics


if __name__ == '__main__':
    metrics = evaluate()
    print("Evaluation Results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")