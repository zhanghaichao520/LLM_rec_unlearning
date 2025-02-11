import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np


# 嵌入层
class CrossAttentionRecommender(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(CrossAttentionRecommender, self).__init__()

        # 用户和物品的嵌入层
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # Cross-Attention的参数
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=1)

        # 定义前馈网络
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, user_ids, item_ids, historical_item_ids):
        # 获取用户和候选物品的嵌入表示
        user_embed = self.user_embedding(user_ids).unsqueeze(0)  # shape: (1, batch_size, embedding_dim)
        item_embed = self.item_embedding(item_ids).unsqueeze(0)  # shape: (1, batch_size, embedding_dim)

        # 获取历史物品的嵌入表示
        historical_item_embed = self.item_embedding(
            historical_item_ids)  # shape: (batch_size, num_history, embedding_dim)

        # 使用Cross-Attention计算历史行为和候选物品之间的相关性
        # 输入格式为 (seq_len, batch_size, embed_dim)，这里我们将历史行为作为"query"，候选物品作为"key"
        attention_output, attention_weights = self.attention(historical_item_embed, item_embed.transpose(0, 1),
                                                             item_embed.transpose(0, 1))

        # 计算加权后的历史行为嵌入
        weighted_history_embed = torch.sum(attention_output, dim=0)  # shape: (batch_size, embedding_dim)

        # 计算用户与加权历史行为的融合嵌入
        user_history_embed = user_embed.squeeze(0) + weighted_history_embed

        # 通过前馈网络得到评分预测
        score = self.fc(user_history_embed)  # shape: (batch_size, 1)

        return score, attention_weights



# 训练代码
def train(model, history_df, candidate_items, num_epochs=5, batch_size=32, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for user_id, candidate_item_list in candidate_items.items():
            # 获取用户的历史行为
            user_history = history_df[history_df['user_id'] == user_id]

            historical_item_ids = torch.tensor(user_history['item_id'].values, dtype=torch.long)
            ratings = torch.tensor(user_history['rating'].values, dtype=torch.float32)

            # 随机选择batch
            indices = np.random.choice(len(historical_item_ids), batch_size, replace=False)
            historical_item_ids_batch = historical_item_ids[indices]
            ratings_batch = ratings[indices]

            # 构建候选物品的tensor
            item_ids_batch = torch.tensor(candidate_item_list, dtype=torch.long)
            user_ids_batch = torch.tensor([user_id] * batch_size, dtype=torch.long)

            # 前向传播
            optimizer.zero_grad()
            score, attention_weights = model(user_ids_batch, item_ids_batch, historical_item_ids_batch)

            # 计算损失
            loss = criterion(score.squeeze(), ratings_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(candidate_items)}")


# 保存训练好的模型
def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


if __name__ == '__main__':
    # 假设数据已加载为DataFrame格式
    # 用户历史交互数据
    history_df = pd.read_csv("ml-100k.inter", sep=" ", names=["user_id", "item_id", "rating", "timestamp"])

    # 假设候选物品列表
    # candidate_items是一个字典，key是user_id，value是该用户的候选物品id列表
    candidate_items = {
        5239: [2054, 2058, 587],
        # 更多用户的候选物品...
    }

    # 创建模型
    embedding_dim = 64
    num_users = history_df['user_id'].nunique()
    num_items = history_df['item_id'].nunique()

    model = CrossAttentionRecommender(num_users, num_items, embedding_dim)
    # 训练模型
    train(model, history_df, candidate_items)
    # 假设训练完成后保存模型
    save_model(model, "saved/cross_attention_recommender.pth")

