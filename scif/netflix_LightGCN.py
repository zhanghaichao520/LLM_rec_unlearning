import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed, init_logger, get_model, get_trainer

# ------------------------------------------------------------
# 1. 数据预处理：将原始 Netflix 或 ML-100K 拆分成不做二次映射的 CSV
# ------------------------------------------------------------
def process_netflix_data(original_file, output_dir, forget_ratio=0.1):
    print("Reading original data...")
    # 原始文件格式：item \t user \t rating \t timestamp
    df = pd.read_csv(
        original_file,
        sep='\t',
        engine='python',
        names=['item', 'user', 'rating', 'timestamp']
    )
    df['label'] = (df['rating'] >= 4).astype(int)

    print(df.head())
    print(f"positive_samples: {len(df[df['rating'] >= 4])}")
    print(f"negative_samples: {len(df[df['rating'] < 4])}")

    user_groups = df.groupby('user')
    all_users = list(user_groups.groups.keys())
    forget_users = set(
        np.random.choice(all_users,
                         size=int(len(all_users) * forget_ratio),
                         replace=False)
    )

    train_normal = []
    train_random = []
    valid = []
    test = []

    # 注意：读到的每一行格式为 [item, user, rating, timestamp, label]
    # 需要把它转换成 [user, item, rating, timestamp, label]
    for user, group in user_groups:
        if user in forget_users:
            n_samples = len(group)
            if n_samples >= 3:
                group = group.sample(frac=1, random_state=42)
                train_size = int(n_samples * 0.6)
                valid_size = int(n_samples * 0.2)
                # 前 train_size 条放 train_random
                for row in group.iloc[:train_size].itertuples(index=False):
                    train_random.append([row.user, row.item, row.rating, row.timestamp, row.label])
                # 接下来 valid_size 条放 valid
                for row in group.iloc[train_size:train_size+valid_size].itertuples(index=False):
                    valid.append([row.user, row.item, row.rating, row.timestamp, row.label])
                # 剩余放 test
                for row in group.iloc[train_size+valid_size:].itertuples(index=False):
                    test.append([row.user, row.item, row.rating, row.timestamp, row.label])
        else:
            n_samples = len(group)
            if n_samples >= 3:
                group = group.sample(frac=1, random_state=42)
                train_size = int(n_samples * 0.6)
                valid_size = int(n_samples * 0.2)
                for row in group.iloc[:train_size].itertuples(index=False):
                    train_normal.append([row.user, row.item, row.rating, row.timestamp, row.label])
                for row in group.iloc[train_size:train_size+valid_size].itertuples(index=False):
                    valid.append([row.user, row.item, row.rating, row.timestamp, row.label])
                for row in group.iloc[train_size+valid_size:].itertuples(index=False):
                    test.append([row.user, row.item, row.rating, row.timestamp, row.label])

    print("Saving processed data...")
    train_normal_df = pd.DataFrame(
        train_normal,
        columns=['user','item','rating','timestamp','label']
    )
    train_random_df = pd.DataFrame(
        train_random,
        columns=['user','item','rating','timestamp','label']
    )
    valid_df = pd.DataFrame(
        valid,
        columns=['user','item','rating','timestamp','label']
    )
    test_df = pd.DataFrame(
        test,
        columns=['user','item','rating','timestamp','label']
    )

    os.makedirs(output_dir, exist_ok=True)
    train_normal_df.to_csv(os.path.join(output_dir, 'train_normal.csv'), index=False)
    train_random_df.to_csv(os.path.join(output_dir, 'train_random.csv'), index=False)
    valid_df.to_csv(os.path.join(output_dir, 'valid.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    print("Computing average labels...")
    all_data = pd.concat([train_normal_df, train_random_df, valid_df, test_df], ignore_index=True)
    avg_labels = all_data.groupby(['user', 'item'], sort=False)['label'].mean().reset_index()
    user_item_pairs = avg_labels[['user','item']].to_numpy()
    avg_labels_array = avg_labels['label'].to_numpy()

    np.save(os.path.join(output_dir, 'avg_labels.npy'), avg_labels_array)
    np.save(os.path.join(output_dir, 'user_item_pairs.npy'), user_item_pairs)
    print("Data processing completed.")


def process_ml100k_data(original_file, output_dir, forget_ratio=0.1):
    print("Reading original data...")
    df = pd.read_csv(
        original_file,
        sep='\t',
        engine='python',
        names=['user','item','rating','timestamp']
    )
    df['label'] = (df['rating'] >= 4).astype(int)

    user_groups = df.groupby('user')
    all_users = list(user_groups.groups.keys())
    forget_users = set(
        np.random.choice(all_users,
                         size=int(len(all_users) * forget_ratio),
                         replace=False)
    )

    train_normal = []
    train_random = []
    valid = []
    test = []

    for user, group in user_groups:
        if user in forget_users:
            n_samples = len(group)
            if n_samples >= 3:
                group = group.sample(frac=1, random_state=42)
                train_size = int(n_samples * 0.6)
                valid_size = int(n_samples * 0.2)
                for row in group.iloc[:train_size].itertuples(index=False):
                    train_random.append([row.user, row.item, row.rating, row.timestamp, row.label])
                for row in group.iloc[train_size:train_size+valid_size].itertuples(index=False):
                    valid.append([row.user, row.item, row.rating, row.timestamp, row.label])
                for row in group.iloc[train_size+valid_size:].itertuples(index=False):
                    test.append([row.user, row.item, row.rating, row.timestamp, row.label])
        else:
            n_samples = len(group)
            if n_samples >= 3:
                group = group.sample(frac=1, random_state=42)
                train_size = int(n_samples * 0.6)
                valid_size = int(n_samples * 0.2)
                for row in group.iloc[:train_size].itertuples(index=False):
                    train_normal.append([row.user, row.item, row.rating, row.timestamp, row.label])
                for row in group.iloc[train_size:train_size+valid_size].itertuples(index=False):
                    valid.append([row.user, row.item, row.rating, row.timestamp, row.label])
                for row in group.iloc[train_size+valid_size:].itertuples(index=False):
                    test.append([row.user, row.item, row.rating, row.timestamp, row.label])

    print("Saving processed data...")
    train_normal_df = pd.DataFrame(
        train_normal,
        columns=['user','item','rating','timestamp','label']
    )
    train_random_df = pd.DataFrame(
        train_random,
        columns=['user','item','rating','timestamp','label']
    )
    valid_df = pd.DataFrame(
        valid,
        columns=['user','item','rating','timestamp','label']
    )
    test_df = pd.DataFrame(
        test,
        columns=['user','item','rating','timestamp','label']
    )

    os.makedirs(output_dir, exist_ok=True)
    train_normal_df.to_csv(os.path.join(output_dir, 'train_normal.csv'), index=False)
    train_random_df.to_csv(os.path.join(output_dir, 'train_random.csv'), index=False)
    valid_df.to_csv(os.path.join(output_dir, 'valid.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    print("Computing average labels...")
    all_data = pd.concat([train_normal_df, train_random_df, valid_df, test_df], ignore_index=True)
    avg_labels = all_data.groupby(['user', 'item'], sort=False)['label'].mean().reset_index()
    user_item_pairs = avg_labels[['user','item']].to_numpy()
    avg_labels_array = avg_labels['label'].to_numpy()

    np.save(os.path.join(output_dir, 'avg_labels.npy'), avg_labels_array)
    np.save(os.path.join(output_dir, 'user_item_pairs.npy'), user_item_pairs)
    print("Data processing completed.")


# ------------------------------------------------------------
# 2. 简单 DataGenerator：不再做 ID 重映射，直接使用原始 user/item
# ------------------------------------------------------------
class SimpleDataForSCIF:
    def __init__(self, data_path, device):
        """
        data_path: 包含四个 CSV：train_normal.csv, train_random.csv, valid.csv, test.csv
                   其中 user/item ID 均为原始值（0~N-1）。
        device: 'cuda' 或 'cpu'。
        """
        self.device = device
        self.train_normal = pd.read_csv(os.path.join(data_path, "train_normal.csv"))
        self.train_random = pd.read_csv(os.path.join(data_path, "train_random.csv"))
        self.valid = pd.read_csv(os.path.join(data_path, "valid.csv"))
        self.test = pd.read_csv(os.path.join(data_path, "test.csv"))

        # 合并完整训练集
        self.train = pd.concat([self.train_normal, self.train_random], axis=0, ignore_index=True)

        # 计算用户数、物品数（假设 ID 连续、从 0 开始）
        self.n_users = max(
            self.train['user'].max(),
            self.valid['user'].max(),
            self.test['user'].max()
        ) + 1
        self.n_items = max(
            self.train['item'].max(),
            self.valid['item'].max(),
            self.test['item'].max()
        ) + 1

    def get_train_tensor(self):
        arr = self.train[['user','item','label']].values
        return torch.from_numpy(arr).long().to(self.device)

    def get_unlearn_tensor(self):
        arr = self.train_random[['user','item','label']].values
        return torch.from_numpy(arr).long().to(self.device)

    def get_valid_tensor(self):
        arr = self.valid[['user','item','label']].values
        return torch.from_numpy(arr).long().to(self.device)

    def get_test_tensor(self):
        arr = self.test[['user','item','label']].values
        return torch.from_numpy(arr).long().to(self.device)


# ------------------------------------------------------------
# 3. SCIF_Unlearner：在 CPU 上计算 Hessian-vector，避免 GPU OOM
# ------------------------------------------------------------
class SCIF_Unlearner:
    def __init__(self, model, data_generator, device,
                 if_epoch=1000, if_lr=5e-2, reg=0.01):
        """
        model: 已训练好的 Recbole LightGCN 模型，其 user_embedding 和 item_embedding
               的 num_embeddings = data_generator.n_users / n_items。
        data_generator: SimpleDataForSCIF 实例，保持原始 ID 0~N-1 不变。
        device: 'cuda' 或 'cpu'
        if_epoch: SCIF 内部迭代次数
        if_lr:    p 的学习率
        reg:      L2 正则化系数
        """
        self.model = model
        self.data_gen = data_generator
        self.device = device
        self.if_epoch = if_epoch
        self.if_lr = if_lr
        self.reg = reg

        # 1. 提取要遗忘的用户/物品唯一 ID
        un_tensor = self.data_gen.get_unlearn_tensor()
        nei_users = un_tensor[:, 0].unique()
        nei_items = un_tensor[:, 1].unique()

        # 2. 过滤越界，确保 ID < embedding 大小
        max_u = self.model.user_embedding.num_embeddings
        max_i = self.model.item_embedding.num_embeddings
        nei_users = nei_users[(nei_users < max_u) & (nei_users >= 0)]
        nei_items = nei_items[(nei_items < max_i) & (nei_items >= 0)]

        self.nei_users = nei_users.to(device)
        self.nei_items = nei_items.to(device)

        # 3. 保存原始 embedding，用于后面对比
        with torch.no_grad():
            self.original_user_emb = self.model.user_embedding.weight[self.nei_users].cpu().clone()
            self.original_item_emb = self.model.item_embedding.weight[self.nei_items].cpu().clone()

    def _full_loss(self, u_emb, i_emb):
        """
        计算完整训练集上的损失 (BCE_mean + L2)
        u_emb, i_emb: 当前模型的 embedding 张量 (已经位于 GPU 上)
        """
        train_tensor = self.data_gen.get_train_tensor()
        users = train_tensor[:, 0]
        items = train_tensor[:, 1]
        labels = (train_tensor[:, 2] > 0).float()

        u_vectors = u_emb[users]
        i_vectors = i_emb[items]
        logits = torch.sum(u_vectors * i_vectors, dim=-1)
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='mean')
        l2 = self.reg * (torch.norm(u_emb) + torch.norm(i_emb))
        return bce + l2

    def _unlearn_loss(self, u_emb, i_emb):
        """
        仅针对待遗忘子集计算损失 (BCE_sum)
        """
        unlearn_tensor = self.data_gen.get_unlearn_tensor()
        if unlearn_tensor.shape[0] == 0:
            return torch.tensor(0.0, device=self.device)
        users = unlearn_tensor[:, 0]
        items = unlearn_tensor[:, 1]
        labels = (unlearn_tensor[:, 2] > 0).float()

        u_vectors = u_emb[users]
        i_vectors = i_emb[items]
        logits = torch.sum(u_vectors * i_vectors, dim=-1)
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='sum')
        return bce

    def run(self, save_path):
        """
        执行 SCIF Unlearning：  
        1) 把 model 的 embedding pull 到 CPU；  
        2) 构造可求导的 flat-vector un_ui_flat；  
        3) 计算 full_loss 和 unlearn_loss 对 un_ui_flat 的梯度；  
        4) 优化 p，使 H*p ≈ unlearn_grad；  
        5) 用 un_ui_flat + p 更新对应 embedding，并写回 GPU；  
        6) 保存更新后的 state_dict。
        """
        # 1. 将当前模型的 user/item embedding pull 到 CPU
        full_u = self.model.user_embedding.weight.detach().cpu().clone()
        full_i = self.model.item_embedding.weight.detach().cpu().clone()

        # 2. 提取要遗忘的用户/物品嵌入，并设 requires_grad=True
        nei_users_cpu = self.nei_users.cpu()
        nei_items_cpu = self.nei_items.cpu()
        un_u_flat = full_u[nei_users_cpu].reshape(-1)  # [num_nei_u * dim]
        un_i_flat = full_i[nei_items_cpu].reshape(-1)  # [num_nei_i * dim]
        un_ui_flat = torch.cat([un_u_flat, un_i_flat], dim=0).clone().requires_grad_(True)

        # 3. 构造 u_para, i_para，将 un_ui_flat 的切片写回
        u_para = full_u.clone()
        i_para = full_i.clone()
        d_u = un_u_flat.numel()
        u_para[nei_users_cpu] = un_ui_flat[:d_u].reshape(-1, full_u.shape[-1])
        i_para[nei_items_cpu] = un_ui_flat[d_u:].reshape(-1, full_i.shape[-1])

        # 4. 在 GPU 上计算 full_loss 及其对 un_ui_flat 的梯度
        u_para_gpu = u_para.to(self.device)
        i_para_gpu = i_para.to(self.device)
        total_loss = self._full_loss(u_para_gpu, i_para_gpu)
        total_grad = torch.autograd.grad(
            total_loss, un_ui_flat, create_graph=True, retain_graph=True
        )[0].reshape(-1, 1)  # [dim_total, 1]

        # 5. 在 GPU 上计算 unlearn_loss 及其对 un_ui_flat 的梯度
        unlearn_loss = self._unlearn_loss(u_para_gpu, i_para_gpu)
        unlearn_grad = torch.autograd.grad(
            unlearn_loss, un_ui_flat, retain_graph=True
        )[0].reshape(-1, 1)  # [dim_total, 1]

        # 6. 定义 Hessian-vector 乘积 H * v
        def hvp(vec):
            prod = torch.sum(total_grad * vec)  # scalar
            return torch.autograd.grad(prod, un_ui_flat, retain_graph=True)[0]  # [dim_total]

        # 7. grad_goal(p) = H*p - unlearn_grad
        def grad_goal(p_vec):
            h = hvp(p_vec).reshape(-1, 1)  # [dim_total, 1]
            return h - unlearn_grad.detach()  # [dim_total, 1]

        # 8. 初始化 p（在 CPU 上），其 shape = [dim_total, 1]
        self.p = Variable(
            torch.randn_like(un_ui_flat).reshape(-1, 1) * 1e-1, requires_grad=True
        )
        optimizer = torch.optim.Adam([self.p], lr=self.if_lr, weight_decay=0.01)

        # 9. 迭代更新 p
        for epoch in range(self.if_epoch):
            optimizer.zero_grad()
            tg = grad_goal(self.p)  # [dim_total, 1]
            self.p.grad = tg        # 匹配形状
            torch.nn.utils.clip_grad_norm_([self.p], max_norm=1.0)
            optimizer.step()
            if epoch % 100 == 0 or epoch == self.if_epoch - 1:
                print(f">> SCIF epoch {epoch}, mean|p| = {self.p.abs().mean().item():.6e}")

        # 10. 用 un_ui_flat + p 更新 embedding，并写回 GPU
        with torch.no_grad():
            adjusted_flat = un_ui_flat + self.p.squeeze()  # [dim_total]
            new_u = full_u.clone()
            new_i = full_i.clone()
            new_u[nei_users_cpu] = adjusted_flat[:d_u].reshape(-1, full_u.shape[-1])
            new_i[nei_items_cpu] = adjusted_flat[d_u:].reshape(-1, full_i.shape[-1])

            # 将新的 embedding 写回 GPU 上的模型
            self.model.user_embedding.weight.data.copy_(new_u.to(self.device))
            self.model.item_embedding.weight.data.copy_(new_i.to(self.device))

        # 11. 保存更新后的模型 state_dict
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)

        # 12. 打印 embedding 变化幅度（Max Δ）
        with torch.no_grad():
            updated_u = self.model.user_embedding.weight[self.nei_users].cpu()
            updated_i = self.model.item_embedding.weight[self.nei_items].cpu()
            delta_u = torch.max(torch.abs(updated_u - self.original_user_emb)).item()
            delta_i = torch.max(torch.abs(updated_i - self.original_item_emb)).item()
            print(
                f">> Finished SCIF Unlearning. "
                f"Max |Δ user-embed| = {delta_u:.6e}, "
                f"Max |Δ item-embed| = {delta_i:.6e}"
            )


# ------------------------------------------------------------
# 4. 主流程：训练 Recbole LightGCN → SCIF Unlearning → 评估
# ------------------------------------------------------------
def train_and_unlearn(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4.1. 数据拆分：如果 scif_data 下没有 CSV，就调用相应的处理函数
    data_path = "./scif_data"
    if not os.path.exists(os.path.join(data_path, "train_normal.csv")):
        os.makedirs(data_path, exist_ok=True)
        if args.use_netflix:
            print("Using Netflix data")
            # 请将 "path/to/netflix.inter" 替换为实际 Netflix 原始路径
            process_netflix_data("scif/netflix.inter", data_path, forget_ratio=0.1)
        else:
            print("Using ML-100K data")
            # 请将 "path/to/ml-100k.inter" 替换为实际 ML-100K 原始路径
            process_ml100k_data("scif/ml-100k.inter", data_path, forget_ratio=0.1)

    # 4.2. 用 Recbole 训练 LightGCN（全量训练集：train_normal + train_random）
    print("\n=== Training LightGCN on full dataset ===")
    config_full = Config(
        model=args.model,
        dataset="netflix" if args.use_netflix else "ml-100k",
        config_dict={
            "epochs": 50,
            "train_batch_size": 1024,
            "eval_batch_size": 1024,
            "learning_rate": 0.001,
            "topk": [5, 10, 20],
            "metrics": ["Hit", "NDCG"],
        }
    )
    init_seed(config_full["seed"], config_full["reproducibility"])
    init_logger(config_full)
    dataset_full = create_dataset(config_full)
    train_data, valid_data, test_data = data_preparation(config_full, dataset_full)
    init_seed(config_full["seed"] + config_full["local_rank"], config_full["reproducibility"])
    LightGCN_model = get_model(config_full["model"])(config_full, train_data._dataset).to(device)
    trainer_full = get_trainer(config_full["MODEL_TYPE"], config_full["model"])(config_full, LightGCN_model)

    trainer_full.fit(train_data, verbose=True)
    save_full = "./saved/LightGCN_full.pth"
    os.makedirs(os.path.dirname(save_full), exist_ok=True)
    torch.save(LightGCN_model.state_dict(), save_full)

    # 4.3. 预遗忘评估（Full LightGCN）
    print("\n=== Pre-Unlearn Evaluation (Full LightGCN) ===")
    res_full = trainer_full.evaluate(
        test_data, load_best_model=False, show_progress=False, model_file=None
    )
    for k in config_full["topk"]:
        print(f"Hit@{k}: {res_full[f'hit@{k}']:.6f}, NDCG@{k}: {res_full[f'ndcg@{k}']:.6f}")

    # 4.4. 构造 SimpleDataForSCIF 并执行 SCIF Unlearning
    print("\n=== Running SCIF Unlearning (10% subset) ===")
    data_gen = SimpleDataForSCIF(data_path, device)

    # 加载之前训练好的 LightGCN 模型
    scif_model = get_model(config_full["model"])(config_full, train_data._dataset).to(device)
    scif_model.load_state_dict(torch.load(save_full))
    scif_model = scif_model.to(device)

    unlearner = SCIF_Unlearner(
        model=scif_model,
        data_generator=data_gen,
        device=device,
        if_epoch=args.if_epoch,
        if_lr=args.if_lr,
        reg=0.01
    )
    save_unlearned = "./saved/LightGCN_unlearned.pth"
    unlearner.run(save_unlearned)

    # 4.5. 后遗忘评估（LightGCN Unlearned）
    print("\n=== Post-Unlearn Evaluation (LightGCN Unlearned) ===")
    LightGCN_un = get_model(config_full["model"])(config_full, train_data._dataset).to(device)
    LightGCN_un.load_state_dict(torch.load(save_unlearned))
    trainer_un = get_trainer(config_full["MODEL_TYPE"], config_full["model"])(config_full, LightGCN_un)

    res_un = trainer_un.evaluate(
        test_data, load_best_model=False, show_progress=False, model_file=None
    )
    for k in config_full["topk"]:
        print(f"Hit@{k}: {res_un[f'hit@{k}']:.6f}, NDCG@{k}: {res_un[f'ndcg@{k}']:.6f}")



import time
if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   "-m", default="LightGCN", help="Recbole 模型名（LightGCN）")
    parser.add_argument("--if_epoch", type=int, default=1000,
                        help="SCIF 遗忘内部迭代次数")
    parser.add_argument("--if_lr",    type=float, default=5e-2,
                        help="SCIF 遗忘学习率")
    parser.add_argument("--use_netflix", action="store_true", default=True,
                        help="若使用 Netflix 数据集则加此标志；否则默认 ML-100K")
    args = parser.parse_args()

    train_and_unlearn(args)
    end_time = time.time()
    print(f"完全完成代码执行时间: {end_time - start_time:.6f} 秒")