import torch
from torch_geometric.utils import degree
from model.MF import *
from util.LightGCN import ndcg_at_k
from time import time


# LightGCN
def sisa_MF_eva(shard_models, config: dict, data, device='cpu'):
    num_users = config['num_users']
    num_books = config['num_books']
    k = config['k']
    epochs = config['epochs']
    batch_size = config['batch_size']
    num_shards = config['num_shards']
    test_topks = []
    epoch_tracks = []
    data = data.to(device)

    # Double-check the data and divide it into different shards
    mask = data.edge_index[0] < data.edge_index[1]
    train_edge_index = data.edge_index[:, mask]
    indices = torch.randperm(train_edge_index.size(1))  # Shuffle the data
    train_edge_index = train_edge_index[:, indices]  # Apply the shuffled indices
    shard_size = train_edge_index.size(1) // num_shards
    shards = [torch.arange(i * shard_size, (i + 1) * shard_size) for i in range(num_shards)]
    optimizers = [torch.optim.Adam(shard_model.parameters(), lr=config["lr"]) for shard_model in shard_models]
    loss_func = torch.nn.MSELoss().to(device)

    for epoch in range(epochs):
        total_loss = 0
        for shard_idx, shard_indices in enumerate(shards):
            shard_model = shard_models[shard_idx]
            optimizer = optimizers[shard_idx]
            shard_model.train()
            shard_loader = torch.utils.data.DataLoader(shard_indices, batch_size=batch_size, shuffle=True)
            for index in shard_loader:
                edge_batch = train_edge_index[:, index].t()
                src, dst = edge_batch[:, 0], edge_batch[:, 1]
                dst = dst - config['num_users']
                optimizer.zero_grad()
                pred = shard_model(src, dst)

                target = torch.ones_like(pred)
                loss = loss_func(pred, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(target)
        train_loss = total_loss / train_edge_index.size(1)

        # Evaluation
        with torch.no_grad():
            # Aggregate embeddings from all shards
            user_emb = torch.zeros(num_users, config['embedding_dim'], device=device)
            book_emb = torch.zeros(num_books, config['embedding_dim'], device=device)
            for shard_model in shard_models:
                shard_model.eval()
                with torch.no_grad():
                    shard_user_emb, shard_item_emb = shard_model.get_embedding(prompt=None)
                    user_emb += shard_user_emb
                    book_emb += shard_item_emb
            user_emb /= num_shards
            book_emb /= num_shards

            precision = recall = ndcg = total_examples = 0
            for start in range(0, num_users, batch_size):
                epoch_tracks.append(epoch)
                end = start + batch_size
                logits = user_emb[start:end] @ book_emb.t()  # User ratings matrix
                mask = ((train_edge_index[0] >= start) & (train_edge_index[0] < end))
                logits[train_edge_index[0, mask] - start, train_edge_index[1, mask] - num_users] = float(
                    '-inf')

                ground_truth = torch.zeros_like(logits, dtype=torch.bool)
                mask = ((data.edge_label_index[0] >= start) & (data.edge_label_index[0] < end))
                ground_truth[data.edge_label_index[0, mask] - start, data.edge_label_index[1, mask] - num_users] = True
                node_count = degree(data.edge_label_index[0, mask] - start, num_nodes=logits.size(0))
                topk_index = logits.topk(k, dim=-1).indices
                isin_mat = ground_truth.gather(1, topk_index)
                precision += float((isin_mat.sum(dim=-1) / k).sum())
                recall += float((isin_mat.sum(dim=-1) / node_count.clamp(1e-6)).sum())
                for i in range(isin_mat.shape[0]):
                    if node_count[i] > 0:
                        ndcg += ndcg_at_k(isin_mat[i].cpu().numpy(), k)
                total_examples += int((node_count > 0).sum())
            precision /= total_examples
            recall /= total_examples
            ndcg /= total_examples
            test_topks.append((precision, recall, ndcg))
            print(f'Epoch: {epoch+1:03d}, Loss: {train_loss:.4f}, HR@{k}: '
                  f'{precision:.4f}, Recall@{k}: {recall:.4f}, NDCG@{k}: {ndcg:.4f}')
    return shard_models, shards, epoch_tracks, test_topks


def sisa_MF_unlearning_eva(shard_models, shards, retain_data, forget_data, config, device='cpu'):
    num_users = config['num_users']
    num_books = config['num_books']
    k = config['k']
    epochs = config['forget_epochs']
    batch_size = config['batch_size']
    num_shards = config['num_shards']
    embedding_dim = config['embedding_dim']
    mask = retain_data.edge_index[0] < retain_data.edge_index[1]
    train_edge_index = retain_data.edge_index[:, mask]
    # retain_data.to(device)
    forget_data = forget_data.to(device)
    loss_func = torch.nn.MSELoss().to(device)

    forget_set = set(tuple(edge) for edge in forget_data.edge_index.t().tolist())
    updated_shards = []
    t1 = time()
    for shard_idx, shard_indices in enumerate(shards):
        shard_edges = train_edge_index[:, shard_indices].t().tolist()
        forget_in_shard = any(tuple(edge) in forget_set for edge in shard_edges)
        if forget_in_shard:
            print(f"Forgetting data found in Shard {shard_idx + 1}, retraining...")

            new_shard_indices = torch.tensor([i for i, edge in zip(shard_indices, shard_edges)
                                              if tuple(edge) not in forget_set], device=device)
            updated_shards.append(new_shard_indices)

            shard_model = MF(
                num_users=config['num_users'],
                num_items=config['num_books'],
                mean=config['global_bias'],
                embedding_dim=config['embedding_dim']
            ).to(device)
            shard_models[shard_idx] = shard_model
            optimizer = torch.optim.Adam(shard_model.parameters(), lr=config["lr"])
            for epoch in range(epoch):
                shard_model.train()
                total_loss = total_examples = 0
                shard_loader = torch.utils.data.DataLoader(new_shard_indices, batch_size=batch_size, shuffle=True)

                for index in shard_loader:
                    edge_batch = train_edge_index[:, index].t()
                    src, dst = edge_batch[:, 0], edge_batch[:, 1]
                    dst = dst - config['num_users']

                    optimizer.zero_grad()
                    pred = shard_model(src, dst)

                    target = torch.ones_like(pred)
                    loss = loss_func(pred, target)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item() * len(target)
                train_loss = total_loss / train_edge_index.size(1)
                print(f'Shard {shard_idx + 1}, Epoch: {epoch + 1:03d}, Loss: {train_loss:.4f}')
        else:
            updated_shards.append(shard_indices)

    with torch.no_grad():
        user_emb = torch.zeros(num_users, embedding_dim, device=device)
        book_emb = torch.zeros(num_books, embedding_dim, device=device)
        for shard_model in shard_models:
            shard_model.eval()
            emb = shard_model.get_embedding(prompt=None)
            user_emb += emb[:num_users]
            book_emb += emb[num_users:]
        user_emb /= num_shards
        book_emb /= num_shards

        precision = recall = total_examples = 0
        for start in range(0, num_users, batch_size):
            end = start + batch_size
            logits = user_emb[start:end] @ book_emb.t()
            # Exclude training edges:
            mask = ((retain_data.edge_index[0] >= start) &
                    (retain_data.edge_index[0] < end))
            logits[retain_data.edge_index[0, mask] - start,
                   retain_data.edge_index[1, mask] - num_users] = float('-inf')
            # Computing precision and recall:
            ground_truth = torch.zeros_like(logits, dtype=torch.bool)
            mask = ((retain_data.edge_label_index[0] >= start) &
                    (retain_data.edge_label_index[0] < end))
            ground_truth[retain_data.edge_label_index[0, mask] - start,
                         retain_data.edge_label_index[1, mask] - num_users] = True
            node_count = degree(retain_data.edge_label_index[0, mask] - start,
                                num_nodes=logits.size(0))
            topk_index = logits.topk(k, dim=-1).indices
            isin_mat = ground_truth.gather(1, topk_index)
            precision += float((isin_mat.sum(dim=-1) / k).sum())
            recall += float((isin_mat.sum(dim=-1) / node_count.clamp(1e-6)).sum())
            total_examples += int((node_count > 0).sum())
        precision = precision / total_examples
        recall = recall / total_examples
        print(f'HR@{k}: {precision:.4f}, Recall@{k}: {recall:.4f}')
    t2 = time()
    print(f"Running time: {t2 - t1:.4f}s")
    return shard_models


def sisa_MF_forget_data_eva(shard_models, forget_data, config, device='cpu'):
    shard_models.eval()
    num_users = config['num_users']
    num_books = config['num_books']
    num_shards = config['num_shards']
    embedding_dim = config['embedding_dim']
    k = config['k']
    batch_size = config['batch_size']
    forget_data = forget_data.to(device)

    with torch.no_grad():
        user_emb = torch.zeros(num_users, embedding_dim, device=device)
        book_emb = torch.zeros(num_books, embedding_dim, device=device)
        for shard_model in shard_models:
            shard_model.eval()
            emb = shard_model.get_embedding(forget_data.edge_index, prompt=None)
            user_emb += emb[:num_users]
            book_emb += emb[num_users:]
        user_emb /= num_shards
        book_emb /= num_shards

        precision = recall = total_examples = 0
        for start in range(0, num_users, batch_size):
            end = start + batch_size
            logits = user_emb[start:end] @ book_emb.t()

            mask = ((forget_data.edge_index[0] >= start) &
                    (forget_data.edge_index[0] < end))
            forget_edges = forget_data.edge_index[:, mask]
            node_count = degree(forget_data.edge_index[0, mask] - start,
                                num_nodes=logits.size(0))
            topk_index = logits.topk(k, dim=-1).indices

            ground_truth = torch.zeros_like(logits, dtype=torch.bool)
            ground_truth[forget_edges[0] - start, forget_edges[1] - num_users] = True

            isin_mat = ground_truth.gather(1, topk_index)

            precision += float((isin_mat.sum(dim=-1) / k).sum())
            recall += float((isin_mat.sum(dim=-1) / node_count.clamp(1e-6)).sum())
            total_examples += int((node_count > 0).sum())
        precision = precision / total_examples
        recall = recall / total_examples
        print(f'HR@{k}: {precision:.4f}, Recall@{k}: {recall:.4f}')