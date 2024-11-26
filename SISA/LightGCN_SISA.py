import torch
from time import time
from torch_geometric.utils import degree
from model.LightGCN import *
from util.LightGCN import recommendation_loss, ndcg_at_k


# LightGCN
def sisa_lightgcn_eva(shard_models, config: dict, data, device='cpu'):
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

    for epoch in range(epochs):
        total_loss = total_examples = 0

        for shard_idx, shard_indices in enumerate(shards):
            shard_model = shard_models[shard_idx]
            optimizer = optimizers[shard_idx]
            shard_model.train()
            shard_loader = torch.utils.data.DataLoader(shard_indices, batch_size=batch_size, shuffle=True)
            for index in shard_loader:
                pos_edge_label_index = train_edge_index[:, index]
                neg_edge_label_index = torch.stack([
                    pos_edge_label_index[0],
                    torch.randint(num_users, num_users + num_books, (index.numel(),), device=device)
                ], dim=0)
                edge_label_index = torch.cat([pos_edge_label_index, neg_edge_label_index], dim=1)
                optimizer.zero_grad()
                pos_rank, neg_rank = shard_model(data.edge_index, edge_label_index).chunk(2)

                loss = recommendation_loss(
                    shard_model,
                    None,
                    pos_rank,
                    neg_rank,
                    node_id=edge_label_index.unique(),
                )
                loss.backward()
                optimizer.step()
                total_loss += float(loss) * pos_rank.numel()
                total_examples += pos_rank.numel()
        loss = total_loss / total_examples

        # Evaluation
        with torch.no_grad():
            # Aggregate embeddings from all shards
            user_emb = torch.zeros(num_users, config['embedding_dim'], device=device)
            book_emb = torch.zeros(num_books, config['embedding_dim'], device=device)
            for shard_model in shard_models:
                shard_model.eval()
                with torch.no_grad():
                    emb = shard_model.get_embedding(data.edge_index)
                    user_emb += emb[:num_users]
                    book_emb += emb[num_users:]
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
            print(f'Epoch: {epoch + 1:03d}, Loss: {loss:.4f}, HR@{k}: '
                  f'{precision:.4f}, Recall@{k}: {recall:.4f}, NDCG@{k}: {ndcg:.4f}')
    return shard_models, shards, epoch_tracks, test_topks


def sisa_lightgcn_unlearning_eva(shard_models, shards, retain_data, forget_data, config, device='cpu'):
    num_users = config['num_users']
    num_books = config['num_books']
    k = config['k']
    epochs = config['forget_epochs']
    batch_size = config['batch_size']
    num_shards = config['num_shards']
    embedding_dim = config['embedding_dim']
    mask = retain_data.edge_index[0] < retain_data.edge_index[1]
    train_edge_label_index = retain_data.edge_index[:, mask]
    data = retain_data.to(device)
    forget_data = forget_data.to(device)

    forget_set = set(tuple(edge) for edge in forget_data.edge_index.t().tolist())
    updated_shards = []
    t1 = time()
    for shard_idx, shard_indices in enumerate(shards):
        shard_edges = train_edge_label_index[:, shard_indices].t().tolist()
        forget_in_shard = any(tuple(edge) in forget_set for edge in shard_edges)
        if forget_in_shard:
            print(f"Forgetting data found in Shard {shard_idx + 1}, retraining...")

            new_shard_indices = torch.tensor([i for i, edge in zip(shard_indices, shard_edges)
                                              if tuple(edge) not in forget_set], device=device)
            updated_shards.append(new_shard_indices)

            shard_model = LightGCN(
                num_nodes=data.num_nodes,
                embedding_dim=config['embedding_dim'],
                num_layers=config['num_layers'],
            ).to(device)
            shard_models[shard_idx] = shard_model
            optimizer = torch.optim.Adam(shard_model.parameters(), lr=config["lr"])
            for epoch in range(epochs):
                shard_model.train()
                total_loss = total_examples = 0
                shard_loader = torch.utils.data.DataLoader(new_shard_indices, batch_size=batch_size, shuffle=True)

                for index in shard_loader:
                    pos_edge_label_index = train_edge_label_index[:, index]
                    neg_edge_label_index = torch.stack([
                        pos_edge_label_index[0],
                        torch.randint(num_users, num_users + num_books, (index.numel(),), device=device)
                    ], dim=0)
                    edge_label_index = torch.cat([pos_edge_label_index, neg_edge_label_index], dim=1)

                    optimizer.zero_grad()
                    pos_rank, neg_rank = shard_model(data.edge_index, edge_label_index).chunk(2)

                    loss = recommendation_loss(
                        shard_model,
                        None,
                        pos_rank,
                        neg_rank,
                        node_id=edge_label_index.unique(),
                    )
                    loss.backward()
                    optimizer.step()

                    total_loss += float(loss) * pos_rank.numel()
                    total_examples += pos_rank.numel()

                loss = total_loss / total_examples
                print(f'Shard {shard_idx + 1}, Epoch: {epoch + 1:03d}, Loss: {loss:.4f}')
        else:
            updated_shards.append(shard_indices)

    with torch.no_grad():
        user_emb = torch.zeros(num_users, embedding_dim, device=device)
        book_emb = torch.zeros(num_books, embedding_dim, device=device)
        for shard_model in shard_models:
            shard_model.eval()
            emb = shard_model.get_embedding(data.edge_index)
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


def sisa_lightgcn_forget_data_eva(shard_models, forget_data, config, device='cpu'):
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