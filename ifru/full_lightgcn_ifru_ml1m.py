import os
import numpy as np
import torch
import torch.nn as nn
from utility.load_data import * 
import scipy.sparse as sp
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import time
from Model.Lightgcn import LightGCN
from utility.compute import *
import random
import argparse
import json
import gc


def parse_args():
    parser = argparse.ArgumentParser(description="IFRU Algorithm Implementation for LightGCN")
    # Model path
    parser.add_argument('--model_path', type=str, 
                        default='./Weights/LightGCN/Quick_LightGCN_ml1m.pth',
                        help='Path to the pretrained model')
    # IFRU parameters
    parser.add_argument('--if_lr', type=float, default=5e-5, help='IFRU learning rate')
    parser.add_argument('--if_epoch', type=int, default=2000, help='IFRU training epochs')
    parser.add_argument('--k_hop', type=int, default=1, help='Number of neighbor hops')
    parser.add_argument('--if_init_std', type=float, default=1e-8, help='IFRU initialization range')
    parser.add_argument('--damping', type=float, default=1e-4, help='Hessian damping coefficient')
    parser.add_argument('--cg_iterations', type=int, default=1000, help='Maximum iterations for conjugate gradient')
    parser.add_argument('--cg_min_iter', type=int, default=50, help='Minimum iterations for conjugate gradient')
    # Model parameters
    parser.add_argument('--embed_size', type=int, default=32, help='Embedding dimension')
    parser.add_argument('--gcn_layers', type=int, default=2, help='Number of GCN layers')
    parser.add_argument('--init_std', type=float, default=0.01, help='Weight initialization std')
    # Recommendation metrics parameters
    parser.add_argument('--topks', nargs='+', type=int, default=[5, 10, 20], help='Top-k values for evaluation')
    parser.add_argument('--num_neg_test', type=int, default=99, help='Number of negative samples per positive sample during testing')
    # Other parameters
    parser.add_argument('--seed', type=int, default=1024, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--eval_interval', type=int, default=100, help='Evaluation interval')
    parser.add_argument('--max_train_samples', type=int, default=50000, help='Maximum training samples')
    # Optimization method selection
    parser.add_argument('--opt_method', type=str, default='slow_ifru', 
                        choices=['slow_ifru', 'direct_ifru'],
                        help='Optimization method: slow_ifru (conjugate gradient) or direct_ifru (direct gradient)')
    
    # AUC evaluation target
    parser.add_argument('--auc_target', type=str, default='and_max',
                        choices=['and_max', 'mid_point'],
                        help='AUC optimization target: and_max (maximize AND AUC) or mid_point (make AUC close to 0.5)')
    return parser.parse_args()


# Standard Leave-One-Out evaluation
def evaluate_model_leave_one_out(model, data_generator, topks=[5, 10, 20], num_neg=99, 
                               test_on_remain=False, test_on_forget=False, forget_users_set=None):
    """
    Standard Leave-One-Out evaluation strategy
    """
    model.eval()
    with torch.no_grad():
        # Get test users and filter them if needed
        test_users = list(set(data_generator.test['user'].values))
        test_users.sort()
        
        if test_on_remain and forget_users_set:
            test_users = [u for u in test_users if u not in forget_users_set]
            print(f"Evaluating on remain set: {len(test_users)} users")
        elif test_on_forget and forget_users_set:
            test_users = [u for u in test_users if u in forget_users_set]
            print(f"Evaluating on forgotten users set: {len(test_users)} users")
        
        if len(test_users) == 0:
            print("Warning: No users available for evaluation")
            return {'hit_ratio': np.zeros(len(topks)), 'ndcg': np.zeros(len(topks))}, 0
        
        u_batch_size = 128  # Larger batch size for ML-1M
        n_test_users = len(test_users)
        n_user_batchs = (n_test_users + u_batch_size - 1) // u_batch_size
        
        # Initialize metrics
        metrics = {'hit_ratio': np.zeros(len(topks)), 'ndcg': np.zeros(len(topks))}
        
        # Compute embeddings
        try:
            all_users, all_items = model.computer()
        except:
            all_users = model.embedding_user.weight
            all_items = model.embedding_item.weight
        
        # Precompute user interactions
        user_to_pos_items = {}
        for user in test_users:
            train_items = set(data_generator.train[data_generator.train['user'] == user]['item'].values)
            test_items = set(data_generator.test[data_generator.test['user'] == user]['item'].values)
            user_to_pos_items[user] = train_items | test_items
        
        item_pool = list(range(data_generator.n_items))
        total_users = 0
        
        # Process users batch by batch
        for u_batch_id in range(n_user_batchs):
            start = u_batch_id * u_batch_size
            end = min((u_batch_id + 1) * u_batch_size, n_test_users)
            batch_test_users = test_users[start:end]
            batch_users_tensor = torch.tensor(batch_test_users, dtype=torch.long).cuda()
            u_embeddings = all_users[batch_users_tensor]
            
            for user_idx, user in enumerate(batch_test_users):
                test_items_for_user = list(data_generator.test[data_generator.test['user'] == user]['item'].values)
                if not test_items_for_user:
                    continue
                
                pos_item = random.choice(test_items_for_user)
                neg_items = random.sample(list(set(item_pool) - user_to_pos_items.get(user, set())), num_neg)
                test_items = [pos_item] + neg_items
                
                item_tensor = torch.tensor(test_items, dtype=torch.long).cuda()
                i_embeddings = all_items[item_tensor]
                
                scores = torch.matmul(u_embeddings[user_idx:user_idx+1], i_embeddings.t()).squeeze()
                _, indices = torch.sort(scores, descending=True)
                indices = indices.cpu().numpy()
                
                # Calculate recommendation metrics
                rank = np.where(indices == 0)[0][0]
                for k_idx, k in enumerate(topks):
                    if rank < k:
                        metrics['hit_ratio'][k_idx] += 1
                        metrics['ndcg'][k_idx] += 1.0 / np.log2(rank + 2)
                
                total_users += 1
        
        # Normalize metrics
        if total_users > 0:
            metrics['hit_ratio'] = metrics['hit_ratio'] / total_users
            metrics['ndcg'] = metrics['ndcg'] / total_users
        
        return metrics, total_users


# Calculate AUC metrics - with support for separate evaluation on forgotten users
def get_eval_result_lightgcn(data_generator, model, mask, test_on_remain=False, test_on_forget=False, forget_users_set=None):
    model.eval()
    with torch.no_grad():
        # Get test data
        test_data = data_generator.test[['user','item','label']].values
        
        if test_on_remain and forget_users_set:
            test_data = test_data[~np.isin(test_data[:,0], list(forget_users_set))]
            print(f"Evaluating AUC on remain set: {len(test_data)} test interactions")
        elif test_on_forget and forget_users_set:
            test_data = test_data[np.isin(test_data[:,0], list(forget_users_set))]
            print(f"Evaluating AUC on forgotten users: {len(test_data)} test interactions")
        
        # Filter invalid data
        test_data = test_data[test_data[:,0] < model.embedding_user.weight.shape[0]]
        test_data = test_data[test_data[:,1] < model.embedding_item.weight.shape[0]]
        
        if len(test_data) == 0:
            print("Warning: No valid test data")
            return 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
        
        # Get forgotten data
        forget_data = data_generator.train_random[['user','item','label']].values
        forget_data = forget_data[forget_data[:,0] < model.embedding_user.weight.shape[0]]
        forget_data = forget_data[forget_data[:,1] < model.embedding_item.weight.shape[0]]
        
        # Calculate test set prediction scores
        test_users = torch.from_numpy(test_data[:,0]).cuda().long()
        test_items = torch.from_numpy(test_data[:,1]).cuda().long()
        
        # Get embeddings
        try:
            all_users, all_items = model.computer()
            user_emb = all_users[test_users]
            item_emb = all_items[test_items]
        except:
            user_emb = model.embedding_user(test_users)
            item_emb = model.embedding_item(test_items)
        
        test_scores = torch.sigmoid(torch.sum(user_emb * item_emb, dim=1)).cpu().numpy()
        test_auc = roc_auc_score(test_data[:,-1], test_scores)
        
        # Calculate scores on forgotten data
        forget_users = torch.from_numpy(forget_data[:,0]).cuda().long()
        forget_items = torch.from_numpy(forget_data[:,1]).cuda().long()
        
        try:
            forget_user_emb = all_users[forget_users]
            forget_item_emb = all_items[forget_items]
        except:
            forget_user_emb = model.embedding_user(forget_users)
            forget_item_emb = model.embedding_item(forget_items)
        
        forget_scores = torch.sigmoid(torch.sum(forget_user_emb * forget_item_emb, dim=1)).cpu().numpy()
        
        # OR logic: union of test data and forgotten data
        or_labels = np.concatenate([test_data[:,-1], forget_data[:,-1]])
        or_scores = np.concatenate([test_scores, forget_scores])
        test_auc_or = roc_auc_score(or_labels, or_scores)
        
        # AND logic: test data with original labels + forgotten data with inverted labels
        # This reflects the goal that "model predictions should be inverted for forgotten data"
        and_labels = np.concatenate([test_data[:,-1], 1 - forget_data[:,-1]])
        and_scores = np.concatenate([test_scores, forget_scores])
        test_auc_and = roc_auc_score(and_labels, and_scores)
        
        return test_auc, test_auc_or, test_auc_and, test_auc, test_auc_or, test_auc_and


class InfluenceFunctionRecommendationUnlearning(nn.Module):
    """
    Influence Function based Recommendation Unlearning (IFRU) implementation
    """
    
    def __init__(self, save_name, if_epoch=2000, if_lr=5e-5, k_hop=1, init_range=1e-8, 
                 damping=1e-4, cg_iterations=1000, cg_min_iter=50):
        super(InfluenceFunctionRecommendationUnlearning, self).__init__()
        self.if_epoch = if_epoch
        self.if_lr = if_lr
        self.k_hop = k_hop
        self.range = init_range
        self.save_name = save_name
        self.damping = damping
        self.cg_iterations = cg_iterations
        self.cg_min_iter = cg_min_iter
        self.time_records = {
            'preparation': 0, 'hessian_computation': 0, 'influence_function': 0,
            'optimization': 0, 'evaluation': 0, 'total': 0
        }
    
    def create_changed_graph(self, data_generator, forget_users_set):
        """
        Create a graph structure with forgotten data removed
        """
        print("Creating graph with forgotten interactions removed...")
        
        # Filter out interactions that need to be forgotten
        train_data = data_generator.train.copy()
        forget_mask = ~train_data['user'].isin(forget_users_set)
        remain_train_data = train_data[forget_mask]
        
        # Build new graph following original processing logic
        u_id = remain_train_data['user'].values
        i_id = remain_train_data['item'].values + data_generator.n_users
        
        n_nodes = data_generator.n_users + data_generator.n_items
        adj_mat = sp.dok_matrix((n_nodes, n_nodes), dtype=np.float32)
        
        # Fill interaction data
        adj_mat[u_id, i_id] = 1.0
        adj_mat[i_id, u_id] = 1.0
        
        # Use the same normalization method as LightGCN
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        
        norm_adj = d_mat.dot(adj_mat).dot(d_mat)

        # Convert to COO format for PyTorch sparse tensor
        coo = norm_adj.tocoo()
        
        # Create indices and values
        indices = torch.LongTensor(np.vstack((coo.row, coo.col))).cuda()
        values = torch.FloatTensor(coo.data).cuda()
        shape = torch.Size(norm_adj.shape)
        
        # Create PyTorch sparse tensor
        sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
        
        return sparse_tensor

    def compute_neighbor_influence_clip(self, data_generator, k_hop=1):
        """
        Compute nodes that need updates based on k-hop neighborhood influence
        """
        print(f"Computing {k_hop}-hop neighbor influence...")
        start_time = time.time()
        
        # Build graph matrix
        train_data = data_generator.train.values.copy()
        matrix_size = data_generator.n_users + data_generator.n_items
        train_data[:,1] += data_generator.n_users
        train_data[:,-1] = np.ones_like(train_data[:,-1])
        
        # Build bidirectional graph with self-loops
        train_data2 = np.ones_like(train_data)
        train_data2[:,0], train_data2[:,1] = train_data[:,1], train_data[:,0]
        padding = np.concatenate([
            np.arange(matrix_size).reshape(-1,1), 
            np.arange(matrix_size).reshape(-1,1), 
            np.ones(matrix_size).reshape(-1,1)
        ], axis=-1)
        
        # Merge all edges
        data = np.concatenate([train_data, train_data2, padding], axis=0).astype(int)
        train_matrix = sp.csc_matrix(
            (data[:,-1], (data[:,0], data[:,1])), 
            shape=(matrix_size, matrix_size)
        )
        
        # Calculate node degrees
        degree = np.array(train_matrix.sum(axis=-1)).squeeze()
        
        # Get users and items to forget
        unlearn_user = data_generator.train_random['user'].values.reshape(-1)
        unlearn_user, cunt_u = np.unique(unlearn_user, return_counts=True)
        unlearn_item = data_generator.train_random['item'].values.reshape(-1) + data_generator.n_users
        unlearn_item, cunt_i = np.unique(unlearn_item, return_counts=True)
        
        # Combine nodes to forget
        unlearn_ui = np.concatenate([unlearn_user, unlearn_item], axis=-1)
        unlearn_ui_cunt = np.concatenate([cunt_u, cunt_i], axis=-1)
        degree_k = degree[unlearn_ui]
        
        # Initialize influence set
        neighbor_set = dict(zip(unlearn_ui, unlearn_ui_cunt*1.0/degree_k))
        neighbor_set_list = [neighbor_set]
        pre_neighbor_set = neighbor_set
        print(f"Number of nodes to forget: {len(neighbor_set)}")
        
        # Calculate k-hop neighbor influence propagation
        for i in range(k_hop):
            neighbor_set = dict()
            existing_node = list(pre_neighbor_set.keys())
            nonzero_raw, nonzero_col = train_matrix[existing_node].nonzero()
            
            for kk in range(nonzero_raw.shape[0]):
                out_node = existing_node[nonzero_raw[kk]]
                in_node = nonzero_col[kk]
                try:
                    neighbor_set[in_node] += pre_neighbor_set[out_node] * 1.0 / degree[in_node]
                except:
                    neighbor_set[in_node] = pre_neighbor_set[out_node] * 1.0 / degree[in_node]
            
            pre_neighbor_set = neighbor_set
            neighbor_set_list.append(neighbor_set)
        
        # Use relaxed filtering criteria
        nei_dict = neighbor_set_list[k_hop if k_hop > 0 else 0].copy()
        nei_weights = np.array(list(nei_dict.values()))
        nei_nodes = np.array(list(nei_dict.keys()))
        quantile_info = [np.quantile(nei_weights, m*0.1) for m in range(1, 11)]
        
        # Keep nodes with weights in the top 70%
        select_index = np.where(nei_weights > quantile_info[2])  # 30th percentile
        neighbor_set = nei_nodes[select_index]
        
        # Merge nodes
        all_nei_ui = np.concatenate([unlearn_ui.squeeze(), neighbor_set.squeeze()])
        all_nei_ui = np.unique(all_nei_ui)
        print(f"Total affected nodes: {all_nei_ui.shape}")
        
        self.time_records['preparation'] = time.time() - start_time
        
        # Return user and item IDs along with forgotten users set
        forget_users_set = set(unlearn_user.squeeze())
        return all_nei_ui[np.where(all_nei_ui < data_generator.n_users)], \
               all_nei_ui[np.where(all_nei_ui >= data_generator.n_users)] - data_generator.n_users, \
               forget_users_set
    
    def compute_hessian_with_test(self, model=None, data_generator=None):
        """
        Main IFRU algorithm implementation
        """
        print("Starting IFRU algorithm...")
        total_start_time = time.time()
        
        # Calculate nodes that need updates
        nei_users, nei_items, forget_users_set = self.compute_neighbor_influence_clip(
            data_generator, k_hop=self.k_hop
        )
        nei_users = torch.from_numpy(nei_users).cuda().long()
        nei_items = torch.from_numpy(nei_items).cuda().long()
        
        # Evaluate initial model performance
        test_auc, _, _, _, _, _ = get_eval_result_lightgcn(data_generator, model, None)
        remain_test_auc, _, _, _, _, _ = get_eval_result_lightgcn(
            data_generator, model, None, test_on_remain=True, forget_users_set=forget_users_set
        )
        print(f"Pre-unlearning results: test AUC:{test_auc:.6f}, remain test AUC:{remain_test_auc:.6f}")
        
        # Calculate initial recommendation metrics
        print("\nCalculating pre-unlearning recommendation metrics...")
        before_metrics, _ = evaluate_model_leave_one_out(
            model, data_generator, topks=self.args.topks, num_neg=self.args.num_neg_test
        )
        
        # Calculate initial remain set recommendation metrics
        print("\nCalculating pre-unlearning recommendation metrics (remain set)...")
        before_remain_metrics, _ = evaluate_model_leave_one_out(
            model, data_generator, topks=self.args.topks, num_neg=self.args.num_neg_test,
            test_on_remain=True, forget_users_set=forget_users_set
        )
        
        # Calculate initial forgotten users recommendation metrics
        print("\nCalculating pre-unlearning recommendation metrics (forgotten users)...")
        before_forget_metrics, before_forget_users = evaluate_model_leave_one_out(
            model, data_generator, topks=self.args.topks, num_neg=self.args.num_neg_test,
            test_on_forget=True, forget_users_set=forget_users_set
        )
        
        # Use complete training data
        train_data = data_generator.train[['user','item','label']].values
        forget_data = data_generator.train_random[['user','item','label']].values
        
        # Ensure indices are valid
        train_data = train_data[train_data[:,0] < model.embedding_user.weight.shape[0]]
        train_data = train_data[train_data[:,1] < model.embedding_item.weight.shape[0]]
        forget_data = forget_data[forget_data[:,0] < model.embedding_user.weight.shape[0]]
        forget_data = forget_data[forget_data[:,1] < model.embedding_item.weight.shape[0]]
        
        # Sample training data if needed
        if len(train_data) > self.args.max_train_samples:
            train_indices = np.random.choice(len(train_data), self.args.max_train_samples, replace=False)
            train_data = train_data[train_indices]
        
        train_data = torch.from_numpy(train_data).cuda()
        forget_data = torch.from_numpy(forget_data).cuda()
        
        # Ensure valid indices
        valid_users = nei_users[nei_users < model.embedding_user.weight.shape[0]]
        valid_items = nei_items[nei_items < model.embedding_item.weight.shape[0]]
        
        # Create parameter variables - using double precision
        u_params = model.embedding_user.weight[valid_users].clone().detach().to(torch.float64).requires_grad_(True)
        i_params = model.embedding_item.weight[valid_items].clone().detach().to(torch.float64).requires_grad_(True)
        
        # Update parameter count
        u_para_num = u_params.numel()
        i_para_num = i_params.numel()

        original_graph = model.Graph
        changed_graph = self.create_changed_graph(data_generator, forget_users_set)

        # Define loss function
        def compute_loss_with_params(u_params, i_params, data_batch, is_training=True, graph=None):
            """
            Calculate loss with given parameters, optionally considering graph structure
            """
            users = data_batch[:, 0].long()
            items = data_batch[:, 1].long() 
            labels = data_batch[:, 2].to(torch.float64)
            
            valid_mask = (users >= 0) & (users < u_params.shape[0]) & \
                    (items >= 0) & (items < i_params.shape[0])
            
            if not valid_mask.all():
                users = users[valid_mask]
                items = items[valid_mask]
                labels = labels[valid_mask]
            
            if len(users) == 0:
                return torch.tensor(0.0, dtype=torch.float64, requires_grad=True).cuda()
            
            if graph is not None and hasattr(model, 'F_computer'):
                # Calculate embeddings using graph structure
                temp_user_embs = torch.zeros((model.num_users, u_params.shape[1]), dtype=torch.float32).cuda()
                temp_item_embs = torch.zeros((model.num_items, i_params.shape[1]), dtype=torch.float32).cuda()
                
                # Fill current parameters
                temp_user_embs[valid_users] = u_params.to(torch.float32)
                temp_item_embs[valid_items] = i_params.to(torch.float32)
                
                # Use graph structure to compute propagation
                all_user_embs, all_item_embs = model.F_computer(temp_user_embs, temp_item_embs, graph)
                user_embs = all_user_embs[users]
                item_embs = all_item_embs[items]
            else:
                # Use embeddings directly
                user_embs = u_params[users]
                item_embs = i_params[items]
            
            scores = torch.sum(user_embs * item_embs, dim=1)
            labels = torch.clamp(labels, 0.0, 1.0)
            
            loss = F.binary_cross_entropy_with_logits(scores, labels, reduction='mean')
            return loss
        
        # Batch gradient computation function
        def compute_batch_gradients(u_params, i_params, data_batch, batch_size=200):  # Larger batch size for ML-1M
            total_grad_u = torch.zeros_like(u_params)
            total_grad_i = torch.zeros_like(i_params)
            
            num_batches = (len(data_batch) + batch_size - 1) // batch_size
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(data_batch))
                batch = data_batch[start_idx:end_idx]
                
                if len(batch) == 0:
                    continue
                
                u_params_batch = u_params.clone().detach().requires_grad_(True)
                i_params_batch = i_params.clone().detach().requires_grad_(True)
                
                loss = compute_loss_with_params(u_params_batch, i_params_batch, batch, is_training=True)
                
                if loss.item() > 0:
                    grad_u, grad_i = torch.autograd.grad(
                        loss, [u_params_batch, i_params_batch], retain_graph=False
                    )
                    total_grad_u += grad_u.detach()
                    total_grad_i += grad_i.detach()
                
                del u_params_batch, i_params_batch, loss
                if 'grad_u' in locals():
                    del grad_u, grad_i
                torch.cuda.empty_cache()
            
            if num_batches > 0:
                total_grad_u /= num_batches
                total_grad_i /= num_batches
            
            return total_grad_u, total_grad_i
        
        # Step 1: Calculate gradients
        print("Computing training and forgotten data gradients...")
        hessian_start = time.time()
        
        # Calculate total training loss gradient
        grad_u, grad_i = compute_batch_gradients(u_params, i_params, train_data, batch_size=200)
        grad_params = torch.cat([grad_u.reshape(-1), grad_i.reshape(-1)])
        total_grad_norm = grad_params.norm().item()
        
        # Calculate forgotten data loss gradient
        forget_grad_u, forget_grad_i = compute_batch_gradients(u_params, i_params, forget_data, batch_size=200)
        forget_grad_params = torch.cat([forget_grad_u.reshape(-1), forget_grad_i.reshape(-1)])
        forget_grad_norm = forget_grad_params.norm().item()
        
        # Normalize gradient vectors
        if total_grad_norm > 0 and forget_grad_norm > 0:
            grad_scale = min(1.0, 1.0 / total_grad_norm)
            forget_grad_scale = min(1.0, 1.0 / forget_grad_norm)
            
            grad_params *= grad_scale
            forget_grad_params *= forget_grad_scale
            
            grad_u = grad_params[:u_para_num].reshape(u_params.shape)
            grad_i = grad_params[u_para_num:].reshape(i_params.shape)
            forget_grad_u = forget_grad_params[:u_para_num].reshape(u_params.shape)
            forget_grad_i = forget_grad_params[u_para_num:].reshape(i_params.shape)
        
        # Define precise Hessian-vector product function
        def hvp_fn(v):
            v = v.detach()
            v_u = v[:u_para_num].reshape(u_params.shape)
            v_i = v[u_para_num:].reshape(i_params.shape)
            
            total_hvp_u = torch.zeros_like(u_params)
            total_hvp_i = torch.zeros_like(i_params)
            
            batch_size = 2048  # Larger batch size for ML-1M
            num_batches = (len(train_data) + batch_size - 1) // batch_size
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(train_data))
                batch = train_data[start_idx:end_idx]
                
                if len(batch) == 0:
                    continue
                
                u_params_batch = u_params.clone().detach().requires_grad_(True)
                i_params_batch = i_params.clone().detach().requires_grad_(True)
                
                # Calculate loss on original graph
                loss_orig = compute_loss_with_params(u_params_batch, i_params_batch, batch, 
                                                   is_training=True, graph=original_graph)
                
                # Calculate loss on modified graph
                loss_changed = compute_loss_with_params(u_params_batch, i_params_batch, batch, 
                                                       is_training=True, graph=changed_graph)
                
                # Combine losses according to official definition
                loss = loss_orig - 0.5 * loss_changed
                
                if loss.item() > 0:
                    grad_u_batch, grad_i_batch = torch.autograd.grad(
                        loss, [u_params_batch, i_params_batch], create_graph=True, retain_graph=True
                    )
                    
                    # Calculate gradient-vector product
                    grad_v = torch.sum(grad_u_batch * v_u) + torch.sum(grad_i_batch * v_i)
                    
                    # Calculate second derivatives
                    hvp_u_batch, hvp_i_batch = torch.autograd.grad(
                        grad_v, [u_params_batch, i_params_batch], retain_graph=False
                    )
                    
                    total_hvp_u += hvp_u_batch.detach()
                    total_hvp_i += hvp_i_batch.detach()
                
                # Clean memory
                del u_params_batch, i_params_batch, loss
                if 'grad_u_batch' in locals():
                    del grad_u_batch, grad_i_batch, grad_v, hvp_u_batch, hvp_i_batch
                torch.cuda.empty_cache()
            
            # Average HVP
            if num_batches > 0:
                total_hvp_u /= num_batches
                total_hvp_i /= num_batches
            
            hvp = torch.cat([total_hvp_u.reshape(-1), total_hvp_i.reshape(-1)])
            
            # Ensure appropriate HVP scale
            hvp_norm = hvp.norm().item()
            if hvp_norm < 1e-10:
                hvp = v.clone()  # Use identity transform
            elif hvp_norm > 1e3:
                hvp = hvp * (1.0 / hvp_norm)  # Normalize
            
            return hvp
        
        self.time_records['hessian_computation'] = time.time() - hessian_start
        
        # Use stable conjugate gradient method
        print("Solving influence function using conjugate gradient method...")
        influence_start = time.time()
        
        def stable_cg_solve(Ax_fn, b, max_iter=1000, min_iter=50, tol=1e-8):
            print(f"Starting conjugate gradient solution, max iterations: {max_iter}, min iterations: {min_iter}")
            x = torch.zeros_like(b)
            r = b.clone()
            p = r.clone()
            rsold = torch.sum(r * r)
            
            if rsold < 1e-20:
                return x
            
            residual_history = []
            
            for i in range(max_iter):
                # Calculate Hessian-vector product
                Ap = Ax_fn(p)
                
                # Add damping term
                if self.damping > 0:
                    Ap = Ap + self.damping * p
                
                pAp = torch.sum(p * Ap)
                pAp_item = pAp.item()
                
                # Handle near-zero pAp
                if abs(pAp_item) < 1e-10:
                    if i < min_iter:
                        pAp = torch.tensor(1e-10, dtype=torch.float64).cuda() * torch.sign(pAp)
                    else:
                        break
                
                alpha = rsold / pAp
                alpha_item = alpha.item()
                
                if abs(alpha_item) > 1e5:
                    alpha = torch.tensor(1e5, dtype=torch.float64).cuda() * torch.sign(alpha)
                
                x = x + alpha * p
                r = r - alpha * Ap
                rsnew = torch.sum(r * r)
                residual = torch.sqrt(rsnew).item()
                residual_history.append(residual)
                
                # Check residual change
                if i > 0 and len(residual_history) > 1:
                    change_ratio = abs(residual_history[-2] - residual) / max(residual_history[-2], 1e-10)
                    if change_ratio < 1e-7 and i >= min_iter:
                        break
                
                if residual < tol and i >= min_iter:
                    break
                
                beta = rsnew / rsold
                p = r + beta * p
                rsold = rsnew
                
                if (i+1) % 10 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
            
            # Limit influence vector norm
            x_norm = x.norm().item()
            if x_norm > 5.0:
                x = x * (5.0 / x_norm)
            
            return x
        
        # Solve influence function
        influence_vector = stable_cg_solve(
            hvp_fn, -forget_grad_params, 
            max_iter=self.cg_iterations,
            min_iter=self.cg_min_iter
        )
        
        self.time_records['influence_function'] = time.time() - influence_start
        
        # Apply influence function updates
        print("Applying influence function updates...")
        optimization_start = time.time()
        
        # Track best model
        best_auc = float('inf')
        best_epoch = 0
        not_change = 0
        
        # Split influence vector back to user and item parameters
        influence_u = influence_vector[:u_para_num].reshape(u_params.shape).to(torch.float32)
        influence_i = influence_vector[u_para_num:].reshape(i_params.shape).to(torch.float32)
        
        influence_norm = influence_vector.norm().item()
        print(f"Influence vector norm: {influence_norm:.6f}")
        
        # Iteratively apply influence function updates
        for epoch in range(self.if_epoch):
            step_size = self.if_lr * (0.999 ** (epoch // 100))
            
            with torch.no_grad():
                update_scale = step_size * 0.1
                
                # Update model parameters
                model.embedding_user.weight.data[valid_users] += update_scale * influence_u
                model.embedding_item.weight.data[valid_items] += update_scale * influence_i
                
                # Regularize
                max_norm = 2.0
                user_norms = torch.norm(model.embedding_user.weight.data[valid_users], dim=1, keepdim=True)
                item_norms = torch.norm(model.embedding_item.weight.data[valid_items], dim=1, keepdim=True)
                
                model.embedding_user.weight.data[valid_users] = torch.where(
                    user_norms > max_norm,
                    model.embedding_user.weight.data[valid_users] * (max_norm / user_norms),
                    model.embedding_user.weight.data[valid_users]
                )
                
                model.embedding_item.weight.data[valid_items] = torch.where(
                    item_norms > max_norm,
                    model.embedding_item.weight.data[valid_items] * (max_norm / item_norms),
                    model.embedding_item.weight.data[valid_items]
                )
                
                # Handle graph propagation effect
                if hasattr(model, 'computer'):
                    # Trigger graph propagation update
                    _ = model.computer()
            
            # Periodic evaluation
            if (epoch + 1) % self.args.eval_interval == 0:
                eval_start = time.time()
                
                _, _, valid_auc_and, _, _, _ = get_eval_result_lightgcn(data_generator, model, None)
                
                forget_score = abs(valid_auc_and - 0.5)
                
                print(f"Epoch {epoch+1}/{self.if_epoch} | AND AUC: {valid_auc_and:.4f} | Unlearning score: {forget_score:.4f}")
                
                # Save best model
                if forget_score < best_auc:
                    best_auc = forget_score
                    best_epoch = epoch + 1
                    print(f"  -> Saved best model (score: {forget_score:.4f})")
                    torch.save(model.state_dict(), self.save_name)
                    not_change = 0
                else:
                    not_change += 1
                
                self.time_records['evaluation'] += time.time() - eval_start
                
                # Early stopping criteria
                if not_change > 30 or forget_score < 0.01:
                    break
                
                # Memory cleanup
                if (epoch + 1) % 100 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
        
        self.time_records['optimization'] = time.time() - optimization_start
        self.time_records['total'] = time.time() - total_start_time
        
        print(f"\nIFRU algorithm completed! Total time: {self.time_records['total']:.2f} seconds")
        
        return best_epoch, best_auc, forget_users_set, before_metrics, before_remain_metrics, before_forget_metrics


def main():
    args = parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Create data loader parameters
    model_args = type('Args', (), {
        'embed_size': args.embed_size,
        'batch_size': args.batch_size,
        'data_path': 'Data/Process/',
        'dataset': 'ml-1m',  # Using ML-1M dataset
        'attack': '0.1',
        'data_type': 'full',
        'A_split': False,
        'A_n_fold': 100,
        'keep_prob': 1.0,
        'gcn_layers': args.gcn_layers,
        'dropout': False,
        'pretrain': 0,
        'init_std': args.init_std
    })
    
    # Load data
    print("\nLoading data...")
    data_generator = Data_for_LightGCN(model_args, path=model_args.data_path + model_args.dataset + '/' + model_args.attack)
    data_generator.set_train_mode(model_args.data_type)
    
    # Create LightGCN model
    model = LightGCN(model_args, data_generator).cuda()
    
    # Load pretrained model
    try:
        model.load_state_dict(torch.load(args.model_path))
        print("Successfully loaded pretrained model")
    except Exception as e:
        print(f"Failed to load pretrained model: {e}")
        nn.init.normal_(model.embedding_user.weight, mean=0, std=0.01)
        nn.init.normal_(model.embedding_item.weight, mean=0, std=0.01)
    
    # Create save directory
    save_dir = './Weights/IFRU'
    os.makedirs(save_dir, exist_ok=True)
    save_name = f"{save_dir}/ifru_lightgcn_1m_lr{args.if_lr}_damping{args.damping}.pth"
    
    # Create and run IFRU algorithm
    ifru = InfluenceFunctionRecommendationUnlearning(
        save_name=save_name,
        if_epoch=args.if_epoch,
        if_lr=args.if_lr,
        k_hop=args.k_hop,
        init_range=args.if_init_std,
        damping=args.damping,
        cg_iterations=args.cg_iterations,
        cg_min_iter=args.cg_min_iter
    )
    
    ifru.args = args

    # Use IFRU with conjugate gradient
    print("\nUsing IFRU with conjugate gradient method...")
    best_epoch, best_score, forget_users_set, before_metrics, before_remain_metrics, before_forget_metrics = ifru.compute_hessian_with_test(
        model=model, data_generator=data_generator
    )

    # Load best model and evaluate
    if best_epoch > 0:
        model.load_state_dict(torch.load(save_name))
    
    print("\n==== Final Comprehensive Evaluation ====")
    
    # Calculate post-unlearning recommendation metrics (all test set)
    print("\nCalculating post-unlearning recommendation metrics (all test set)...")
    after_metrics, _ = evaluate_model_leave_one_out(
        model, data_generator, topks=args.topks, num_neg=args.num_neg_test
    )
    
    # Calculate post-unlearning recommendation metrics (remain set)
    print("\nCalculating post-unlearning recommendation metrics (remain set)...")
    after_remain_metrics, _ = evaluate_model_leave_one_out(
        model, data_generator, topks=args.topks, num_neg=args.num_neg_test,
        test_on_remain=True, forget_users_set=forget_users_set
    )
    
    # Calculate post-unlearning recommendation metrics (forgotten users)
    print("\nCalculating post-unlearning recommendation metrics (forgotten users)...")
    after_forget_metrics, after_forget_users = evaluate_model_leave_one_out(
        model, data_generator, topks=args.topks, num_neg=args.num_neg_test,
        test_on_forget=True, forget_users_set=forget_users_set
    )
    
    # Final AUC evaluation
    test_auc, test_auc_or, test_auc_and, _, _, _ = get_eval_result_lightgcn(data_generator, model, None)
    remain_test_auc, remain_test_auc_or, remain_test_auc_and, _, _, _ = get_eval_result_lightgcn(
        data_generator, model, None, test_on_remain=True, forget_users_set=forget_users_set
    )
    
    # Forgotten users AUC evaluation
    forget_test_auc, forget_test_auc_or, forget_test_auc_and, _, _, _ = get_eval_result_lightgcn(
        data_generator, model, None, test_on_forget=True, forget_users_set=forget_users_set
    )

    # Report detailed AUC metrics
    print(f"\n==== Detailed AUC Metrics ====")
    print(f"All test set: Standard AUC={test_auc:.6f}, OR AUC={test_auc_or:.6f}, AND AUC={test_auc_and:.6f}")
    print(f"Remain set: Standard AUC={remain_test_auc:.6f}, OR AUC={remain_test_auc_or:.6f}, AND AUC={remain_test_auc_and:.6f}")
    print(f"Forgotten users: Standard AUC={forget_test_auc:.6f}, OR AUC={forget_test_auc_or:.6f}, AND AUC={forget_test_auc_and:.6f}")

    # Output results
    print(f"\nTest AUC: {test_auc:.6f}")
    print(f"Remain set Test AUC: {remain_test_auc:.6f}")
    print(f"Forgotten users Test AUC: {forget_test_auc:.6f}")
    
    print(f"\n==== Recommendation Metrics Comparison (All Test Set) ====")
    print("Before unlearning:")
    for k_idx, k in enumerate(args.topks):
        print(f"HR@{k}: {before_metrics['hit_ratio'][k_idx]:.6f}")
        print(f"NDCG@{k}: {before_metrics['ndcg'][k_idx]:.6f}")
    
    print("\nAfter unlearning:")
    for k_idx, k in enumerate(args.topks):
        print(f"HR@{k}: {after_metrics['hit_ratio'][k_idx]:.6f}")
        print(f"NDCG@{k}: {after_metrics['ndcg'][k_idx]:.6f}")
    
    print(f"\n==== Recommendation Metrics Comparison (Remain Set) ====")
    print("Before unlearning:")
    for k_idx, k in enumerate(args.topks):
        print(f"HR@{k}: {before_remain_metrics['hit_ratio'][k_idx]:.6f}")
        print(f"NDCG@{k}: {before_remain_metrics['ndcg'][k_idx]:.6f}")
    
    print("\nAfter unlearning:")
    for k_idx, k in enumerate(args.topks):
        print(f"HR@{k}: {after_remain_metrics['hit_ratio'][k_idx]:.6f}")
        print(f"NDCG@{k}: {after_remain_metrics['ndcg'][k_idx]:.6f}")
    
    # Add: Forgotten users recommendation metrics comparison
    print(f"\n==== Recommendation Metrics Comparison (Forgotten Users) ====")
    print("Before unlearning:")
    for k_idx, k in enumerate(args.topks):
        print(f"HR@{k}: {before_forget_metrics['hit_ratio'][k_idx]:.6f}")
        print(f"NDCG@{k}: {before_forget_metrics['ndcg'][k_idx]:.6f}")
    
    print("\nAfter unlearning:")
    for k_idx, k in enumerate(args.topks):
        print(f"HR@{k}: {after_forget_metrics['hit_ratio'][k_idx]:.6f}")
        print(f"NDCG@{k}: {after_forget_metrics['ndcg'][k_idx]:.6f}")
    
    # Save results
    results = {
        'model': 'IFRU_LightGCN_1M',  # Changed to reflect ML-1M
        'model_path': args.model_path,
        'if_lr': args.if_lr,
        'k_hop': args.k_hop,
        'if_epoch': args.if_epoch,
        'damping': args.damping,
        'cg_iterations': args.cg_iterations,
        'cg_min_iter': args.cg_min_iter,
        'best_epoch': best_epoch,
        'final_test_auc': float(test_auc),
        'remain_test_auc': float(remain_test_auc),
        'forget_test_auc': float(forget_test_auc),
        'best_forget_score': float(best_score),
        'time_records': ifru.time_records,
        'metrics_all': {
            'before': {
                'hit_ratio': {str(k): float(before_metrics['hit_ratio'][i]) for i, k in enumerate(args.topks)},
                'ndcg': {str(k): float(before_metrics['ndcg'][i]) for i, k in enumerate(args.topks)}
            },
            'after': {
                'hit_ratio': {str(k): float(after_metrics['hit_ratio'][i]) for i, k in enumerate(args.topks)},
                'ndcg': {str(k): float(after_metrics['ndcg'][i]) for i, k in enumerate(args.topks)}
            }
        },
        'metrics_remain': {
            'before': {
                'hit_ratio': {str(k): float(before_remain_metrics['hit_ratio'][i]) for i, k in enumerate(args.topks)},
                'ndcg': {str(k): float(before_remain_metrics['ndcg'][i]) for i, k in enumerate(args.topks)}
            },
            'after': {
                'hit_ratio': {str(k): float(after_remain_metrics['hit_ratio'][i]) for i, k in enumerate(args.topks)},
                'ndcg': {str(k): float(after_remain_metrics['ndcg'][i]) for i, k in enumerate(args.topks)}
            }
        },
        # Add: Forgotten users metrics
        'metrics_forget': {
            'before': {
                'hit_ratio': {str(k): float(before_forget_metrics['hit_ratio'][i]) for i, k in enumerate(args.topks)},
                'ndcg': {str(k): float(before_forget_metrics['ndcg'][i]) for i, k in enumerate(args.topks)}
            },
            'after': {
                'hit_ratio': {str(k): float(after_forget_metrics['hit_ratio'][i]) for i, k in enumerate(args.topks)},
                'ndcg': {str(k): float(after_forget_metrics['ndcg'][i]) for i, k in enumerate(args.topks)}
            }
        }
    }
    
    result_dir = './results'
    os.makedirs(result_dir, exist_ok=True)
    results_path = f'{result_dir}/ifru_lightgcn_1m_lr{args.if_lr}_damping{args.damping}_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    # Time statistics
    print("\n==== Time Statistics ====")
    print(f"Preparation time: {ifru.time_records['preparation']:.2f}s")
    print(f"Hessian computation: {ifru.time_records['hessian_computation']:.2f}s")
    print(f"Influence function: {ifru.time_records['influence_function']:.2f}s")
    print(f"Optimization: {ifru.time_records['optimization']:.2f}s")
    print(f"Evaluation: {ifru.time_records['evaluation']:.2f}s")
    print(f"Total time: {ifru.time_records['total']:.2f}s")


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    main()