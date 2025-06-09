import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import time
from Model.bpr import BPR
from utility.compute import *
import random
import argparse
import json
import gc
import scipy.sparse as sp


def parse_args():
    parser = argparse.ArgumentParser(description="Influence Function Recommendation Unlearning (IFRU) for BPR - ML-100K")
    
    # Model path
    parser.add_argument('--model_path', type=str, 
                        default='./Weights/BPR/Quick_BPR_ml100k.pth',
                        help='Path to pretrained model')
    
    # IFRU parameters
    parser.add_argument('--if_lr', type=float, default=5e-5, 
                        help='IFRU learning rate')
    parser.add_argument('--if_epoch', type=int, default=2000, 
                        help='Number of IFRU training epochs')
    parser.add_argument('--k_hop', type=int, default=1, 
                        help='Number of neighbor hops')
    parser.add_argument('--if_init_std', type=float, default=1e-8,
                        help='IFRU initialization range')
    parser.add_argument('--damping', type=float, default=1e-4,
                        help='Hessian damping coefficient')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Loss function scaling factor')
    parser.add_argument('--cg_iterations', type=int, default=1000,
                        help='Maximum iterations for conjugate gradient')
    parser.add_argument('--cg_min_iter', type=int, default=50,
                        help='Minimum iterations for conjugate gradient')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding dimension')
    parser.add_argument('--init_std', type=float, default=0.01,
                        help='Weight initialization std')
    
    # Recommendation metrics parameters
    parser.add_argument('--topks', nargs='+', type=int, default=[5, 10, 20],
                        help='Top-k values for evaluation')
    parser.add_argument('--num_neg_test', type=int, default=99,
                        help='Number of negative samples per positive sample during testing')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=1024, 
                        help='Random seed')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size')
    parser.add_argument('--verbose', action='store_true', 
                        help='Whether to output detailed information')
    parser.add_argument('--eval_interval', type=int, default=100,
                        help='Evaluation interval')
    parser.add_argument('--max_train_samples', type=int, default=50000,
                        help='Maximum training samples')
    
    return parser.parse_args()


class ML100KDataGenerator:
    """ML-100K data generator, compatible with Data_for_LightGCN interface"""
    
    def __init__(self, args, path):
        self.args = args
        self.path = path
        
        # Load .inter file
        inter_file = path + '.inter'
        print(f"Loading file: {inter_file}")
        
        # Read data
        data = pd.read_csv(inter_file, sep='\t')
        print(f"Original data shape: {data.shape}")
        print(f"Column names: {list(data.columns)}")
        
        # Process column name mapping
        column_map = {}
        for col in data.columns:
            if 'user' in col.lower():
                column_map[col] = 'user'
            elif 'item' in col.lower():
                column_map[col] = 'item'
            elif 'rating' in col.lower():
                column_map[col] = 'rating'
            elif 'time' in col.lower():
                column_map[col] = 'timestamp'
        
        data.rename(columns=column_map, inplace=True)
        
        # Ensure user IDs and item IDs start from 0
        unique_users = sorted(data['user'].unique())
        unique_items = sorted(data['item'].unique())
        
        user_map = {old_id: new_id for new_id, old_id in enumerate(unique_users)}
        item_map = {old_id: new_id for new_id, old_id in enumerate(unique_items)}
        
        data['user'] = data['user'].map(user_map)
        data['item'] = data['item'].map(item_map)
        
        self.n_users = len(unique_users)
        self.n_items = len(unique_items)
        
        print(f"Users: {self.n_users}, Items: {self.n_items}")
        
        # Create labels - ratings>=4 are positive samples for ML-100K
        if 'rating' in data.columns:
            data['label'] = (data['rating'] >= 4.0).astype(float)
        else:
            data['label'] = 1.0
        
        # Split training and testing sets
        self.split_data(data)
        
        # Create training data in DataFrame format
        self.train = pd.DataFrame({
            'user': self.train_users,
            'item': self.train_items,
            'label': self.train_labels
        })
        
        self.test = pd.DataFrame({
            'user': self.test_users,
            'item': self.test_items,
            'label': self.test_labels
        })
        
        # Create random training subset for forgetting experiments
        # ML-100K is smaller, adjust subset size accordingly
        subset_size = min(len(self.train)//10, 1000)  # 10% or max 1000
        random_indices = np.random.choice(len(self.train), size=subset_size, replace=False)
        self.train_random = self.train.loc[random_indices].reset_index(drop=True)
        
        # Set n_train attribute
        self.n_train = len(self.train)
        
        print(f"Training set size: {len(self.train)}")
        print(f"Test set size: {len(self.test)}")
        print(f"Random training subset size: {len(self.train_random)}")
    
    def split_data(self, data, test_ratio=0.2):
        """Improved data splitting method - ensure each user has sufficient training and testing data"""
        np.random.seed(42)
        
        train_data = []
        test_data = []
        
        # Reserve appropriate proportion of interactions for testing for each user
        for user_id in range(self.n_users):
            user_data = data[data['user'] == user_id]
            if len(user_data) >= 3:  # At least 3 interactions for ML-100K
                # Ensure at least 1 test sample, but not more than 30% of total
                n_test = max(1, min(int(len(user_data) * test_ratio), len(user_data) - 2))
                
                # Sort by timestamp if available, or randomly select
                if 'timestamp' in user_data.columns:
                    user_data = user_data.sort_values('timestamp')
                    # Latest ones for testing
                    test_data.append(user_data.tail(n_test))
                    train_data.append(user_data.head(len(user_data) - n_test))
                else:
                    # Random split
                    test_indices = np.random.choice(len(user_data), n_test, replace=False)
                    test_mask = np.zeros(len(user_data), dtype=bool)
                    test_mask[test_indices] = True
                    
                    test_data.append(user_data[test_mask])
                    train_data.append(user_data[~test_mask])
            else:
                # Too few interactions, all to training set
                train_data.append(user_data)
        
        train_df = pd.concat(train_data, ignore_index=True) if train_data else pd.DataFrame(columns=['user', 'item', 'label'])
        test_df = pd.concat(test_data, ignore_index=True) if test_data else pd.DataFrame(columns=['user', 'item', 'label'])
        
        print(f"Data split result:")
        print(f"  Training set: {len(train_df)} records")
        print(f"  Test set: {len(test_df)} records")
        print(f"  Users with test data: {len(test_df['user'].unique()) if len(test_df) > 0 else 0}")
        
        self.train_users = train_df['user'].values
        self.train_items = train_df['item'].values
        self.train_labels = train_df['label'].values
        
        self.test_users = test_df['user'].values if len(test_df) > 0 else np.array([])
        self.test_items = test_df['item'].values if len(test_df) > 0 else np.array([])
        self.test_labels = test_df['label'].values if len(test_df) > 0 else np.array([])
    
    def set_train_mode(self, mode):
        """Compatible interface"""
        print(f"Setting training mode: {mode}")
        pass


class RecBoleDatasetAdapter:
    """Adapter class, adapt ML100KDataGenerator to RecBole expected format"""
    
    def __init__(self, data_generator, config):
        self.data_generator = data_generator
        self.config = config
        
        # Set basic attributes
        self.n_users = data_generator.n_users
        self.n_items = data_generator.n_items
        
        # Set field mapping
        self.USER_ID = config.USER_ID_FIELD
        self.ITEM_ID = config.ITEM_ID_FIELD
        
        # Create field to number mapping
        self._field_num_map = {
            config.USER_ID_FIELD: self.n_users,
            config.ITEM_ID_FIELD: self.n_items,
            'user_id': self.n_users,
            'item_id': self.n_items,
            'user': self.n_users,
            'item': self.n_items
        }
    
    def num(self, field):
        """Return number of specified field"""
        if field in self._field_num_map:
            return self._field_num_map[field]
        else:
            # Return user count or item count by default
            if 'user' in field.lower():
                return self.n_users
            elif 'item' in field.lower():
                return self.n_items
            else:
                return 1  # Default value
    
    def __getattr__(self, name):
        """Proxy other attributes to original data_generator"""
        return getattr(self.data_generator, name)


def create_recbole_config(embed_size):
    """Create complete RecBole configuration object"""
    config_dict = {
        # Basic fields
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id', 
        'RATING_FIELD': 'rating',
        'TIME_FIELD': 'timestamp',
        'NEG_PREFIX': 'neg_',
        
        # Data fields
        'LABEL_FIELD': 'label',
        'HEAD_ENTITY_ID_FIELD': 'head_id',
        'TAIL_ENTITY_ID_FIELD': 'tail_id',
        'RELATION_ID_FIELD': 'relation_id',
        'ENTITY_ID_FIELD': 'entity_id',
        
        # Model parameters
        'embedding_size': embed_size,
        'train_batch_size': 1024,  # Appropriate for ML-100K
        'eval_batch_size': 1024,   # Appropriate for ML-100K
        'learning_rate': 0.001,    # Standard learning rate
        'weight_decay': 1e-4,
        'train_neg_sample_args': {'distribution': 'uniform', 'sample_num': 1, 'alpha': 1.0, 'dynamic': False, 'candidate_num': 0},
        
        # Training parameters
        'epochs': 100,
        'eval_step': 1,
        'stopping_step': 10,
        'checkpoint_dir': './saved',
        'loss_type': 'BPR',
        'eval_type': 'ranking',
        'metrics': ['Recall', 'NDCG'],
        'topk': [5, 10, 20],
        'valid_metric': 'NDCG@10',
        'metric_decimal_place': 4,
        
        # Device settings  
        'device': 'cuda',
        'use_gpu': True,
        'seed': 2020,
        'reproducibility': True,
        'state': 'INFO',
        
        # Data settings
        'field_separator': '\t',
        'seq_separator': ' ',
        'USER_ID': 'user',
        'ITEM_ID': 'item', 
        'NEG_ITEM_ID': 'neg_item',
        
        # Other necessary fields
        'MODEL_TYPE': 'general',
        'data_path': '/data/IFRU-main/Data/',
        'dataset': 'ml-100k',
        'config_files': [],
        'neg_sampling': None,
        'eval_args': {'split': {'RS': [0.8, 0.1, 0.1]}, 'group_by': 'user', 'order': 'RO', 'mode': 'full'},
        
        # Avoid RecBole automatic data processing
        'load_col': None,
        'unload_col': None,
        'unused_col': None,
        'additional_feat_suffix': None,
    }
    
    class SimpleConfig:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)
        
        def __getitem__(self, key):
            return getattr(self, key, None)
        
        def __contains__(self, key):
            return hasattr(self, key)
        
        def get(self, key, default=None):
            return getattr(self, key, default)
    
    return SimpleConfig(config_dict)


def evaluate_model_leave_one_out(model, data_generator, topks=[5, 10, 20], num_neg=99, 
                               test_on_remain=False, test_on_forget=False, forget_users_set=None):
    """Leave-One-Out evaluation for ML-100K"""
    model.eval()
    with torch.no_grad():
        # Get all users with test data
        test_users = list(set(data_generator.test['user'].values))
        test_users.sort()
        
        print(f"Original test users: {len(test_users)}")
        
        if test_on_remain and forget_users_set:
            test_users = [u for u in test_users if u not in forget_users_set]
            print(f"Evaluating on remain set: {len(test_users)} users")
        elif test_on_forget and forget_users_set:
            test_users = [u for u in test_users if u in forget_users_set]
            print(f"Evaluating on forgotten users set: {len(test_users)} users")
        
        # For ML-100K, we can evaluate all users
        # Initialize metrics
        metrics = {'hit_ratio': np.zeros(len(topks)), 'ndcg': np.zeros(len(topks))}
        
        # Precompute user interactions
        user_to_train_items = {}
        user_to_test_items = {}
        
        for user in test_users:
            train_items = set(data_generator.train[data_generator.train['user'] == user]['item'].values)
            test_items = set(data_generator.test[data_generator.test['user'] == user]['item'].values)
            user_to_train_items[user] = train_items
            user_to_test_items[user] = test_items
        
        item_pool = list(range(data_generator.n_items))
        total_users = 0
        valid_evaluations = 0
        
        # Process each test user
        for user in test_users:
            test_items_for_user = list(user_to_test_items.get(user, []))
            train_items_for_user = user_to_train_items.get(user, set())
            
            if not test_items_for_user:
                continue
            
            # Randomly select one positive sample
            pos_item = random.choice(test_items_for_user)
            
            # Build candidate pool: exclude all known items by user
            known_items = train_items_for_user | user_to_test_items.get(user, set())
            candidate_pool = list(set(item_pool) - known_items)
            
            if len(candidate_pool) < num_neg:
                # If not enough candidates, use all available items
                neg_items = candidate_pool
            else:
                # Randomly select negative samples
                neg_items = random.sample(candidate_pool, num_neg)
            
            if not neg_items:  # Skip if no negative samples
                continue
                
            test_items = [pos_item] + neg_items
            
            # Calculate scores
            user_tensor = torch.tensor([user] * len(test_items), dtype=torch.long).cuda()
            item_tensor = torch.tensor(test_items, dtype=torch.long).cuda()
            
            # Use BPR model to calculate scores
            try:
                if hasattr(model, 'forward'):
                    output = model.forward(user_tensor, item_tensor)
                    if isinstance(output, tuple) and len(output) >= 2:
                        user_emb, item_emb = output[0], output[1]
                    else:
                        user_emb = model.user_embedding(user_tensor)
                        item_emb = model.item_embedding(item_tensor)
                else:
                    user_emb = model.user_embedding(user_tensor)
                    item_emb = model.item_embedding(item_tensor)
                
                scores = torch.sum(user_emb * item_emb, dim=1)
            except Exception as e:
                print(f"Error calculating scores: {e}")
                continue
            
            # Sort and find positive sample rank
            _, indices = torch.sort(scores, descending=True)
            indices = indices.cpu().numpy()
            
            # Position of positive sample in ranking
            rank = np.where(indices == 0)[0]
            if len(rank) == 0:
                continue
                
            rank = rank[0]
            
            # Calculate recommendation metrics
            for k_idx, k in enumerate(topks):
                if rank < k:
                    metrics['hit_ratio'][k_idx] += 1
                    metrics['ndcg'][k_idx] += 1.0 / np.log2(rank + 2)
            
            total_users += 1
            valid_evaluations += 1
        
        print(f"Valid evaluations: {valid_evaluations}/{len(test_users)}")
        
        # Normalize metrics
        if total_users > 0:
            metrics['hit_ratio'] = metrics['hit_ratio'] / total_users
            metrics['ndcg'] = metrics['ndcg'] / total_users
        else:
            print("Warning: No valid evaluation results")
        
        return metrics, total_users


# Calculate AUC metrics function
def get_eval_result_bpr(data_generator, model, mask, test_on_remain=False, test_on_forget=False, forget_users_set=None):
    model.eval()
    with torch.no_grad():
        test_data = data_generator.test[['user','item','label']].values
        
        if test_on_remain and forget_users_set:
            test_data = test_data[~np.isin(test_data[:,0], list(forget_users_set))]
            print(f"Evaluating AUC on remain set: {len(test_data)} test interactions")
        elif test_on_forget and forget_users_set:
            test_data = test_data[np.isin(test_data[:,0], list(forget_users_set))]
            print(f"Evaluating AUC on forgotten users: {len(test_data)} test interactions")
        
        # Filter invalid data
        test_data = test_data[test_data[:,0] < model.user_embedding.weight.shape[0]]
        test_data = test_data[test_data[:,1] < model.item_embedding.weight.shape[0]]
        
        # For ML-100K, we can use all test data
        if len(test_data) == 0:
            print("Warning: No valid test data")
            return 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
        
        test_users = torch.from_numpy(test_data[:,0]).cuda().long()
        test_items = torch.from_numpy(test_data[:,1]).cuda().long()
        
        # Calculate scores using BPR model
        try:
            output = model.forward(test_users, test_items)
            if isinstance(output, tuple) and len(output) == 2:
                user_emb, item_emb = output
                scores = torch.sum(user_emb * item_emb, dim=1)
            else:
                user_emb = model.user_embedding(test_users)
                item_emb = model.item_embedding(test_items)
                scores = torch.sum(user_emb * item_emb, dim=1)
        except:
            user_emb = model.user_embedding(test_users)
            item_emb = model.item_embedding(test_items)
            scores = torch.sum(user_emb * item_emb, dim=1)
        
        test_scores = torch.sigmoid(scores).cpu().numpy()
        test_auc = roc_auc_score(test_data[:,-1], test_scores)
        
        # Get forgotten data
        forget_data = data_generator.train_random[['user','item','label']].values
        forget_data = forget_data[forget_data[:,0] < model.user_embedding.weight.shape[0]]
        forget_data = forget_data[forget_data[:,1] < model.item_embedding.weight.shape[0]]
        
        # Calculate scores on forgotten data
        forget_users = torch.from_numpy(forget_data[:,0]).cuda().long()
        forget_items = torch.from_numpy(forget_data[:,1]).cuda().long()
        
        try:
            forget_user_emb = model.user_embedding(forget_users)
            forget_item_emb = model.item_embedding(forget_items)
            forget_scores = torch.sigmoid(torch.sum(forget_user_emb * forget_item_emb, dim=1)).cpu().numpy()
        except:
            forget_scores = np.zeros(len(forget_data))
        
        # OR logic: test data with original labels + forgotten data with original labels
        or_labels = np.concatenate([test_data[:,-1], forget_data[:,-1]])
        or_scores = np.concatenate([test_scores, forget_scores])
        test_auc_or = roc_auc_score(or_labels, or_scores)
        
        # AND logic: test data with original labels + forgotten data with inverted labels
        and_labels = np.concatenate([test_data[:,-1], 1 - forget_data[:,-1]])
        and_scores = np.concatenate([test_scores, forget_scores])
        test_auc_and = roc_auc_score(and_labels, and_scores)
        
        return test_auc, test_auc_or, test_auc_and, test_auc, test_auc_or, test_auc_and
    
class ReallySlowIFRUBPR(nn.Module):
    """Original really slow IFRU algorithm implementation - BPR version for ML-100K"""
    
    def __init__(self, save_name, if_epoch=2000, if_lr=5e-5, k_hop=1, init_range=1e-8, 
                 damping=1e-4, scale=1.0, cg_iterations=1000, cg_min_iter=50):
        super(ReallySlowIFRUBPR, self).__init__()
        self.if_epoch = if_epoch
        self.if_lr = if_lr
        self.k_hop = k_hop
        self.range = init_range
        self.save_name = save_name
        self.damping = damping
        self.scale = scale
        self.cg_iterations = cg_iterations
        self.cg_min_iter = cg_min_iter
        self.time_records = {
            'preparation': 0, 'hessian_computation': 0, 'influence_function': 0,
            'optimization': 0, 'evaluation': 0, 'total': 0
        }
    
    def compute_neighbor_influence_clip(self, data_generator, k_hop=1):
        """Compute nodes that need updates based on k-hop neighborhood influence"""
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
        neighbor_set = dict(zip(unlearn_ui, unlearn_ui_cunt*1.0/np.maximum(degree_k, 1e-10)))
        neighbor_set_list = [neighbor_set]
        pre_neighbor_set = neighbor_set
        print(f"Nodes to forget: {len(neighbor_set)}")
        
        # Calculate k-hop neighbor influence propagation
        for i in range(k_hop):
            neighbor_set = dict()
            existing_node = list(pre_neighbor_set.keys())
            
            if len(existing_node) == 0:
                break
                
            nonzero_raw, nonzero_col = train_matrix[existing_node].nonzero()
            
            for kk in range(nonzero_raw.shape[0]):
                out_node = existing_node[nonzero_raw[kk]]
                in_node = nonzero_col[kk]
                influence_weight = pre_neighbor_set[out_node] * 1.0 / max(degree[in_node], 1e-10)
                
                if in_node in neighbor_set:
                    neighbor_set[in_node] += influence_weight
                else:
                    neighbor_set[in_node] = influence_weight
            
            pre_neighbor_set = neighbor_set
            neighbor_set_list.append(neighbor_set)
        
        # Use appropriate filtering criteria for ML-100K
        nei_dict = neighbor_set_list[k_hop if k_hop > 0 else 0].copy()
        nei_weights = np.array(list(nei_dict.values()))
        nei_nodes = np.array(list(nei_dict.keys()))
        
        if len(nei_weights) > 0:
            quantile_info = [np.quantile(nei_weights, m*0.1) for m in range(1, 11)]
            # Keep nodes with weights in top 70% (30th percentile)
            select_index = np.where(nei_weights > quantile_info[2])
            neighbor_set = nei_nodes[select_index]
        else:
            neighbor_set = nei_nodes
        
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
        """Original really slow IFRU algorithm - BPR version for ML-100K"""
        print("Starting IFRU algorithm...")
        total_start_time = time.time()
        
        # Calculate nodes that need updates
        nei_users, nei_items, forget_users_set = self.compute_neighbor_influence_clip(
            data_generator, k_hop=self.k_hop
        )
        nei_users = torch.from_numpy(nei_users).cuda().long()
        nei_items = torch.from_numpy(nei_items).cuda().long()
        
        # Evaluate initial model performance
        test_auc, _, _, _, _, _ = get_eval_result_bpr(data_generator, model, None)
        remain_test_auc, _, _, _, _, _ = get_eval_result_bpr(
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
        train_data = train_data[train_data[:,0] < model.user_embedding.weight.shape[0]]
        train_data = train_data[train_data[:,1] < model.item_embedding.weight.shape[0]]
        forget_data = forget_data[forget_data[:,0] < model.user_embedding.weight.shape[0]]
        forget_data = forget_data[forget_data[:,1] < model.item_embedding.weight.shape[0]]
        
        # Sample training data if needed
        if len(train_data) > self.args.max_train_samples:
            train_indices = np.random.choice(len(train_data), self.args.max_train_samples, replace=False)
            train_data = train_data[train_indices]
            print(f"Sampled {len(train_data)} training interactions due to max_train_samples limit")
        
        train_data = torch.from_numpy(train_data).cuda()
        forget_data = torch.from_numpy(forget_data).cuda()
        
        # Ensure valid indices
        valid_users = nei_users[nei_users < model.user_embedding.weight.shape[0]]
        valid_items = nei_items[nei_items < model.item_embedding.weight.shape[0]]
        
        # Create parameter variables - using double precision
        u_params = model.user_embedding.weight[valid_users].clone().detach().to(torch.float64).requires_grad_(True)
        i_params = model.item_embedding.weight[valid_items].clone().detach().to(torch.float64).requires_grad_(True)
        
        # Update parameter count
        u_para_num = u_params.numel()
        i_para_num = i_params.numel()
        
        # Define loss function
        def compute_loss_with_params(u_params, i_params, data_batch, is_training=True):
            """Calculate loss with given parameters"""
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
            
            user_embs = u_params[users]
            item_embs = i_params[items]
            scores = torch.sum(user_embs * item_embs, dim=1)
            labels = torch.clamp(labels, 0.0, 1.0)
            
            loss = F.binary_cross_entropy_with_logits(scores, labels, reduction='mean')
            return loss
        
        # Batch gradient computation function - smaller batch size for ML-100K
        def compute_batch_gradients(u_params, i_params, data_batch, batch_size=256):
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
        grad_u, grad_i = compute_batch_gradients(u_params, i_params, train_data, batch_size=256)
        grad_params = torch.cat([grad_u.reshape(-1), grad_i.reshape(-1)])
        total_grad_norm = grad_params.norm().item()
        
        # Calculate forgotten data loss gradient
        forget_grad_u, forget_grad_i = compute_batch_gradients(u_params, i_params, forget_data, batch_size=256)
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
            
            batch_size = 64  # Smaller batch size for ML-100K
            num_batches = (len(train_data) + batch_size - 1) // batch_size
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(train_data))
                batch = train_data[start_idx:end_idx]
                
                if len(batch) == 0:
                    continue
                
                u_params_batch = u_params.clone().detach().requires_grad_(True)
                i_params_batch = i_params.clone().detach().requires_grad_(True)
                
                loss = compute_loss_with_params(u_params_batch, i_params_batch, batch, is_training=True)
                
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
                    if change_ratio < 1e-5 and i >= min_iter:
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
                # For ML-100K we can use a slightly larger update scale
                update_scale = step_size / max(data_generator.n_train, 2000)
                
                # Update model parameters
                model.user_embedding.weight.data[valid_users] += update_scale * influence_u
                model.item_embedding.weight.data[valid_items] += update_scale * influence_i
                
                # Regularize
                max_norm = 1.0
                user_norms = torch.norm(model.user_embedding.weight.data[valid_users], dim=1, keepdim=True)
                item_norms = torch.norm(model.item_embedding.weight.data[valid_items], dim=1, keepdim=True)
                
                # Apply norm clipping
                model.user_embedding.weight.data[valid_users] = torch.where(
                    user_norms > max_norm,
                    model.user_embedding.weight.data[valid_users] * (max_norm / user_norms),
                    model.user_embedding.weight.data[valid_users]
                )
                
                model.item_embedding.weight.data[valid_items] = torch.where(
                    item_norms > max_norm,
                    model.item_embedding.weight.data[valid_items] * (max_norm / item_norms),
                    model.item_embedding.weight.data[valid_items]
                )
            
            # Periodic evaluation
            if (epoch + 1) % self.args.eval_interval == 0:
                eval_start = time.time()
                
                _, _, valid_auc_and, _, _, _ = get_eval_result_bpr(data_generator, model, None)
                
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
    
    print("="*80)
    print("Influence Function Recommendation Unlearning (IFRU) for BPR - ML-100K")
    print(f"Model: {args.model_path}")
    print(f"Parameters: if_lr={args.if_lr}, k_hop={args.k_hop}, if_epoch={args.if_epoch}")
    print(f"Hessian damping: {args.damping}, Loss scaling: {args.scale}")
    print(f"CG max/min iterations: {args.cg_iterations}/{args.cg_min_iter}")
    print("="*80)
    
    # Create data loader parameters
    model_args = type('Args', (), {
        'embed_size': args.embed_size,
        'batch_size': args.batch_size,
        'data_path': '/data/IFRU-main/Data/',
        'dataset': 'ml-100k',
        'attack': '',
        'data_type': 'full',
        'A_split': False,
        'A_n_fold': 100,
        'keep_prob': 1.0,
        'gcn_layers': 2,
        'dropout': False,
        'pretrain': 0,
        'init_std': args.init_std
    })
    
    # Load data
    print("\nLoading data...")
    data_path = '/data/IFRU-main/Data/ml-100k'
    data_generator = ML100KDataGenerator(model_args, path=data_path)
    data_generator.set_train_mode(model_args.data_type)
    print(f"Data loaded: {data_generator.n_users} users, {data_generator.n_items} items")
    
    # Create BPR model
    config = create_recbole_config(args.embed_size)
    adapted_dataset = RecBoleDatasetAdapter(data_generator, config)
    
    try:
        model = BPR(config, adapted_dataset).cuda()
        
        # Manually set user and item counts
        if not hasattr(model, 'n_users'):
            model.n_users = data_generator.n_users
        if not hasattr(model, 'n_items'):
            model.n_items = data_generator.n_items
        
        print("BPR model created successfully")
        
    except Exception as e:
        print(f"BPR model creation failed: {e}")
        return
    
    # Load pretrained model
    try:
        model.load_state_dict(torch.load(args.model_path))
        print(f"Pretrained model loaded successfully: {args.model_path}")
    except Exception as e:
        print(f"Failed to load pretrained model: {e}")
        print("Using randomly initialized parameters...")
        nn.init.normal_(model.user_embedding.weight, mean=0, std=0.01)
        nn.init.normal_(model.item_embedding.weight, mean=0, std=0.01)
    
    # Create save directory
    save_dir = './Weights/IFRU'
    os.makedirs(save_dir, exist_ok=True)
    save_name = f"{save_dir}/ifru_bpr_ml100k_lr{args.if_lr}_damping{args.damping}.pth"
    
    # Create and run IFRU algorithm
    ifru = ReallySlowIFRUBPR(
        save_name=save_name,
        if_epoch=args.if_epoch,
        if_lr=args.if_lr,
        k_hop=args.k_hop,
        init_range=args.if_init_std,
        damping=args.damping,
        scale=args.scale,
        cg_iterations=args.cg_iterations,
        cg_min_iter=args.cg_min_iter
    )
    
    ifru.args = args
    
    # Use IFRU algorithm
    print("\nRunning IFRU algorithm...")
    best_epoch, best_score, forget_users_set, before_metrics, before_remain_metrics, before_forget_metrics = ifru.compute_hessian_with_test(
        model=model, data_generator=data_generator
    )
    
    # Load best model and evaluate
    if best_epoch > 0:
        model.load_state_dict(torch.load(save_name))
        print(f"Loaded best model from epoch {best_epoch}")
    else:
        print("Warning: No best model saved")
    
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
    test_auc, test_auc_or, test_auc_and, _, _, _ = get_eval_result_bpr(data_generator, model, None)
    remain_test_auc, remain_test_auc_or, remain_test_auc_and, _, _, _ = get_eval_result_bpr(
        data_generator, model, None, test_on_remain=True, forget_users_set=forget_users_set
    )
    
    # Forgotten users AUC evaluation
    forget_test_auc, forget_test_auc_or, forget_test_auc_and, _, _, _ = get_eval_result_bpr(
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
    
    # Add forgotten users recommendation metrics comparison
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
        'model': 'IFRU_BPR_ML100K',
        'model_path': args.model_path,
        'if_lr': args.if_lr,
        'k_hop': args.k_hop,
        'if_epoch': args.if_epoch,
        'damping': args.damping,
        'scale': args.scale,
        'cg_iterations': args.cg_iterations,
        'cg_min_iter': args.cg_min_iter,
        'best_epoch': best_epoch,
        'final_test_auc': float(test_auc),
        'remain_test_auc': float(remain_test_auc),
        'forget_test_auc': float(forget_test_auc),
        'test_auc_and': float(test_auc_and),
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
    results_path = f'{result_dir}/ifru_bpr_ml100k_lr{args.if_lr}_damping{args.damping}_results.json'
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