import os
import sys
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
import time
import random
from sklearn.metrics import roc_auc_score

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from Model.bpr import BPR
from netflix_data_generator import NetflixDataGenerator


class RecBoleDatasetAdapter:
    """适配器类，将NetflixDataGenerator适配为RecBole期望的数据集格式"""
    
    def __init__(self, data_generator, config):
        self.data_generator = data_generator
        self.config = config
        
        # 设置基本属性
        self.n_users = data_generator.n_users
        self.n_items = data_generator.n_items
        
        # 设置字段映射
        self.USER_ID = config.USER_ID_FIELD
        self.ITEM_ID = config.ITEM_ID_FIELD
        
        # 创建字段到数量的映射
        self._field_num_map = {
            config.USER_ID_FIELD: self.n_users,
            config.ITEM_ID_FIELD: self.n_items,
            'user_id': self.n_users,
            'item_id': self.n_items,
            'user': self.n_users,
            'item': self.n_items
        }
    
    def num(self, field):
        """返回指定字段的数量"""
        if field in self._field_num_map:
            return self._field_num_map[field]
        else:
            # 默认返回用户数量或物品数量
            if 'user' in field.lower():
                return self.n_users
            elif 'item' in field.lower():
                return self.n_items
            else:
                return 1  # 默认值
    
    def __getattr__(self, name):
        """代理其他属性到原始data_generator"""
        return getattr(self.data_generator, name)


def create_recbole_config(embed_size):
    """创建完整的RecBole配置对象"""
    config_dict = {
        # 基本字段
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id', 
        'RATING_FIELD': 'rating',
        'TIME_FIELD': 'timestamp',
        'NEG_PREFIX': 'neg_',
        
        # 数据字段
        'LABEL_FIELD': 'label',
        'HEAD_ENTITY_ID_FIELD': 'head_id',
        'TAIL_ENTITY_ID_FIELD': 'tail_id',
        'RELATION_ID_FIELD': 'relation_id',
        'ENTITY_ID_FIELD': 'entity_id',
        
        # 模型参数
        'embedding_size': embed_size,
        'train_batch_size': 2048,  # Netflix数据更大，增加批次
        'eval_batch_size': 2048,
        'learning_rate': 0.01,
        'weight_decay': 1e-4,
        'train_neg_sample_args': {'distribution': 'uniform', 'sample_num': 1, 'alpha': 1.0, 'dynamic': False, 'candidate_num': 0},
        
        # 训练参数
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
        
        # 设备设置  
        'device': 'cuda',
        'use_gpu': True,
        'seed': 2020,
        'reproducibility': True,
        'state': 'INFO',
        
        # 数据设置
        'field_separator': '\t',
        'seq_separator': ' ',
        'USER_ID': 'user',
        'ITEM_ID': 'item', 
        'NEG_ITEM_ID': 'neg_item'
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


class QuickBPRTrainer:
    def __init__(self, data_generator):
        # 设置BPR模型参数 - 针对Netflix调整
        self.args = type('Args', (), {
            'embedding_size': 32,    
            'batch_size': 2048,      # Netflix数据更大
            'lr': 0.005,             
            'regs': 1e-4,
            'init_std': 0.01         
        })
        
        self.data_generator = data_generator
        
        # 创建完整的RecBole配置对象
        config = create_recbole_config(self.args.embedding_size)
        
        # 创建适配后的数据集
        dataset = RecBoleDatasetAdapter(data_generator, config)
        
        # 创建BPR模型
        self.model = BPR(config, dataset).cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.regs)
        
        # 定义保存路径
        self.save_path = './Weights/BPR/Quick_BPR_netflix.pth'
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        print(f"模型将保存到: {self.save_path}")
    
    def get_bpr_scores(self, users, items):
        """获取BPR模型的预测分数"""
        try:
            # 转换为tensor
            if not isinstance(users, torch.Tensor):
                users = torch.tensor(users, dtype=torch.long).cuda()
            if not isinstance(items, torch.Tensor):
                items = torch.tensor(items, dtype=torch.long).cuda()
            
            # BPR的forward方法返回(user_emb, item_emb)
            output = self.model.forward(users, items)
            if isinstance(output, tuple) and len(output) == 2:
                user_emb, item_emb = output
                scores = torch.sum(user_emb * item_emb, dim=1)
            else:
                scores = output
            return scores
        except Exception as e:
            print(f"评分计算出错: {e}")
            # 回退到手动计算
            user_emb = self.model.user_embedding(users)
            item_emb = self.item_embedding(items)
            scores = torch.sum(user_emb * item_emb, dim=1)
            return scores
        
    def train_quick(self, epochs=500):  # Netflix数据较大，训练更多轮
        """快速训练BPR模型"""
        print(f"\n开始快速训练 {epochs} 轮...")
        
        # 记录起始时间
        start_time = time.time()
        
        # 训练前检查参数
        self._print_model_stats("初始模型参数")
        
        # 训练历史记录
        train_history = {
            'loss': [],
            'auc': [],
            'best_auc': 0,
            'best_epoch': 0
        }
        
        # 获取训练数据
        train_users = torch.tensor(self.data_generator.train_users, dtype=torch.long).cuda()
        train_items = torch.tensor(self.data_generator.train_items, dtype=torch.long).cuda()
        train_labels = torch.tensor(self.data_generator.train_labels, dtype=torch.float).cuda()
        
        batch_size = self.args.batch_size
        n_batches = (len(train_users) + batch_size - 1) // batch_size
        
        for epoch in range(epochs):
            epoch_start = time.time()
            total_loss = 0
            
            # 随机打乱数据
            perm = torch.randperm(len(train_users))
            train_users_shuffled = train_users[perm]
            train_items_shuffled = train_items[perm]
            train_labels_shuffled = train_labels[perm]
            
            self.model.train()
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(train_users))
                
                batch_users = train_users_shuffled[start_idx:end_idx]
                batch_items = train_items_shuffled[start_idx:end_idx]
                batch_labels = train_labels_shuffled[start_idx:end_idx]
                
                # 生成负样本
                neg_items = torch.randint(0, self.data_generator.n_items, (len(batch_users),), dtype=torch.long).cuda()
                
                # 正样本损失
                pos_scores = self.get_bpr_scores(batch_users, batch_items)
                neg_scores = self.get_bpr_scores(batch_users, neg_items)
                
                # BPR损失
                bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))
                
                # 正则化
                reg_loss = self.args.regs * (
                    torch.norm(self.model.user_embedding.weight) + 
                    torch.norm(self.model.item_embedding.weight)
                )
                
                loss = bpr_loss + reg_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / n_batches
            train_history['loss'].append(avg_loss)
            
            # 每10轮评估一次
            if epoch % 10 == 0 or epoch == epochs - 1:
                auc = self.evaluate_on_test_set()
                train_history['auc'].append(auc)
                
                if auc > train_history['best_auc']:
                    train_history['best_auc'] = auc
                    train_history['best_epoch'] = epoch
                
                epoch_time = time.time() - epoch_start
                print(f"Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | AUC: {auc:.4f} | Time: {epoch_time:.2f}s | Best: {train_history['best_auc']:.4f}@{train_history['best_epoch']}")
            else:
                epoch_time = time.time() - epoch_start
                print(f"Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")
        
        # 训练结束，计算总时间
        train_time = time.time() - start_time
        print(f"\n快速训练完成! 总耗时: {train_time:.2f}秒")
        print(f"最佳性能: AUC={train_history['best_auc']:.4f} (第{train_history['best_epoch']}轮)")
        
        # 训练后检查参数
        self._print_model_stats("训练后模型参数")
        
        # 保存模型
        print(f"\n保存模型到: {self.save_path}")
        torch.save(self.model.state_dict(), self.save_path)
        
        # 验证保存的模型
        self._validate_saved_model()
        
        # 保存训练历史
        self.save_training_history(train_history, train_history['best_auc'], train_history['best_epoch'], train_time)
        
        return self.save_path

    def evaluate_on_test_set(self):
        """在测试集上评估模型"""
        if len(self.data_generator.test_users) == 0:
            return 0.0
            
        self.model.eval()
        with torch.no_grad():
            test_users = torch.tensor(self.data_generator.test_users, dtype=torch.long).cuda()
            test_items = torch.tensor(self.data_generator.test_items, dtype=torch.long).cuda()
            test_labels = self.data_generator.test_labels
            
            # 批量预测
            batch_size = 2048
            all_scores = []
            
            for i in range(0, len(test_users), batch_size):
                batch_users = test_users[i:i+batch_size]
                batch_items = test_items[i:i+batch_size]
                scores = self.get_bpr_scores(batch_users, batch_items)
                all_scores.append(scores.cpu().numpy())
            
            pred_scores = np.concatenate(all_scores)
            auc = roc_auc_score(test_labels, pred_scores)
            
        return auc

    def save_training_history(self, history, best_auc, best_epoch, total_time):
        """保存训练历史"""
        result_dir = './results'
        os.makedirs(result_dir, exist_ok=True)
        
        result_file = f'{result_dir}/bpr_netflix_training_history.json'
        
        result_data = {
            'model': 'BPR',
            'dataset': 'netflix',
            'embedding_size': self.args.embedding_size,
            'batch_size': self.args.batch_size,
            'learning_rate': self.args.lr,
            'regularization': self.args.regs,
            'total_time': total_time,
            'best_auc': best_auc,
            'best_epoch': best_epoch,
            'train_loss': history['loss'],
            'test_auc': history['auc'],
            'data_stats': {
                'n_users': self.data_generator.n_users,
                'n_items': self.data_generator.n_items,
                'n_train': len(self.data_generator.train),
                'n_test': len(self.data_generator.test)
            }
        }
        
        import json
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        print(f"训练历史保存到: {result_file}")
    
    def _print_model_stats(self, title):
        """打印模型参数统计信息"""
        print(f"\n{title}:")
        user_weights = self.model.user_embedding.weight.data
        item_weights = self.model.item_embedding.weight.data
        
        print(f"用户嵌入: mean={user_weights.mean():.6f}, std={user_weights.std():.6f}")
        print(f"物品嵌入: mean={item_weights.mean():.6f}, std={item_weights.std():.6f}")
    
    def _validate_saved_model(self):
        """验证保存的模型"""
        try:
            # 创建新模型实例
            config = create_recbole_config(self.args.embedding_size)
            dataset = RecBoleDatasetAdapter(self.data_generator, config)
            test_model = BPR(config, dataset).cuda()
            
            # 加载保存的权重
            test_model.load_state_dict(torch.load(self.save_path))
            print("✅ 模型验证成功，权重加载正常")
            
        except Exception as e:
            print(f"❌ 模型验证失败: {e}")


def main():
    # 设置随机种子
    np.random.seed(1024)
    torch.manual_seed(1024)
    torch.cuda.manual_seed(1024)
    
    print("="*80)
    print("训练BPR模型用于IFRU测试 (Netflix数据集)")
    print("="*80)
    
    # 创建完整的args对象
    args = type('Args', (), {
        'embed_size': 32,
        'batch_size': 2048,
        'data_path': '/data/IFRU-main/Data/',
        'dataset': 'netflix',
        'attack': '',
        'data_type': 'full',
        'A_split': False,
        'A_n_fold': 100,
        'keep_prob': 1.0,
        'gcn_layers': 2,
        'dropout': False,
        'pretrain': 0,
        'lr': 0.005,
        'regs': 1e-4,
        'init_std': 0.01
    })
    
    try:
        # 加载数据
        print("加载Netflix数据集...")
        data_path = '/data/IFRU-main/Data/netflix'
        data_generator = NetflixDataGenerator(args, path=data_path)
        data_generator.set_train_mode(args.data_type)
        
        print(f"✅ 数据加载成功!")
        print(f"用户数: {data_generator.n_users}")
        print(f"物品数: {data_generator.n_items}")
        print(f"训练样本: {len(data_generator.train)}")
        print(f"测试样本: {len(data_generator.test)}")
        
        # 创建训练器并训练
        print("\n创建BPR训练器...")
        trainer = QuickBPRTrainer(data_generator)
        
        # 开始训练
        saved_model_path = trainer.train_quick(epochs=500)
        
        print(f"\n🎉 训练完成!")
        print(f"模型保存路径: {saved_model_path}")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    main()