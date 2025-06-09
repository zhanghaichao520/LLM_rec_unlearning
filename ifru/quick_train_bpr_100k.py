import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utility.load_data import *
from Model.bpr import BPR
from utility.compute import *
from sklearn.metrics import roc_auc_score
import time
import json


class RecBoleDatasetAdapter:
    """适配器类，将Data_for_LightGCN适配为RecBole期望的数据集格式"""
    
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


class QuickBPRTrainer:
    def __init__(self, data_generator):
        # 设置BPR模型参数 - 针对ML-100K优化
        self.args = type('Args', (), {
            'embedding_size': 64,    # ML-100K使用更大的嵌入维度
            'batch_size': 1024,      # ML-100K使用更小的批次大小
            'lr': 0.005,             # 保持相同的学习率
            'regs': 1e-4,            # 保持相同的正则化
            'init_std': 0.01         # 保持相同的初始化标准差
        })
        
        self.data_generator = data_generator
        
        # 创建完整的RecBole配置对象
        config = self.create_recbole_config()
        
        # 创建适配后的数据集
        adapted_dataset = RecBoleDatasetAdapter(data_generator, config)
        
        # 创建BPR模型
        self.model = BPR(config, adapted_dataset).cuda()
        
        # 手动设置用户和物品数量（如果需要）
        if not hasattr(self.model, 'n_users'):
            self.model.n_users = data_generator.n_users
        if not hasattr(self.model, 'n_items'):
            self.model.n_items = data_generator.n_items
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        
        # 定义保存路径
        self.save_path = './Weights/BPR/Quick_BPR_ml100k.pth'
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        print(f"模型将保存到: {self.save_path}")
        
        # 保存配置以便后续使用
        self.config = config
        self.adapted_dataset = adapted_dataset
    
    def create_recbole_config(self):
        """创建完整的RecBole配置对象"""
        # 创建一个包含所有必要字段的配置字典
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
            'embedding_size': self.args.embedding_size,
            'train_batch_size': self.args.batch_size,
            'eval_batch_size': self.args.batch_size,
            'learning_rate': self.args.lr,
            'weight_decay': self.args.regs,
            'train_neg_sample_args': {'distribution': 'uniform', 'sample_num': 1, 'alpha': 1.0, 'dynamic': False, 'candidate_num': 0},
            
            # 训练参数
            'epochs': 500,  # 减少轮数对于更小的数据集
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
            'NEG_ITEM_ID': 'neg_item',
            
            # 其他必要字段
            'MODEL_TYPE': 'general',
            'data_path': 'Data/Process/',
            'dataset': 'ml-100k',  # 修改为ml-100k
            'config_files': [],
            'neg_sampling': None,
            'eval_args': {'split': {'RS': [0.8, 0.1, 0.1]}, 'group_by': 'user', 'order': 'RO', 'mode': 'full'},
            
            # 避免RecBole自动处理数据
            'load_col': None,
            'unload_col': None,
            'unused_col': None,
            'additional_feat_suffix': None,
        }
        
        # 创建一个简单的配置类来模拟RecBole的Config对象
        class SimpleConfig:
            def __init__(self, config_dict):
                for key, value in config_dict.items():
                    setattr(self, key, value)
                self._config_dict = config_dict
            
            def __getitem__(self, key):
                return getattr(self, key, None)
            
            def __setitem__(self, key, value):
                setattr(self, key, value)
            
            def __contains__(self, key):
                return hasattr(self, key)
            
            def get(self, key, default=None):
                return getattr(self, key, default)
        
        return SimpleConfig(config_dict)
    
    def get_bpr_scores(self, users, items):
        """获取BPR模型的预测分数"""
        try:
            # BPR的forward方法返回(user_emb, item_emb)
            output = self.model.forward(users, items)
            if isinstance(output, tuple) and len(output) == 2:
                user_emb, item_emb = output
                # 计算点积得分
                scores = torch.sum(user_emb * item_emb, dim=1)
                return scores
            else:
                # 如果返回格式不符合预期，回退到手动方法
                user_emb = self.model.user_embedding(users)
                item_emb = self.model.item_embedding(items)
                scores = torch.sum(user_emb * item_emb, dim=1)
                return scores
        except Exception as e:
            print(f"BPR前向传播失败: {e}")
            # 回退到手动计算
            user_emb = self.model.user_embedding(users)
            item_emb = self.model.item_embedding(items)
            scores = torch.sum(user_emb * item_emb, dim=1)
            return scores
        
    def train_quick(self, epochs=500):
        """训练BPR模型 - 使用更少的轮数对于ML-100K"""
        print(f"\n开始训练BPR模型 {epochs} 轮...")
        
        # 记录起始时间
        start_time = time.time()
        
        # 训练前检查参数
        self._print_model_stats("初始模型参数")
        
        # 添加最佳模型保存
        best_auc = 0.0
        best_epoch = 0
        patience = 30  # 早停耐心值，对于较小数据集可以减少
        no_improve_count = 0
        
        # 记录训练历史
        train_history = {
            'epochs': [],
            'losses': [],
            'aucs': [],
            'times': []
        }
        
        # 保存初始模型，确保后续加载时文件存在
        print(f"保存初始模型到 {self.save_path}...")
        torch.save(self.model.state_dict(), self.save_path)
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # 设置为训练模式
            self.model.train()
            
            # 从训练数据中采样
            train_data = self.data_generator.train[['user', 'item', 'label']].values
            
            # 使用适合ML-100K的批次大小
            batch_size = min(4096, len(train_data))
            indices = np.random.choice(len(train_data), batch_size, replace=False)
            batch_data = train_data[indices]
            
            users = torch.LongTensor(batch_data[:, 0]).cuda()
            items = torch.LongTensor(batch_data[:, 1]).cuda()
            labels = torch.FloatTensor(batch_data[:, 2]).cuda()
            
            # 前向传播
            self.optimizer.zero_grad()
            
            # 使用我们的BPR得分计算方法
            scores = self.get_bpr_scores(users, items)
            
            # 调试信息（仅第一个epoch）
            if epoch == 0:
                print(f"调试信息:")
                print(f"  scores类型: {type(scores)}")
                print(f"  scores形状: {scores.shape}")
                print(f"  labels形状: {labels.shape}")
                print(f"  scores范围: [{scores.min().item():.4f}, {scores.max().item():.4f}]")
                print(f"  labels范围: [{labels.min().item():.4f}, {labels.max().item():.4f}]")
                print(f"  批次大小: {batch_size}")
            
            # 使用BCE损失
            loss = nn.BCEWithLogitsLoss()(scores, labels)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            epoch_time = time.time() - epoch_start
            
            # 计算AUC
            with torch.no_grad():
                probs = torch.sigmoid(scores).cpu().numpy()
                auc = roc_auc_score(batch_data[:, 2], probs)
            
            # 记录训练历史
            train_history['epochs'].append(epoch + 1)
            train_history['losses'].append(loss.item())
            train_history['aucs'].append(auc)
            train_history['times'].append(epoch_time)
            
            # 检查是否是最佳模型
            if auc > best_auc:
                best_auc = auc
                best_epoch = epoch + 1
                no_improve_count = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), self.save_path)
            else:
                no_improve_count += 1
            
            # 每10轮打印一次进度
            if (epoch + 1) % 10 == 0:
                avg_time = np.mean(train_history['times'][-10:])
                print(f"Epoch {epoch+1:4d}/{epochs} | "
                    f"损失: {loss.item():.6f} | "
                    f"AUC: {auc:.6f} | "
                    f"最佳AUC: {best_auc:.6f} (Epoch {best_epoch}) | "
                    f"平均用时: {avg_time:.3f}s")
            
            # 每50轮详细评估一次 (对于更小的数据集更频繁评估)
            if (epoch + 1) % 50 == 0:
                eval_auc = self.evaluate_on_test_set()
                print(f"  -> Epoch {epoch+1} 测试集AUC: {eval_auc:.6f}")
                
                # 如果测试集AUC也很好，更新最佳模型
                if eval_auc > best_auc:
                    best_auc = eval_auc
                    best_epoch = epoch + 1
                    torch.save(self.model.state_dict(), self.save_path)
                    print(f"  -> 保存新的最佳模型!")
            
            # 早停检查
            if no_improve_count >= patience:
                print(f"\n早停: {patience} 轮无改进，停止训练")
                print(f"最佳AUC: {best_auc:.6f} (Epoch {best_epoch})")
                break
            
            # 学习率调度 (可选)
            if (epoch + 1) % 100 == 0 and epoch > 0:
                # 每100轮降低学习率
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.8
                print(f"  -> 学习率调整为: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        # 训练结束，计算总时间
        train_time = time.time() - start_time
        print(f"\n训练完成! 总耗时: {train_time:.2f}秒")
        print(f"最佳AUC: {best_auc:.6f} (Epoch {best_epoch})")
        
        # 加载最佳模型
        print(f"\n加载最佳模型...")
        try:
            self.model.load_state_dict(torch.load(self.save_path))
            print(f"成功加载模型: {self.save_path}")
        except Exception as e:
            print(f"加载模型失败: {e}，将使用当前模型参数")
            # 确保至少保存当前模型
            torch.save(self.model.state_dict(), self.save_path)
        
        # 训练后检查参数
        self._print_model_stats("训练后模型参数")
        
        # 最终测试集评估
        final_test_auc = self.evaluate_on_test_set()
        print(f"最终测试集AUC: {final_test_auc:.6f}")
        
        # 保存训练历史
        self.save_training_history(train_history, best_auc, best_epoch, train_time)
        
        # 验证保存的模型
        self._validate_saved_model()
        
        return self.save_path

    def evaluate_on_test_set(self):
        """在测试集上评估模型"""
        self.model.eval()
        
        # 获取测试数据
        test_data = self.data_generator.test[['user', 'item', 'label']].values
        
        # ML-100K测试集较小，可以全部评估
        users = torch.LongTensor(test_data[:, 0]).cuda()
        items = torch.LongTensor(test_data[:, 1]).cuda()
        labels = test_data[:, 2]
        
        with torch.no_grad():
            scores = self.get_bpr_scores(users, items)
            probs = torch.sigmoid(scores).cpu().numpy()
        
        auc = roc_auc_score(labels, probs)
        return auc

    def save_training_history(self, history, best_auc, best_epoch, total_time):
        """保存训练历史"""
        # 转换numpy类型为Python原生类型
        def convert_numpy_types(obj):
            """递归转换numpy类型为Python原生类型"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        history_data = {
            'model_config': {
                'embedding_size': int(self.args.embedding_size),
                'learning_rate': float(self.args.lr),
                'batch_size': int(self.args.batch_size),
                'weight_decay': float(self.args.regs)
            },
            'training_history': {
                'epochs': [int(x) for x in history['epochs']],
                'losses': [float(x) for x in history['losses']],
                'aucs': [float(x) for x in history['aucs']],
                'times': [float(x) for x in history['times']]
            },
            'best_results': {
                'best_auc': float(best_auc),
                'best_epoch': int(best_epoch),
                'total_training_time': float(total_time)
            },
            'dataset_info': {
                'n_users': int(self.data_generator.n_users),
                'n_items': int(self.data_generator.n_items),
                'n_train_samples': int(len(self.data_generator.train))
            }
        }
        
        # 确保所有数据都是JSON可序列化的
        history_data = convert_numpy_types(history_data)
        
        # 保存到JSON文件
        history_path = self.save_path.replace('.pth', '_history.json')
        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        print(f"训练历史已保存到: {history_path}")
    
    def _print_model_stats(self, title):
        """打印模型参数统计信息"""
        print(f"\n{title}:")
        
        # 获取用户和物品嵌入
        user_weights = self.model.user_embedding.weight.data
        item_weights = self.model.item_embedding.weight.data
        
        print(f"  用户嵌入 - 均值: {user_weights.mean().item():.6f}, 方差: {user_weights.var().item():.6f}")
        print(f"  物品嵌入 - 均值: {item_weights.mean().item():.6f}, 方差: {item_weights.var().item():.6f}")
        print(f"  数据集大小 - 用户: {user_weights.shape[0]}, 物品: {item_weights.shape[0]}")
    
    def _validate_saved_model(self):
        """验证保存的模型可以正确加载"""
        print("\n验证保存的模型...")
        
        try:
            # 创建新模型
            config = self.create_recbole_config()
            adapted_dataset = RecBoleDatasetAdapter(self.data_generator, config)
            new_model = BPR(config, adapted_dataset).cuda()
            
            # 手动设置用户和物品数量
            if not hasattr(new_model, 'n_users'):
                new_model.n_users = self.data_generator.n_users
            if not hasattr(new_model, 'n_items'):
                new_model.n_items = self.data_generator.n_items
            
            # 加载保存的模型
            new_model.load_state_dict(torch.load(self.save_path))
            
            # 检查加载后的参数
            user_weights = new_model.user_embedding.weight.data
            item_weights = new_model.item_embedding.weight.data
            
            print(f"加载后 - 用户嵌入均值: {user_weights.mean().item():.6f}, 方差: {user_weights.var().item():.6f}")
            print(f"加载后 - 物品嵌入均值: {item_weights.mean().item():.6f}, 方差: {item_weights.var().item():.6f}")
            
            if abs(user_weights.mean().item()) < 1e-6 and abs(item_weights.mean().item()) < 1e-6:
                print("❌ 警告: 加载后的参数接近零，保存可能有问题")
            else:
                print("✅ 模型参数正常，保存/加载成功!")
                
        except Exception as e:
            print(f"❌ 模型验证失败: {e}")
            import traceback
            traceback.print_exc()


def main():
    # 设置随机种子
    np.random.seed(1024)
    torch.manual_seed(1024)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1024)
    
    print("="*80)
    print("训练BPR模型用于IFRU测试 (ML-100K)")
    print("="*80)
    
    # 创建完整的args对象，添加所有必要参数
    args = type('Args', (), {
        'embed_size': 64,            # 增大嵌入维度
        'batch_size': 1024,          # 适合ML-100K的批次大小
        'data_path': 'Data/Process/',
        'dataset': 'ml-100k',        # 使用ml-100k数据集
        'attack': '0.1',
        'data_type': 'full',
        'A_split': False,            # 添加所有必要的GCN参数
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
        print("加载ML-100K数据...")
        data_generator = Data_for_LightGCN(args, path=args.data_path + args.dataset + '/' + args.attack)
        data_generator.set_train_mode(args.data_type)
        
        print(f"数据加载成功: {data_generator.n_users} 用户, {data_generator.n_items} 物品")
        print(f"训练样本: {len(data_generator.train)} 条")
        print(f"测试样本: {len(data_generator.test)} 条")
        
        # 创建训练器并训练
        trainer = QuickBPRTrainer(data_generator)
        
        print(f"\n开始500轮训练...")
        print(f"预计训练时间: ~{500 * 0.05 / 60:.1f} 分钟")
        
        model_path = trainer.train_quick(epochs=500)  # ML-100K训练500轮
        
        # 打印使用此模型进行IFRU测试的命令
        print("\n"+"="*80)
        print("使用此训练好的模型测试IFRU:")
        print(f"python full_bpr_ifru_100k.py --model_path={model_path} --embed_size=64")
        print("="*80)
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    main()