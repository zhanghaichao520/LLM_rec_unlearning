import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time as time_module  # 避免名称冲突
from sklearn.metrics import roc_auc_score

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from utility.load_data import *
from Model.Lightgcn import LightGCN
from utility.compute import *
from netflix_data_generator import NetflixDataGenerator


class NetflixLightGCNAdapter:
    """将NetflixDataGenerator适配为LightGCN期望的Data_for_LightGCN格式"""
    
    def __init__(self, netflix_generator):
        self.netflix_generator = netflix_generator
        
        # 复制基本属性
        self.n_users = netflix_generator.n_users
        self.n_items = netflix_generator.n_items
        self.train_users = netflix_generator.train_users
        self.train_items = netflix_generator.train_items
        self.train_labels = netflix_generator.train_labels
        self.test_users = netflix_generator.test_users
        self.test_items = netflix_generator.test_items  
        self.test_labels = netflix_generator.test_labels
        
        # 创建图结构
        self.create_graph_structure()
        
        print(f"✅ LightGCN适配器创建成功")
        print(f"图结构: {self.Graph.shape}")
        print(f"稀疏图非零元素: {self.Graph.nnz}")
    
    def create_graph_structure(self):
        """创建用户-物品二分图"""
        print("开始创建图结构...")
        
        # 用户-物品交互矩阵
        user_item_matrix = sp.coo_matrix(
            (np.ones(len(self.train_users)), 
             (self.train_users, self.train_items)),
            shape=(self.n_users, self.n_items)
        )
        
        print(f"用户-物品交互矩阵: {user_item_matrix.shape}, 非零元素: {user_item_matrix.nnz}")
        
        # 创建二分图邻接矩阵
        # [0,     R]
        # [R^T,   0]
        zero_user = sp.coo_matrix((self.n_users, self.n_users))
        zero_item = sp.coo_matrix((self.n_items, self.n_items))
        
        # 构建完整的邻接矩阵
        adj_mat = sp.bmat([
            [zero_user, user_item_matrix],
            [user_item_matrix.T, zero_item]
        ], format='coo')
        
        print(f"邻接矩阵: {adj_mat.shape}, 非零元素: {adj_mat.nnz}")
        
        # 归一化
        adj_mat = adj_mat.tocsr()  # 转换为CSR以便计算
        rowsum = np.array(adj_mat.sum(1)).flatten()
        
        # 处理零度数节点
        d_inv = np.power(rowsum, -0.5)
        d_inv[np.isinf(d_inv)] = 0.0
        d_inv[np.isnan(d_inv)] = 0.0
        d_mat_inv = sp.diags(d_inv)
        
        norm_adj = d_mat_inv.dot(adj_mat).dot(d_mat_inv)
        self.Graph = norm_adj.tocsr()
        
        print(f"归一化后图: {self.Graph.shape}, 非零元素: {self.Graph.nnz}")
        
        # 创建稀疏tensor版本（用于GPU）- 关键修复！
        # 转换为COO格式以获取indices
        coo_graph = self.Graph.tocoo()
        
        indices = torch.from_numpy(
            np.vstack([coo_graph.row, coo_graph.col]).astype(np.int64)
        )
        values = torch.from_numpy(coo_graph.data.astype(np.float32))
        shape = coo_graph.shape
        
        # 创建稀疏张量并移到GPU
        self.sparse_graph = torch.sparse.FloatTensor(indices, values, shape).cuda()
        print(f"稀疏张量创建成功并移到GPU: {shape}")
    
    def getSparseGraph(self):
        """返回稀疏图（兼容LightGCN接口）"""
        return self.sparse_graph
    
    def set_train_mode(self, mode):
        """兼容接口"""
        print(f"设置训练模式: {mode}")
        pass
    
    def __getattr__(self, name):
        """代理其他属性到原始netflix_generator"""
        return getattr(self.netflix_generator, name)


class CustomLightGCN(nn.Module):
    """自定义LightGCN模型，确保正确使用我们的稀疏图"""
    
    def __init__(self, args, dataset):
        super(CustomLightGCN, self).__init__()
        
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        self.latent_dim = args.embed_size
        self.n_layers = args.gcn_layers
        self.keep_prob = args.keep_prob
        self.A_split = args.A_split
        self.dropout = args.dropout
        
        # 获取稀疏图
        self.Graph = dataset.getSparseGraph()
        print(f"LightGCN接收到的图类型: {type(self.Graph)}")
        
        # 初始化嵌入
        self.embedding_user = nn.Embedding(self.n_users, self.latent_dim)
        self.embedding_item = nn.Embedding(self.n_items, self.latent_dim)
        
        # 初始化权重
        self.f = nn.Sigmoid()
        self.__init_weight()
    
    def __init_weight(self):
        """初始化权重"""
        nn.init.normal_(self.embedding_user.weight, std=0.01)
        nn.init.normal_(self.embedding_item.weight, std=0.01)
        print('use NORMAL distribution initilizer')
    
    def __dropout_x(self, x, keep_prob):
        """Dropout"""
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        """对图进行dropout"""
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """图卷积传播"""
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        
        embs = [all_emb]
        
        if self.dropout:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                # 确保使用PyTorch稀疏矩阵乘法
                all_emb = torch.sparse.mm(g_droped, all_emb)
                
            embs.append(all_emb)
        
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.n_users, self.n_items])
        
        return users, items
    
    def getUsersRating(self, users):
        """获取用户对所有物品的评分"""
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        """获取嵌入"""
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        """BPR损失"""
        (users_emb, pos_emb, neg_emb, 
        userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
       
    def forward(self, users, items):
        """前向传播"""
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma


class QuickTrainer:
    def __init__(self, data_generator):
        # 设置更小的模型参数以加速训练 - 针对Netflix调整
        self.args = type('Args', (), {
            'embed_size': 32,
            'batch_size': 2048,      # Netflix数据较大
            'gcn_layers': 2,
            'lr': 0.005,
            'regs': 1e-4,
            'keep_prob': 1.0,
            'A_n_fold': 100,
            'A_split': False,
            'dropout': False,
            'pretrain': 0,
            'init_std': 0.01
        })
        
        self.data_generator = data_generator
        
        # 使用自定义LightGCN模型
        print("初始化自定义LightGCN模型...")
        self.model = CustomLightGCN(self.args, dataset=data_generator).cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        
        # 定义保存路径
        self.save_path = './Weights/LightGCN/Quick_LightGCN_netflix.pth'
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        print(f"模型将保存到: {self.save_path}")
        
    def train_quick(self, epochs=10):  # Netflix数据较大，适当增加训练轮数
        """快速训练几个epoch"""
        print(f"\n开始快速训练 {epochs} 轮...")
        
        # 记录起始时间
        start_time = time_module.time()
        
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
            epoch_start = time_module.time()
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
                
                try:
                    # 前向传播
                    all_users, all_items = self.model.computer()  # 自定义LightGCN的computer方法
                    
                    # 获取嵌入
                    pos_user_emb = all_users[batch_users]
                    pos_item_emb = all_items[batch_items]
                    neg_item_emb = all_items[neg_items]
                    
                    # 计算分数
                    pos_scores = torch.sum(pos_user_emb * pos_item_emb, dim=1)
                    neg_scores = torch.sum(pos_user_emb * neg_item_emb, dim=1)
                    
                    # BPR损失
                    bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))
                    
                    # 正则化损失
                    reg_loss = self.args.regs * (
                        torch.norm(pos_user_emb) + 
                        torch.norm(pos_item_emb) + 
                        torch.norm(neg_item_emb)
                    ) / len(batch_users)
                    
                    loss = bpr_loss + reg_loss
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    
                except Exception as e:
                    print(f"批次 {batch_idx} 训练出错: {e}")
                    # 检查图的类型
                    graph = self.data_generator.getSparseGraph()
                    print(f"图类型: {type(graph)}")
                    print(f"图设备: {graph.device if hasattr(graph, 'device') else 'N/A'}")
                    raise e
            
            avg_loss = total_loss / n_batches
            train_history['loss'].append(avg_loss)
            
            # 每轮都评估（因为总轮数较少）
            auc = self.evaluate_on_test_set()
            train_history['auc'].append(auc)
            
            if auc > train_history['best_auc']:
                train_history['best_auc'] = auc
                train_history['best_epoch'] = epoch
            
            epoch_time = time_module.time() - epoch_start
            print(f"Epoch {epoch:2d}/{epochs} | Loss: {avg_loss:.4f} | AUC: {auc:.4f} | Time: {epoch_time:.2f}s | Best: {train_history['best_auc']:.4f}@{train_history['best_epoch']}")
        
        # 训练结束，计算总时间
        train_time = time_module.time() - start_time
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
            try:
                # 获取所有嵌入
                all_users, all_items = self.model.computer()
                
                test_users = torch.tensor(self.data_generator.test_users, dtype=torch.long).cuda()
                test_items = torch.tensor(self.data_generator.test_items, dtype=torch.long).cuda()
                test_labels = self.data_generator.test_labels
                
                # 批量预测
                batch_size = 2048
                all_scores = []
                
                for i in range(0, len(test_users), batch_size):
                    batch_users = test_users[i:i+batch_size]
                    batch_items = test_items[i:i+batch_size]
                    
                    user_emb = all_users[batch_users]
                    item_emb = all_items[batch_items]
                    scores = torch.sum(user_emb * item_emb, dim=1)
                    all_scores.append(scores.cpu().numpy())
                
                pred_scores = np.concatenate(all_scores)
                auc = roc_auc_score(test_labels, pred_scores)
                
            except Exception as e:
                print(f"评估时出错: {e}")
                return 0.0
            
        return auc

    def save_training_history(self, history, best_auc, best_epoch, total_time):
        """保存训练历史"""
        result_dir = './results'
        os.makedirs(result_dir, exist_ok=True)
        
        result_file = f'{result_dir}/lightgcn_netflix_training_history.json'
        
        result_data = {
            'model': 'LightGCN',
            'dataset': 'netflix',
            'embedding_size': self.args.embed_size,
            'gcn_layers': self.args.gcn_layers,
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
        
        # 检查模型是否有嵌入层
        if hasattr(self.model, 'embedding_user') and hasattr(self.model, 'embedding_item'):
            user_weights = self.model.embedding_user.weight.data
            item_weights = self.model.embedding_item.weight.data
            
            print(f"用户嵌入: mean={user_weights.mean():.6f}, std={user_weights.std():.6f}")
            print(f"物品嵌入: mean={item_weights.mean():.6f}, std={item_weights.std():.6f}")
        else:
            print("模型参数信息不可用")
    
    def _validate_saved_model(self):
        """验证保存的模型"""
        try:
            # 创建新模型实例
            test_model = CustomLightGCN(self.args, dataset=self.data_generator).cuda()
            
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
    print("快速训练LightGCN小模型用于Netflix IFRU测试")
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
        netflix_generator = NetflixDataGenerator(args, path=data_path)
        
        # 创建LightGCN适配器
        print("创建LightGCN适配器...")
        data_generator = NetflixLightGCNAdapter(netflix_generator)
        data_generator.set_train_mode(args.data_type)
        
        print(f"✅ 数据加载成功!")
        print(f"用户数: {data_generator.n_users}")
        print(f"物品数: {data_generator.n_items}")
        print(f"训练样本: {len(data_generator.train_users)}")
        print(f"测试样本: {len(data_generator.test_users)}")
        
        # 创建训练器并训练
        print("\n创建LightGCN训练器...")
        trainer = QuickTrainer(data_generator)
        
        # 开始训练
        saved_model_path = trainer.train_quick(epochs=1000)
        
        print(f"\n🎉 训练完成!")
        print(f"模型保存路径: {saved_model_path}")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    main()