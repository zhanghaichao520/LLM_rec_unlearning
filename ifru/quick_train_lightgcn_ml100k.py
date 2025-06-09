import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utility.load_data import *
from Model.Lightgcn import LightGCN
from utility.compute import *
from sklearn.metrics import roc_auc_score
import time


class QuickTrainer:
    def __init__(self, data_generator):
        # 设置更小的模型参数以加速训练
        self.args = type('Args', (), {
            'embed_size': 32,        # 较小的嵌入尺寸
            'batch_size': 2048,      # ml-100k数据量小，减小batch size
            'gcn_layers': 2,         # 较少的GCN层
            'lr': 0.005,             # 更高的学习率加速训练
            'regs': 1e-4,
            'keep_prob': 1.0,
            'A_n_fold': 100,
            'A_split': False,
            'dropout': False,
            'pretrain': 0,
            'init_std': 0.01         # 较大的初始化标准差，确保值不为0
        })
        
        self.data_generator = data_generator
        self.model = LightGCN(self.args, dataset=data_generator).cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        
        # 定义保存路径 - 修改为ml100k
        self.save_path = './Weights/LightGCN/Quick_LightGCN_ml100k.pth'
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        print(f"模型将保存到: {self.save_path}")
        
    def train_quick(self, epochs=8):  # ml-100k增加到8轮
        """快速训练几个epoch"""
        print(f"\n开始快速训练 {epochs} 轮...")
        
        # 记录起始时间
        start_time = time.time()
        
        # 训练前检查参数
        self._print_model_stats("初始模型参数")
        
        for epoch in range(epochs):
            # 设置为训练模式
            self.model.train()
            
            # 从训练数据中采样
            train_data = self.data_generator.train[['user', 'item', 'label']].values
            
            # 动态调整batch size适配数据量
            batch_size = min(2048, len(train_data))
            indices = np.random.choice(len(train_data), batch_size, replace=False)
            batch_data = train_data[indices]
            
            users = torch.LongTensor(batch_data[:, 0]).cuda()
            items = torch.LongTensor(batch_data[:, 1]).cuda()
            labels = torch.FloatTensor(batch_data[:, 2]).cuda()
            
            # 前向传播
            self.optimizer.zero_grad()
            scores = self.model.forward(users, items)
            loss = nn.BCEWithLogitsLoss()(scores, labels)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 计算AUC
            with torch.no_grad():
                probs = torch.sigmoid(scores).cpu().numpy()
                auc = roc_auc_score(batch_data[:, 2], probs)
            
            print(f"Epoch {epoch+1}/{epochs} | 损失: {loss.item():.6f} | AUC: {auc:.6f}")
        
        # 训练结束，计算总时间
        train_time = time.time() - start_time
        print(f"\n快速训练完成! 总耗时: {train_time:.2f}秒")
        
        # 训练后检查参数
        self._print_model_stats("训练后模型参数")
        
        # 保存模型
        print(f"\n保存模型到: {self.save_path}")
        torch.save(self.model.state_dict(), self.save_path)
        
        # 验证保存的模型
        self._validate_saved_model()
        
        return self.save_path
    
    def _print_model_stats(self, title):
        """打印模型参数统计信息"""
        print(f"\n{title}:")
        user_weights = self.model.embedding_user.weight.data
        item_weights = self.model.embedding_item.weight.data
        
        print(f"  用户嵌入 - 均值: {user_weights.mean().item():.6f}, 方差: {user_weights.var().item():.6f}")
        print(f"  物品嵌入 - 均值: {item_weights.mean().item():.6f}, 方差: {item_weights.var().item():.6f}")
        print(f"  数据集大小 - 用户: {user_weights.shape[0]}, 物品: {item_weights.shape[0]}")
    
    def _validate_saved_model(self):
        """验证保存的模型可以正确加载"""
        print("\n验证保存的模型...")
        
        try:
            # 创建新模型
            new_model = LightGCN(self.args, dataset=self.data_generator).cuda()
            
            # 加载保存的模型
            new_model.load_state_dict(torch.load(self.save_path))
            
            # 检查加载后的参数
            user_weights = new_model.embedding_user.weight.data
            item_weights = new_model.embedding_item.weight.data
            
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
    torch.cuda.manual_seed(1024)
    
    print("="*80)
    print("快速训练LightGCN小模型用于ml-100k IFRU测试")
    print("="*80)
    
    # 创建完整的args对象 - 修改为ml-100k
    args = type('Args', (), {
        'embed_size': 32,
        'batch_size': 2048,
        'data_path': 'Data/Process/',
        'dataset': 'ml-100k',  # 修改数据集名称
        'attack': '0.1',       # 保持和ml-1m一样的结构
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
        print("加载ML-100K数据...")
        data_generator = Data_for_LightGCN(args, path=args.data_path + args.dataset + '/' + args.attack)
        data_generator.set_train_mode(args.data_type)
        
        print(f"数据加载成功: {data_generator.n_users} 用户, {data_generator.n_items} 物品")
        
        # 创建训练器并训练
        trainer = QuickTrainer(data_generator)
        model_path = trainer.train_quick(epochs=8)  # ml-100k训练8轮
        
        # 打印使用此模型进行IFRU测试的命令
        print("\n"+"="*80)
        print("使用此快速训练模型测试IFRU:")
        print(f"python full_lightgcn_ifru_ml100k.py --model_path={model_path}")
        print("="*80)
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    main()