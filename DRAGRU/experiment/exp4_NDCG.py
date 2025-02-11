import matplotlib.pyplot as plt
import numpy as np

# 随机生成数据 (每个子图随机生成)
np.random.seed(0)
data = {
    'ML-1M (BPR)': {
        'RecEraser': [0.1084, 0.1055 , 0.1123 ],
        'Origin CRAGRU': [0.2247, 0.2127 , 0.2167 ],
        'User Preference-based Filtering': [0.2814 , 0.2588 , 0.2553 ],
        'Diversity and Coverage-based Filtering': [0.2767 , 0.2552 , 0.2530 ],
        'Model-Based Filtering': [0.2976 , 0.2698 , 0.2612 ],

    },
    'ML-1M (LightGCN)': {
        'RecEraser': [0.1464 , 0.1467 , 0.1574 ],
        'Origin CRAGRU': [0.2354 , 0.2221 , 0.2270  ],
        'User Preference-based Filtering': [0.2890 , 0.2673 , 0.2643 ],
        'Diversity and Coverage-based Filtering': [0.2899 , 0.2675 , 0.2644 ],
        'Model-Based Filtering': [0.3131 , 0.2837 , 0.2761 ],

    },
    'Netflix (BPR)': {
        'RecEraser': [0.1032, 0.1086 , 0.1038 ],
        'Origin CRAGRU': [0.2639 , 0.2335 ,0.2014 ],
        'User Preference-based Filtering': [0.3207 , 0.2735 , 0.2272 ],
        'Diversity and Coverage-based Filtering': [0.3179 , 0.2727 , 0.2271 ],
        'Model-Based Filtering': [0.3717 , 0.3084 , 0.2481 ],
    },
    'Netflix (LightGCN)': {
        'RecEraser': [0.1303 , 0.1314 , 0.1254 ],
        'Origin CRAGRU': [0.2717 , 0.2400 , 0.2072 ],
        'User Preference-based Filtering': [0.3299 , 0.2827 , 0.2349 ],
        'Diversity and Coverage-based Filtering': [0.3335 , 0.2849 , 0.2353 ],
        'Model-Based Filtering': [0.3883 , 0.3218 , 0.2587 ],
    },
}

pc_ratios = ['TopK=5', 'TopK=10', 'TopK=20']
datasets = list(data.keys())
methods = ['RecEraser', 'Origin CRAGRU', 'User Preference-based Filtering', 'Diversity and Coverage-based Filtering', 'Model-Based Filtering']

# 设置字体和样式
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
plt.rcParams['legend.fontsize'] = 16

# 调整figsize，增加画布宽度
fig, axes = plt.subplots(1, 4, figsize=(22, 5))  # 4个子图横着排列

# 颜色设置，使用高对比度的淡色
bar_colors = ['#FFB6C1','#D3D3D3', '#B0E0E6', '#FFFACD', '#D8BFD8']  # Light blue, light pink, light green, light purple

# 图例标签
legend_labels_bar = [f'{method}' for method in methods]

# 创建图例对象
bar_handles, bar_labels = [], []

for i, dataset in enumerate(datasets):
    ax = axes[i]  # 获取当前子图

    x = np.arange(len(pc_ratios))
    width = 0.15  # 调整宽度使每组柱子为4个

    # 绘制柱状图
    for j, method in enumerate(methods):
        bars = ax.bar(x + j * width, data[dataset][method], width, color=bar_colors[j], edgecolor='black', label=legend_labels_bar[j] if i == 0 else "")
        if i == 0:  # 只为第一个子图添加图例
            bar_handles.append(bars[0])  # 取第一个条形图句柄
            bar_labels.append(legend_labels_bar[j])

    # 设置 x 轴刻度和标签
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(pc_ratios)

    # 设置 y 轴标签
    ax.set_ylabel('NDCG')

    # 设置标题
    ax.set_title(dataset)

    # 设置 y 轴范围 (根据图片目测估算，请根据实际数据调整)
    # if dataset == 'ML-1M (BPR)':
    #     ax.set_ylim(0, 0.3)
    # if dataset == 'ML-1M (LightGCN)':
    #     ax.set_ylim(0, 0.35)
    # 设置网格线，增强可读性
    ax.grid(axis='y', linestyle='--', alpha=0.7)

# 使用 subplots_adjust 精确控制子图间距
plt.subplots_adjust(left=0.07, right=0.92, wspace=0.2, top=0.8)  # 调整左右边界和水平间距

# 在上方居中显示图例，调整 y 值使其远离柱状图, 并添加边框
legend = plt.legend(handles=bar_handles, labels=bar_labels,
           loc='best', bbox_to_anchor=(1, 1.25), ncol=5, frameon=True, edgecolor='gray')

plt.savefig('exp4_NDCG.svg', format='svg')

# 显示图表
plt.show()
