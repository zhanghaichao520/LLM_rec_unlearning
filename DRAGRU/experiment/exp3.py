import matplotlib.pyplot as plt
import numpy as np

# 数据 (请根据您的实际数据进行替换)
data = {
    # 'ML-100K': {
    #     'Forget set': [0.2021  , 0.3617    ,0.4680   ],
    #     'Remain set': [0.4405  , 0.6454    , 0.7196  ],
    #     'Forget set ndcg': [0.2021  , 0.1675    ,0.1655  ],
    #     'Remain set ndcg': [0.4405 , 0.3742    , 0.3516 ]
    # },
    'ML-1M (BPR)': {
        'Forget set': [0.2367  , 0.4139   ,0.4966   ],
        'Remain set': [0.4238  , 0.6122   , 0.6863 ],
        'Forget set ndcg': [0.2367  , 0.2044   ,0.1952  ],
        'Remain set ndcg': [0.4238  , 0.3593   , 0.3298 ]
    },
    'Netflix (BPR)': {
        'Forget set': [0.32 ,0.5627 ,0.6910   ],
        'Remain set': [0.5468, 0.7597,0.8386   ],
        'Forget set ndcg': [0.32,0.2814 ,0.2596   ],
        'Remain set ndcg': [0.5468, 0.4630,0.4124   ]
    },
    'ML-1M (LightGCN)': {
            'Forget set': [0.2367, 0.4039, 0.5099  ],
            'Remain set': [0.4435, 0.6245, 0.6948],
            'Forget set ndcg': [0.2367, 0.2065,0.1946 ],
            'Remain set ndcg': [0.4435, 0.3769, 0.3436 ]
        },
        'Netflix (LightGCN)': {
            'Forget set': [0.3379,0.5641,  0.691  ],
            'Remain set': [0.5578, 0.775, 0.8549],
            'Forget set ndcg': [0.3379,0.2806, 0.261],
            'Remain set ndcg': [0.5578,0.4786, 0.4293 ]
        },
}

pc_ratios = ['TopK=1', 'TopK=3', 'TopK=5']
datasets = list(data.keys())
methods = ['Forget set', 'Remain set']
rdr_methods = ['Forget set ndcg', 'Remain set ndcg']

# 设置字体和样式
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
plt.rcParams['legend.fontsize'] = 16
# 调整figsize，增加画布宽度
fig, axes = plt.subplots(1, 4, figsize=(22, 5))

# 颜色设置，使用高对比度的淡色
bar_colors = ['#ADD8E6', '#FFB6C1']  # Light blue, light pink
line_colors = ['#4682B4', '#FA8072']  # Blue, Salmon

# 图例标签
legend_labels_bar = [f'{method} (HR)' for method in methods]
legend_labels_line = [f'{rdr_method.replace(" ndcg", "")} (NDCG)' for rdr_method in rdr_methods]

# 创建图例对象
bar_handles, bar_labels = [], []
line_handles, line_labels = [], []

for i, dataset in enumerate(datasets):
    ax1 = axes[i]
    ax2 = ax1.twinx()

    x = np.arange(len(pc_ratios))
    width = 0.35

    # 绘制柱状图，同一比例的柱子挨着
    for j, method in enumerate(methods):
        bars = ax1.bar(x + j * (width / 2), data[dataset][method], width / 2,
                       color=bar_colors[j], edgecolor='black', label=legend_labels_bar[j] if i == 0 else "")
        if i == 0:  # 只为第一个子图添加图例
            bar_handles.append(bars[0])  # 取第一个条形图句柄
            bar_labels.append(legend_labels_bar[j])

    for j, rdr_method in enumerate(rdr_methods):
        line, = ax2.plot(x, data[dataset][rdr_method], marker='o' if j == 0 else 's',
                         linestyle=':' if j == 0 else '--', color=line_colors[j], label=legend_labels_line[j] if i == 0 else "")
        if i == 0:  # 只为第一个子图添加图例
            line_handles.append(line)  # 取折线图句柄
            line_labels.append(legend_labels_line[j])

    # 设置 x 轴刻度和标签
    ax1.set_xticks(x + width / 3)
    ax1.set_xticklabels(pc_ratios)
    # ax1.set_xlabel('PC Interactions Ratio')

    # 设置 y 轴标签
    ax1.set_ylabel('Hit Ratio')
    ax2.set_ylabel('NDCG')

    # 设置标题
    ax1.set_title(dataset)

    # 设置 y 轴范围 (根据图片目测估算，请根据实际数据调整)
    # if dataset == 'ML-100K':
    #     ax1.set_ylim(0.1, 0.8)
    #     ax2.set_ylim(0.1, 0.5)
    # elif dataset == 'ML-1M':
    #     ax1.set_ylim(0.1, 0.7)
    #     ax2.set_ylim(0.1, 0.5)
    # else:  # Netflix
    #     ax1.set_ylim(0.1, 0.9)
    #     ax2.set_ylim(0.1, 0.6)

    # 设置网格线，增强可读性
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

# 使用 subplots_adjust 精确控制子图间距
plt.subplots_adjust(left=0.07, right=0.92, wspace=0.45, top=0.82)  # 调整左右边界和水平间距

# 在上方居中显示图例，调整 y 值使其远离柱状图, 并添加边框
legend = plt.legend(handles=bar_handles + line_handles, labels=bar_labels + line_labels,
           loc='best', bbox_to_anchor=(1.1, 1.25), ncol=4, frameon=True, edgecolor='gray')

plt.savefig('exp3.svg', format='svg')

# 显示图表
plt.show()