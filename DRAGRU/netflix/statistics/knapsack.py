import numpy as np

## DP问题的比例分配计算
def knapsack(hit_rate):
    # 获取类目数目和比例数目
    num_categories = len(hit_rate)
    num_ratios = len(hit_rate[0])

    # 背包容量（最大比例总和为100%）
    max_capacity = 100

    # 动态规划数组
    dp = np.full((num_categories + 1, max_capacity + 1), -np.inf)
    dp[0][0] = 0  # 初始条件：没有选择类目时，命中率为0

    # 动态规划计算最大命中率
    for i in range(1, num_categories + 1):
        for j in range(max_capacity + 1):
            for k in range(num_ratios):
                ratio = (k + 1) * 5  # 比例：5%, 10%, ..., 50%
                if j >= ratio:  # 确保不会超出背包容量
                    dp[i][j] = max(dp[i][j], dp[i - 1][j - ratio] + hit_rate[i - 1][k])

    # 回溯选择每个类目的比例
    selected_ratios = []
    capacity = max_capacity
    for i in range(num_categories, 0, -1):
        for k in range(num_ratios):
            ratio = (k + 1) * 5
            if capacity >= ratio and dp[i][capacity] == dp[i - 1][capacity - ratio] + hit_rate[i - 1][k]:
                selected_ratios.append(ratio)
                capacity -= ratio
                break

    selected_ratios.reverse()  # 反转，因为我们是从后往前回溯的

    return selected_ratios, dp[num_categories][max_capacity]


# ml - 100k bpr
hit_rate = np.array([
    [0.0518, 0.0618, 0.0373, 0.117, 0.043, 0.1822, 0.1839, 0.2444, 0.3032, 0.3424],
    [0.0889, 0.0829, 0.0687, 0.153, 0.0756, 0.1845, 0.2074, 0.3026, 0.3361, 0.3594],
    [0.0085, 0.0384, 0.0398, 0.0712, 0.1208, 0.1692, 0.2537, 0.2351, 0.2806, 0.3337],
    [0.0326, 0.0255, 0.0773, 0.0232, 0.1578, 0.1713, 0.2048, 0.2667, 0.3294, 0.3642],
    [0.0451, 0.0149, 0.0282, 0.1197, 0.1517, 0.1721, 0.2256, 0.2584, 0.3363, 0.3607]
])
# ml-1m bpr
# hit_rate = np.array([
#     [0.0413, 0.0687, 0.0893, 0.1257, 0.1463, 0.1935, 0.2101, 0.2413, 0.2719, 0.3146],
#     [0.0259, 0.052,  0.0792, 0.108,  0.1335, 0.1581, 0.1946, 0.2277, 0.2793, 0.3126],
#     [0.039,  0.0632, 0.0904, 0.1217, 0.1512, 0.1809, 0.2038, 0.2393, 0.2743, 0.3124],
#     [0.0633, 0.0622, 0.067,  0.1306, 0.1679, 0.1899, 0.215,  0.2494, 0.2834, 0.3084],
#     [0.06,   0.0823, 0.1065, 0.1334, 0.1678, 0.2064, 0.2555, 0.2729, 0.3124, 0.334 ]
# ])

# ml-100k lightgcn
# hit_rate = np.array([
#     [0.0677, 0.0682, 0.1254, 0.1696, 0.2231, 0.2663, 0.2224, 0.2834, 0.3692, 0.3741],
#     [0.0844, 0.1073, 0.1202, 0.2625, 0.3081, 0.27,   0.3431, 0.3702, 0.4211, 0.3746],
#     [0.0222, 0.057,  0.1064, 0.1301, 0.1545, 0.1756, 0.2111, 0.1947, 0.2891, 0.3241],
#     [0.0372, 0.0686, 0.104,  0.1439, 0.2053, 0.2088, 0.2266, 0.2559, 0.279,  0.3067],
#     [0.0475, 0.0365, 0.1394, 0.1682, 0.2209, 0.2314, 0.3073, 0.2695, 0.3065, 0.318]
# ])
# ml-1m lightgcn
# hit_rate = np.array([
#     [0.061, 0.0951, 0.1144, 0.143, 0.1752, 0.1777, 0.2316, 0.2655, 0.2811, 0.3276],
#     [0.0492, 0.0739, 0.0843, 0.109, 0.1197, 0.1842, 0.1629, 0.2573, 0.2898, 0.3229],
#     [0.0602, 0.1011, 0.1263, 0.1306, 0.169, 0.1661, 0.2216, 0.2063, 0.286, 0.3378],
#     [0.0751, 0.129, 0.1589, 0.1858, 0.2017, 0.2448, 0.2639, 0.3061, 0.3209, 0.3486],
#     [0.0866, 0.1186, 0.1837, 0.2145, 0.2413, 0.26, 0.2767, 0.315, 0.3132, 0.3669]
# ])
# 计算最大命中率及每个类目的选择比例
selected_ratios, max_hit_rate = knapsack(hit_rate)

# 输出结果
print("每个类目的选择比例:", selected_ratios)
print("最大命中率总和:", max_hit_rate)