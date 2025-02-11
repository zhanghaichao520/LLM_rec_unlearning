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


# netflix BPR
hit_rate = np.array([
    [0.0358, 0.0565, 0.0782, 0.0973, 0.1302, 0.1747, 0.2241, 0.2704, 0.3105, 0.3631],
    [0.0613, 0.0785, 0.1186, 0.1314, 0.1514, 0.1541, 0.1740, 0.2075, 0.2227, 0.2890],
    [0.0566, 0.0664, 0.1100, 0.1227, 0.1307, 0.1455, 0.1712, 0.2071, 0.2502, 0.2903],
    [0.0328, 0.0557, 0.0723, 0.1076, 0.1512, 0.1792, 0.2125, 0.2967, 0.3357, 0.4018],
    [0.1234, 0.1071, 0.1402, 0.1849, 0.2398, 0.2719, 0.3112, 0.3326, 0.3549, 0.3646]
])
# # netflix lightGCN
# hit_rate = np.array([
#     [0.0586, 0.065,  0.0767, 0.0947, 0.1254, 0.1473, 0.2468, 0.2238, 0.268,  0.3024],
#     [0.0728, 0.128,  0.14,   0.1511, 0.1465, 0.1596, 0.1725, 0.1833, 0.2153, 0.2518],
#     [0.0666, 0.1165, 0.1219, 0.1262, 0.1324, 0.1337, 0.1932, 0.1866, 0.2174, 0.2494],
#     [0.0488, 0.0551, 0.0703, 0.1232, 0.1445, 0.1812, 0.2201, 0.3063, 0.3448, 0.3406],
#     [0.1378, 0.1566, 0.2173, 0.3009, 0.3101, 0.3252, 0.3343, 0.3496, 0.3564, 0.3625]
# ])
# 计算最大命中率及每个类目的选择比例
selected_ratios, max_hit_rate = knapsack(hit_rate)

# 输出结果
print("每个类目的选择比例:", selected_ratios)
print("最大命中率总和:", max_hit_rate)
