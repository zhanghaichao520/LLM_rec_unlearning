import scipy.stats as stats

def paired_ttest(data1, data2):
    """
    进行配对t检验。

    参数：
    data1: 第一组数据，列表或NumPy数组。
    data2: 第二组数据，列表或NumPy数组，与data1长度相同。

    返回：
    t统计量和p值。如果输入数据长度不一致，则返回错误信息。
    """
    if len(data1) != len(data2):
        return "错误：两组数据长度不一致。"

    t_statistic, p_value = stats.ttest_rel(data1, data2)
    return t_statistic, p_value

# 示例数据
ml_100k_RecEraser = [0.3204,0.1011,0.4582,0.1100,0.6337,0.1274,0.4806,0.1557,0.6137,0.1668,0.7491,0.1873]
ml_100k_DRAGRU = [0.6193,0.2864,0.7116,0.2801,0.7805,0.2886,0.5366,0.2241,0.6384,0.2156,0.7169,0.2216]

# 进行配对t检验
t_ml_100k, p_ml_100k = paired_ttest(ml_100k_RecEraser, ml_100k_DRAGRU)

print("ml-100k p值:", p_ml_100k)

ml_1m_RecEraser = [0.3514,0.1084,0.4833,0.1055,0.6260,0.1123,0.4560,0.1464,0.6003,0.1467,0.7379,0.1574]
ml_1m_DRAGRU = [0.6007,0.2764,0.6849,0.2563,0.7526,0.2532,0.6215,0.2889,0.7005,0.2669,0.7616,0.2637]

# 进行配对t检验
t_ml_1m, p_ml_1m = paired_ttest(ml_1m_RecEraser, ml_1m_DRAGRU)

print("ml-1m p值:", p_ml_1m)

netflix_RecEraser = [0.4303,0.1032,0.6673,0.1086,0.8361,0.1038,0.4959,0.1303,0.7129,0.1314,0.8772,0.1254]
netflix_DRAGRU = [0.7266,0.3230,0.8190,0.2759,0.8651,0.2286,0.7315,0.3352,0.8198,0.2852,0.8655,0.2365]

# 进行配对t检验
t_netflix, p_netflix = paired_ttest(netflix_RecEraser, netflix_DRAGRU)

print("netflix p值:", p_netflix)

# 判断是否显著
alpha = 0.05  # 显著性水平
if p_ml_1m < alpha:
    print("p值小于", alpha, "，差异具有统计学意义。")
else:
    print("p值大于等于", alpha, "，差异不具有统计学意义。")