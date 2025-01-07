import pandas as pd
import os


# 创建保存目录
output_dir = "dataset"
DATASET = "netflix-process"
COL_NAME = "movie_title:token_seq"
K = 5
item_attr_path = os.path.join(f"{output_dir}/{DATASET}", f"{DATASET}.item")
items = pd.read_csv(item_attr_path, delimiter='\t')

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


from gensim.models import KeyedVectors
from gensim.test.utils import datapath
model = KeyedVectors.load_word2vec_format(datapath(r"GoogleNews-vectors-negative300.bin"), binary=True)

# 获取每部电影的词向量（通过电影名）
def get_movie_vector(movie_name):
    words = movie_name.split()  # 按空格拆分电影名
    vectors = []
    for word in words:
        if word.lower() in model.key_to_index:  # 只使用模型中有的词
            vectors.append(model[word.lower()])
    if vectors:
        return np.mean(vectors, axis=0)  # 取平均值作为电影的特征向量
    else:
        return np.zeros(model.vector_size)  # 如果没有匹配的词，返回零向量

# 为每部电影生成特征向量
movie_vectors = np.array([get_movie_vector(movie) for movie in items[COL_NAME]])

# 输出类别种数
print(f'Number of unique categories: {len(movie_vectors)}')



# 数据标准化（对词向量进行标准化，以便KMeans表现更好）
scaler = StandardScaler()
movie_vectors_scaled = scaler.fit_transform(movie_vectors)

# 使用PCA进行降维，以便可视化
pca = PCA(n_components=2)
movie_vectors_2d = pca.fit_transform(movie_vectors_scaled)

# 应用KMeans聚类
kmeans = KMeans(n_clusters=K, random_state=42)
kmeans.fit(movie_vectors_2d)

# 获取每个类目的聚类标签
cluster_labels = kmeans.labels_

categories_map = {}
# 输出每个类目对应的聚类标签
for movie, label in zip(items[COL_NAME], cluster_labels):
    print(f"{movie}, Cluster: {label}")
    label = int(label)
    if label not in categories_map:
        categories_map[label] = []  # 如果该标签还没有，初始化为空列表
    categories_map[label].append(movie)


import json
# 将矩阵写入 CSV 文件
with open(f'{DATASET}-{K}-cluster.csv', 'w', newline='') as f:
    json.dump(categories_map, f, indent=4)


import matplotlib.pyplot as plt
# 可视化聚类结果
plt.figure(figsize=(10, 8))
plt.scatter(movie_vectors_2d[:, 0], movie_vectors_2d[:, 1], c=cluster_labels, cmap='viridis')

# 为每个类目添加标签
for i, category in enumerate(items[COL_NAME]):
    plt.annotate(category, (movie_vectors_2d[i, 0], movie_vectors_2d[i, 1]))

plt.title("Clustering of Movie Categories using Word2Vec + KMeans")
plt.colorbar()
plt.show()

# 用不同的K值进行评估，以确定最优的聚类数（可选）
from sklearn.metrics import silhouette_score

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(movie_vectors_2d)
    score = silhouette_score(movie_vectors_2d, kmeans.labels_)
    print(f"K={k}, Silhouette Score={score}")
