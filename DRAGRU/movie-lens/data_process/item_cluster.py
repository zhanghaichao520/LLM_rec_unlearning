import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# 创建保存目录
output_dir = "dataset"
DATASET = "ml-100k"
COL_NAME = "class:token_seq"

item_attr_path = os.path.join(f"{output_dir}/{DATASET}", f"{DATASET}.item")
items = pd.read_csv(item_attr_path, delimiter='\t')

# 将 'class:token_seq' 列按空格拆分成单独的类别
categories = items[COL_NAME].str.split(' ', expand=True)

# 将所有类别合并为一个长列表
categories_list = categories.stack().reset_index(drop=True)

# 统计唯一类别的数量
categories = categories_list.unique()

# 输出类别种数
print(f'Number of unique categories: {len(categories)}')
print('Unique categories:', categories)

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


from gensim.models import KeyedVectors
from gensim.test.utils import datapath
model = KeyedVectors.load_word2vec_format(datapath(r"GoogleNews-vectors-negative300.bin"), binary=True)

# 获取每个类别的词向量
def get_word_vector(word):
    try:
        return model[word]  # 获取词的向量
    except KeyError:
        return np.zeros(model.vector_size)  # 如果模型中没有该词，返回零向量

# 获取所有类目的词向量
word_embeddings = np.array([get_word_vector(category) for category in categories])

# 数据标准化（对词向量进行标准化，以便KMeans表现更好）
scaler = StandardScaler()
word_embeddings_scaled = scaler.fit_transform(word_embeddings)

# 使用PCA进行降维，以便可视化
pca = PCA(n_components=2)
word_embeddings_2d = pca.fit_transform(word_embeddings_scaled)

# 应用KMeans聚类
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(word_embeddings_2d)

# 获取每个类目的聚类标签
cluster_labels = kmeans.labels_

# 输出每个类目对应的聚类标签
for category, label in zip(categories, cluster_labels):
    print(f"Category: {category}, Cluster: {label}")

import matplotlib.pyplot as plt
# 可视化聚类结果
plt.figure(figsize=(10, 8))
plt.scatter(word_embeddings_2d[:, 0], word_embeddings_2d[:, 1], c=cluster_labels, cmap='viridis')

# 为每个类目添加标签
for i, category in enumerate(categories):
    plt.annotate(category, (word_embeddings_2d[i, 0], word_embeddings_2d[i, 1]))

plt.title("Clustering of Movie Categories using Word2Vec + KMeans")
plt.colorbar()
plt.show()

# 用不同的K值进行评估，以确定最优的聚类数（可选）
from sklearn.metrics import silhouette_score

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(word_embeddings_2d)
    score = silhouette_score(word_embeddings_2d, kmeans.labels_)
    print(f"K={k}, Silhouette Score={score}")
