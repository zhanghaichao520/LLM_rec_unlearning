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


item_attr_path = os.path.join(f"{output_dir}/{DATASET}", f"{DATASET}.item")
items = pd.read_csv(item_attr_path, delimiter='\t')

# 将 'class:token_seq' 列按空格拆分成单独的类别
categories = items['class:token_seq'].str.split(' ', expand=True)

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

import gensim.downloader as api
wv = api.load('word2vec-google-news-300')
