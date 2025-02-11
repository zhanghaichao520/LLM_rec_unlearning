import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sentence_transformers import SentenceTransformer
import os

cache_folder = os.path.expanduser("sentence_transformers") # 设置缓存目录

try:
    model = SentenceTransformer('all-mpnet-base-v2', cache_folder=cache_folder)
    print("模型加载成功！")
except Exception as e:
    print(f"自动下载/加载模型失败: {e}")
    print("请尝试手动下载模型或检查网络连接。")