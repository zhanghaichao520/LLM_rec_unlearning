<div align="center">
  <img src="assets/logo.png" width="55%" alt="DeepSeek AI" />
</div>

<hr>





<p align="center">
  <a href="https://arxiv.org/abs/2511.05494" target="_blank"><img src="https://img.shields.io/badge/arXiv-2510.19600-red"></a>
  <a href='https://cragru.pages.dev/'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
  <a href="./LICENSE" target="_blank"><img src="https://img.shields.io/badge/License-MIT-blue.svg" target="_blank"></a>
  <a href="https://huggingface.co/papers/2511.05494" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Daily Papers-red"></a>


</p >


<p align="center">
<strong><big>如果您觉得我们的工作有用，请考虑给我们点个星🌟</big></strong>
</p>




## :memo: 待办事项 (TODO)

- [x] 代码发布
- [x] 论文发布
- [x] 数据集

## 📋 目录

- 基于LLM的定制化检索增强生成用于去偏推荐遗忘
  - [🔎 概览](#概览)
  - [🛠️ 安装](#安装)
  - [🚀 快速开始](#-快速开始)
  - [⚙️ 引用](#引用)


## 概览

**CRAGRU** 是一个集成了 **RAG（检索增强生成）**、**大型语言模型（LLMs）** 和 **推荐遗忘（Recommendation Unlearning）** 的统一框架。  
它支持：

- 用户级和物品级的遗忘
- 通过受控提示词设计进行去偏
- 基于 LLM 的推荐生成
- 与传统推荐模型的对比与融合
- 数据集聚类、DP 策略探索、背包问题优化等

该框架具有模块化、可复现的特点，专为灵活实验而设计。

<p align="center">
<img src="assets/framework.png" style="width: 500px" align=center>
</p>
<p align="center">
<a href="">CRAGRU 框架图。</a>        
</p>


## 📦 核心特性

🔍 RAG 增强的 LLM 推荐
结构化的提示词设计确保了可控且可解释的 LLM 推理。

🧹 高效的推荐遗忘
支持灵活移除用户交互或物品历史记录。

📈 数据集分析套件
包含聚类、统计分析和基于背包问题的优化。

🧩 模块化架构
每个阶段都可以为了研究目的轻松替换或扩展。

## 安装
DRAGRU 支持以下操作系统：

* Linux
* Windows 10
* macOS X

DRAGRU 需要 Python 3.10.12 或更高版本。

DRAGRU 需要 torch 2.5.1 或更高版本。如果您想在 GPU 上使用 DRAGRU，请参考 PyTorch 官方安装指南。

### 安装步骤
```bash
pip install -r requirements.txt
```

下载 GoogleNews-vectors-negative300.bin 并将其放入您 python 目录的库文件中。

🚀 快速开始
以下是 完整的 DRAGRU 工作流，包含 一句话解释 和 可直接运行的命令。

1️⃣ 分割 遗忘 / 保留 集
描述： 将数据集分割为 遗忘集 和 保留集，这是所有下游遗忘任务的基础。

```bash
python DRAGRU/movie-lens/dataset_split.py
```

2️⃣ 物品聚类

描述：使用 K-means + Word2Vec 执行物品聚类，为 DP 策略和提示词构建提供语义分组。

```bash
python DRAGRU/movie-lens/statistics/item_cluster.py
```

3️⃣ 构建 LLM 提示词

描述：基于保留集创建提示词文件，作为 LLM 推荐的结构化输入。

```bash
python DRAGRU/movie-lens/data_preprocess_unlearning.py
```

4️⃣ 运行 LLM 推荐

描述：使用大语言模型生成推荐结果，并可选择回退到传统模型。

```bash
python DRAGRU/movie-lens/llm_recommender.py --input prompt_file.json
```

5️⃣ 评估结果

描述：使用上一步的推荐结果计算评估指标。

```bash
python DRAGRU/movie-lens/evaluation.py --input recommender_output.json
```

## 🤝 贡献
欢迎贡献代码、提出建议和提交 Pull Request。 如有改进需求（README、可视化、脚本等），请随时提出。

## ⭐ 如果您觉得本项目有用
请考虑给此仓库点个 ⭐ 星星 —— 这是支持本项目最好的方式。

## 引用

```bibtex
@article{zhang2025customized,
  title={Customized Retrieval-Augmented Generation with LLM for Debiasing Recommendation Unlearning},
  author={Zhang, Haichao and Zhang, Chong and Hu, Peiyu and Qiu, Shi and Wang, Jia},
  journal={arXiv preprint arXiv:2511.05494},
  year={2025}
}
```

---


<div align="center">
    
<!-- [![GitHub contributors](https://img.shields.io/github/contributors/zhanghaichao520/LLM_rec_unlearning.svg)](https://github.com/zhanghaichao520/LLM_rec_unlearning/graphs/contributors) -->

[![GitHub release](https://img.shields.io/github/v/release/zhanghaichao520/LLM_rec_unlearning)](https://github.com/zhanghaichao520/LLM_rec_unlearning/releases/latest)
[![GitHub license](https://img.shields.io/github/license/zhanghaichao520/LLM_rec_unlearning?color=blue)](https://github.com/zhanghaichao520/LLM_rec_unlearning/blob/master/LICENSE)

[![GitHub stars](https://img.shields.io/github/stars/zhanghaichao520/LLM_rec_unlearning)](https://github.com/zhanghaichao520/LLM_rec_unlearning)
[![GitHub forks](https://img.shields.io/github/forks/zhanghaichao520/LLM_rec_unlearning)](https://github.com/zhanghaichao520/LLM_rec_unlearning/fork)
</div>
