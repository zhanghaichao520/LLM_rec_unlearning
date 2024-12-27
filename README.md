# LLM_rec_unlearning

## Installation
DRAGRU works with the following operating systems:

* Linux
* Windows 10
* macOS X

DRAGRU requires Python version 3.10.12 or later.

DRAGRU requires torch version 2.5.1 or later. If you want to use DRAGRU with GPU,

### Install 
```bash
pip install -r requirements.txt
```
Download [GoogleNews-vectors-negative300.bin](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) and put it in the library file of your python directory 

## Quick-Start
```plaintext
├── `SISA/`
│   ├── `LightGCN_SISA.py`    - SISA implementation for the LightGCN model
│   └── `MF_SISA.py`  - SISA implementation for the MF model
│
├── `Retrain/`
│   ├── `tradition_model.py`    - Training and testing of traditional recommendation models; model files are stored in the `saved` directory
│   └── `tradition_model_test.py`      - Testing of traditional recommendation models; requires specifying the model file in the code
│
├── `DRAGRU/`
│   ├── `llm` - Large Language Model (LLM) module
│      ├── `model_download.py` - Downloads the large model from Modelscope
│   └── `movie-lens` - Unlearning code for the MovieLens dataset based on the large model
│      ├── `data_preprocess_unlearning.py` - 2. Data preprocessing for unlearning, constructs prompts based on the forget set and remain set, and outputs the constructed prompt files
│      ├── `dataset_split.py` - 1. Splits the forget set and remain set, outputs the split datasets stored in the `dataset` directory
│      ├── `evaluation.py` - 4. Evaluation of results, generates evaluation metrics based on the recommendation results from step 3
│      ├── `llm_recommender.py` - 3. Makes recommendations based on the large model, takes prompt files as input and outputs recommendations; can also default to traditional model recommendations if LLM is not used
│   └── `data_process` - Splits datasets based on various ratios and categories; split results are stored in the `dataset/` directory for training and generating recommendations
│        └── `batch_execute.py`  - Script for batch execution of Python commands, used for running multiple training tasks at once
│   └── `statistics` - Statistical analysis of dataset metrics
│      ├── `item_cluster.py` - Item clustering (K-means + Google Word2Vec), with visualization
│      ├── `knapsack.py` - Knapsack optimization algorithm, calculates the most relevant dataset ratios per category
│      ├── `statistics.py` - Computes dataset statistics, such as average interactions per user
│
└── `config_file`  - Configuration files for running parameters of different datasets

```
# overall process of DRAGAU
1. split forget set and remain set : DRAGRU/movie-lens/dataset_split.py 
2. item cluster for DP strategy : DRAGRU/movie-lens/statistics/item_cluster.py
3. construct prompt by remain set (1) : DRAGRU/movie-lens/data_preprocess_unlearning.py
4. run llm recommandation by (3 result as input file): DRAGRU/movie-lens/llm_recommender.py
5. obtain metrics (4 result as input file) : DRAGRU/movie-lens/evaluation.py