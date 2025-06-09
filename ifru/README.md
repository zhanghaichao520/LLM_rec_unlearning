The implementation supports:

Models: BPR, LightGCN
Datasets: MovieLens-1M, MovieLens-100K, Netflix
Metrics: AUC, Hit Ratio, NDCG

Project Structure

IFRU-main/
├── Data/                    # Dataset directory
│   ├── ml-1m/              # MovieLens 1M dataset
│   ├── ml-100k/            # MovieLens 100K dataset
│   └── netflix/            # Netflix dataset
├── Model/                   # Model implementations
│   ├── bpr.py              # BPR model
│   └── Lightgcn.py         # LightGCN model
├── utility/                 # Utility functions
│   ├── compute.py          # Computation utilities
│   └── load_data.py        # Data loading utilities
├── srcqs/                   # Extended functionality
│   └── netflix_data_generator.py  # Netflix data processor
├── quick_train_*.py         # Quick model training scripts
└── full_*_ifru_*.py         # IFRU implementation scripts

Requirements

Python 3.8+
PyTorch 1.8+
NumPy
Pandas
SciPy
scikit-learn

Workflow
The workflow consists of two main stages:

1.  Train recommendation models using quick training scripts
2.  Apply IFRU to unlearn specific user data using the full IFRU scripts

Step 1: Train Recommendation Models
Select the appropriate quick training script based on your dataset and model:

For BPR model:

# For MovieLens-1M dataset
python srcqs/quick_train_bpr_1m.py

# For MovieLens-100K dataset  
python srcqs/quick_train_bpr_100k.py

# For Netflix dataset
python srcqs/quick_train_bpr_netflix.py

For LightGCN model:

# For MovieLens-1M dataset
python srcqs/quick_train_lightgcn_ml1m.py

# For MovieLens-100K dataset
python srcqs/quick_train_lightgcn_ml100k.py

# For Netflix dataset  
python srcqs/quick_train_lightgcn_netflix.py

These scripts will train models and save the weights to the ./Weights/ directory.

Step 2: Run IFRU to Unlearn User Data
After training, use the IFRU scripts to unlearn specific user interactions:

For BPR model:

# For MovieLens-1M dataset
python full_bpr_ifru_1m.py --model_path=./Weights/BPR/Quick_BPR_ml1m.pth

# For MovieLens-100K dataset with optimized parameters
python full_bpr_ifru_100k.py --cg_iterations=800 --cg_min_iter=40 --damping=1e-3 --if_epoch=500 --eval_interval=50

# For Netflix dataset with optimized parameters
python full_bpr_ifru_netflix.py --cg_iterations=800 --cg_min_iter=40 --damping=1e-3 --if_epoch=500 --eval_interval=50

For LightGCN model:

# For MovieLens-1M dataset
python full_lightgcn_ifru_ml1m.py --model_path=./Weights/LightGCN/Quick_LightGCN_ml1m.pth

# For MovieLens-100K dataset
python full_lightgcn_ifru_ml100k.py --model_path=./Weights/LightGCN/Quick_LightGCN_ml100k.pth

# For Netflix dataset
python full_lightgcn_ifru_netflix.py --model_path=./Weights/LightGCN/Quick_LightGCN_netflix.pth


GitHub Copilot
IFRU: Influence Function Recommendation Unlearning
Overview
This repository implements Influence Function Recommendation Unlearning (IFRU), a machine unlearning method for recommender systems. IFRU can effectively remove specific user data from trained recommendation models without requiring full retraining, preserving model utility while respecting data deletion requests.

The implementation supports:

Models: BPR, LightGCN
Datasets: MovieLens-1M, MovieLens-100K, Netflix
Metrics: AUC, Hit Ratio, NDCG
Project Structure
utilities
Requirements
Python 3.8+
PyTorch 1.8+
NumPy
Pandas
SciPy
scikit-learn
Data Preparation
The code expects datasets to be in appropriate directories with .inter files. Each dataset should contain user-item interactions with:

User IDs
Item IDs
Ratings (optional)
Timestamps (optional)
For example, for the Netflix dataset:

Workflow
The workflow consists of two main stages:

Train recommendation models using quick training scripts
Apply IFRU to unlearn specific user data using the full IFRU scripts
Step 1: Train Recommendation Models
Select the appropriate quick training script based on your dataset and model:

For BPR model:

For LightGCN model:

These scripts will train models and save the weights to the ./Weights/ directory.

Step 2: Run IFRU to Unlearn User Data
After training, use the IFRU scripts to unlearn specific user interactions:

For BPR model:

For LightGCN model:

Important Parameters
--if_lr: Learning rate for IFRU (default: 5e-5)
--if_epoch: Number of IFRU training epochs (default: 2000)
--k_hop: Number of neighbor hops (default: 1)
--damping: Hessian damping coefficient (default: 1e-4)
--cg_iterations: Maximum iterations for conjugate gradient (default: 1000)
--cg_min_iter: Minimum iterations for conjugate gradient (default: 50)
--eval_interval: Evaluation interval during unlearning (default: 100)

Optimized Parameters for Large Datasets
For Netflix and ML-100K datasets, use these optimized parameters to improve performance:

python full_bpr_ifru_netflix.py --cg_iterations=800 --cg_min_iter=40 --damping=1e-3 --if_epoch=500 --eval_interval=50

python full_bpr_ifru_100k.py --cg_iterations=800 --cg_min_iter=40 --damping=1e-3 --if_epoch=500 --eval_interval=50

These parameters provide a good trade-off between unlearning effectiveness and computational efficiency.
