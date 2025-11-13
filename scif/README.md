# SCIF Unlearning with Recbole

This README explains how to run the SCIF unlearning pipeline using Recbole. The main script is `scif.py`.

## Usage

1. **Prepare data files**  
   - Place your original `.inter` files (Netflix, ML-100K, or ML-1M) somewhere accessible (e.g., `data/netflix.inter`, `data/ml-100k.inter`, `data/ml-1m.inter`).  
   - When you first run the script, it will generate four CSVs:  
     ```
     train_normal.csv
     train_random.csv
     valid.csv
     test.csv
     ```
     in the `./scif_data` directory.  
   - It then combines them into Recbole-format files:  
     ```
     <dataset>_train.csv
     <dataset>_valid.csv
     <dataset>_test.csv
     ```
     where `<dataset>` is `netflix_scif`, `ml100k_scif`, or `ml1m_scif`.

2. **Run training + unlearning**  
   - By default, the script uses **ML-1M**. Example:  
     ```bash
     python scif.py --model LightGCN 
     ```  
   - To use **ML-100K**, add the `--use_ml100k` flag:  
     ```bash
     python scif.py --model LightGCN  --use_ml100k
     ```  
   - To use **Netflix**, add the `--use_netflix` flag:  
     ```bash
     python scif.py --model LightGCN  --use_netflix
     ```  
   - Optionally, you can switch the Recbole model to **BPR** instead of `LightGCN`:  
     ```bash
     python scif.py --model BPR 
     ```

3. **What happens**  
   - **Step 1**: Preprocess raw `.inter` into CSV, split users into “normal” vs. “to-forget” subsets.  
   - **Step 2**: Train a Recbole model (LightGCN/BPR) on the combined `train_normal + train_random`.  
   - **Step 3**: Evaluate “pre-unlearn” performance (Hit@K, NDCG@K).  
   - **Step 4**: Perform SCIF unlearning on the 10% subset (optimize a correction vector `p`).  
   - **Step 5**: Save the unlearned model and evaluate its performance on the same test split.

4. **Output**  
   - Full-model state dict: `./saved/<dataset>_full.pth`  
   - Unlearned model state dict: `./saved/<dataset>_unlearned.pth`  
   - Console logs show pre- and post-unlearning metrics.

## Example

```bash
# ML-1M (default)
python scif/scif.py --model LightGCN 

# ML-100K
python scif/scif.py --model LightGCN  --use_ml100k

# Netflix
python scif/netflix_bpr.py
python scif/netflix_LightGCN.py

# Use BPR instead of LightGCN
python scif/scif.py --model BPR 
