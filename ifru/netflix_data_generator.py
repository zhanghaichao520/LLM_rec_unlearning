import os
import numpy as np
import pandas as pd
import torch


class NetflixDataGenerator:
    """Netflix data generator, compatible with Data_for_LightGCN interface"""
    
    def __init__(self, args, path):
        self.args = args
        self.path = path
        
        # Load .inter file
        inter_file = path + '.inter'
        print(f"Loading file: {inter_file}")
        
        # First read the raw data to check format
        with open(inter_file, 'r') as f:
            first_line = f.readline().strip()
            print(f"First line original content: '{first_line}'")
        
        # Check if it's a merged column name format
        if 'item_id:token' in first_line and 'user_id:token' in first_line:
            print("Merged column name format detected, need to re-parse...")
            
            # Read again, skip the first line, manually set column names
            data = pd.read_csv(inter_file, sep='\t', skiprows=1, header=None)
            
            # Manually set column names
            if data.shape[1] == 4:
                data.columns = ['item', 'user', 'rating', 'timestamp']
            elif data.shape[1] == 3:
                data.columns = ['item', 'user', 'rating']
            else:
                print(f"Unexpected number of columns: {data.shape[1]}")
                # Set based on actual column count
                if data.shape[1] >= 4:
                    data.columns = ['item', 'user', 'rating', 'timestamp']
                else:
                    data.columns = ['item', 'user', 'rating']
        else:
            # Normal reading
            data = pd.read_csv(inter_file, sep='\t')
            
            # Process column name mapping
            column_map = {}
            for col in data.columns:
                if 'user' in col.lower():
                    column_map[col] = 'user'
                elif 'item' in col.lower():
                    column_map[col] = 'item'
                elif 'rating' in col.lower():
                    column_map[col] = 'rating'
                elif 'time' in col.lower():
                    column_map[col] = 'timestamp'
            
            data.rename(columns=column_map, inplace=True)
        
        print(f"Processed data shape: {data.shape}")
        print(f"Processed column names: {list(data.columns)}")
        print(f"First 5 rows of data:")
        print(data.head())
        
        # Check if rating column exists
        if 'rating' in data.columns:
            print(f"Rating distribution:")
            print(data['rating'].value_counts().sort_index())
        else:
            print("Warning: No rating column found")
            
        # Ensure user IDs and item IDs start from 0 and are continuous
        unique_users = sorted(data['user'].unique())
        unique_items = sorted(data['item'].unique())
        
        print(f"Original user ID range: {min(unique_users)} - {max(unique_users)} (total: {len(unique_users)})")
        print(f"Original item ID range: {min(unique_items)} - {max(unique_items)} (total: {len(unique_items)})")
        
        user_map = {old_id: new_id for new_id, old_id in enumerate(unique_users)}
        item_map = {old_id: new_id for new_id, old_id in enumerate(unique_items)}
        
        data['user'] = data['user'].map(user_map)
        data['item'] = data['item'].map(item_map)
        
        self.n_users = len(unique_users)
        self.n_items = len(unique_items)
        
        print(f"After re-encoding - Users: {self.n_users}, Items: {self.n_items}")
        print(f"Total interactions: {len(data)}")
        
        # Create labels - Netflix ratings 1-5, >=4 is positive
        if 'rating' in data.columns:
            # Check rating range
            min_rating = data['rating'].min()
            max_rating = data['rating'].max()
            print(f"Rating range: {min_rating} - {max_rating}")
            
            # Netflix usually has 1-5 ratings, >=4 is positive
            if max_rating <= 5:
                threshold = 4.0
            else:
                # If rating range is not 1-5, use median as threshold
                threshold = data['rating'].median()
            
            data['label'] = (data['rating'] >= threshold).astype(float)
            print(f"Using threshold: {threshold}, positive sample ratio: {data['label'].mean():.4f}")
        else:
            data['label'] = 1.0
            print("No rating column, all interactions set as positive samples")
        
        # Split training and testing sets
        self.split_data(data)
        
        # Create training data in DataFrame format
        self.train = pd.DataFrame({
            'user': self.train_users,
            'item': self.train_items,
            'label': self.train_labels
        })
        
        self.test = pd.DataFrame({
            'user': self.test_users,
            'item': self.test_items,
            'label': self.test_labels
        })
        
        # Create random training subset for forgetting experiments - Netflix data is large, select appropriate subset
        # Dynamically adjust subset size based on data scale
        subset_size = min(len(self.train) // 10, 10000)  # 10% or max 10,000
        if len(self.train) > 0:
            random_indices = np.random.choice(len(self.train), size=subset_size, replace=False)
            self.train_random = self.train.loc[random_indices].reset_index(drop=True)
        else:
            self.train_random = self.train.copy()
        
        print(f"Training set size: {len(self.train)}")
        print(f"Test set size: {len(self.test)}")
        print(f"Random training subset size: {len(self.train_random)}")
        
        # Dataset statistics
        if len(self.train) > 0:
            self.print_dataset_stats()
    
    def split_data(self, data, test_ratio=0.2):
        """Split training and test sets - optimized for large Netflix dataset"""
        np.random.seed(42)
        
        train_data = []
        test_data = []
        
        print("Starting to split training and test sets...")
        
        # Reserve some interactions for each user for testing
        user_ids = range(self.n_users)
        total_users = len(user_ids)
        
        for i, user_id in enumerate(user_ids):
            if i % 1000 == 0:
                print(f"Processing user {i}/{total_users}")
                
            user_data = data[data['user'] == user_id]
            if len(user_data) > 1:
                # For users with multiple interactions, split by ratio
                n_test = max(1, int(len(user_data) * test_ratio))
                n_test = min(n_test, len(user_data) - 1)  # Ensure at least 1 sample for training
                
                # Randomly select test samples
                test_indices = np.random.choice(len(user_data), n_test, replace=False)
                test_mask = np.zeros(len(user_data), dtype=bool)
                test_mask[test_indices] = True
                
                test_data.append(user_data[test_mask])
                train_data.append(user_data[~test_mask])
            else:
                # Users with only one interaction, put all in training set
                train_data.append(user_data)
        
        print("Merging data...")
        if train_data:
            train_df = pd.concat(train_data, ignore_index=True)
            self.train_users = train_df['user'].values
            self.train_items = train_df['item'].values
            self.train_labels = train_df['label'].values
        else:
            self.train_users = np.array([])
            self.train_items = np.array([])
            self.train_labels = np.array([])
            
        if test_data:
            test_df = pd.concat(test_data, ignore_index=True)
            self.test_users = test_df['user'].values
            self.test_items = test_df['item'].values
            self.test_labels = test_df['label'].values
        else:
            self.test_users = np.array([])
            self.test_items = np.array([])
            self.test_labels = np.array([])
        
        print("Data splitting complete")
    
    def print_dataset_stats(self):
        """Print dataset statistics"""
        print("\n==== Netflix Dataset Statistics ====")
        
        if len(self.train) == 0:
            print("Training set is empty, cannot calculate statistics")
            return
            
        # User statistics
        user_interactions = self.train['user'].value_counts()
        print(f"User interaction statistics:")
        print(f"  Average interactions: {user_interactions.mean():.2f}")
        print(f"  Median interactions: {user_interactions.median():.2f}")
        print(f"  Maximum interactions: {user_interactions.max()}")
        print(f"  Minimum interactions: {user_interactions.min()}")
        
        # Item statistics
        item_interactions = self.train['item'].value_counts()
        print(f"Item interaction statistics:")
        print(f"  Average interactions: {item_interactions.mean():.2f}")
        print(f"  Median interactions: {item_interactions.median():.2f}")
        print(f"  Maximum interactions: {item_interactions.max()}")
        print(f"  Minimum interactions: {item_interactions.min()}")
        
        # Sparsity
        total_possible = self.n_users * self.n_items
        actual_interactions = len(self.train)
        sparsity = 1 - (actual_interactions / total_possible)
        print(f"Data sparsity: {sparsity:.6f}")
        
        # Label distribution
        print(f"Training set label distribution:")
        print(f"  Positive samples: {(self.train['label'] == 1).sum()} ({(self.train['label'] == 1).mean():.4f})")
        print(f"  Negative samples: {(self.train['label'] == 0).sum()} ({(self.train['label'] == 0).mean():.4f})")
        
        if len(self.test) > 0:
            print(f"Test set label distribution:")
            print(f"  Positive samples: {(self.test['label'] == 1).sum()} ({(self.test['label'] == 1).mean():.4f})")
            print(f"  Negative samples: {(self.test['label'] == 0).sum()} ({(self.test['label'] == 0).mean():.4f})")
    
    def set_train_mode(self, mode):
        """Compatible interface"""
        print(f"Setting training mode: {mode}")
        pass


def test_netflix_data_generator():
    """Test Netflix data generator"""
    print("="*80)
    print("Testing Netflix Data Generator")
    print("="*80)
    
    # Create test parameters
    args = type('Args', (), {
        'embed_size': 32,
        'batch_size': 1024,
        'data_path': '/data/IFRU-main/Data/',
        'dataset': 'netflix',
        'attack': '',
        'data_type': 'full'
    })
    
    try:
        # Test data loading
        data_path = '/data/IFRU-main/Data/netflix'
        data_generator = NetflixDataGenerator(args, path=data_path)
        data_generator.set_train_mode(args.data_type)
        
        print(f"\n✅ Data loaded successfully!")
        print(f"Users: {data_generator.n_users}")
        print(f"Items: {data_generator.n_items}")
        print(f"Training samples: {len(data_generator.train)}")
        print(f"Test samples: {len(data_generator.test)}")
        print(f"Forgetting subset: {len(data_generator.train_random)}")
        
        # Validate data ranges
        if len(data_generator.train) > 0:
            print(f"\nData range validation:")
            print(f"User ID range: {data_generator.train['user'].min()} - {data_generator.train['user'].max()}")
            print(f"Item ID range: {data_generator.train['item'].min()} - {data_generator.train['item'].max()}")
            print(f"Label range: {data_generator.train['label'].min()} - {data_generator.train['label'].max()}")
            
            # Check data types
            print(f"\nData types:")
            print(f"train.dtypes:\n{data_generator.train.dtypes}")
        
        return data_generator
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    test_netflix_data_generator()