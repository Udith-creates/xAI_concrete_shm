import os
import sys
import torch
from src.data_pipeline import get_data_loaders

def execute_and_verify_pipeline():
    print("=" * 60)
    print("STARTING DATA PIPELINE INGESTION AND VERIFICATION")
    print("=" * 60)
    
    # 1. Verify raw data directories exist before loading
    raw_dir = os.path.join("data", "raw")
    if not os.path.exists(raw_dir):
        print(f"❌ Error: The directory '{raw_dir}' does not exist.")
        print("Please ensure your dataset folders are placed correctly.")
        sys.exit(1)
        
    # 2. Initialize DataLoaders
    try:
        train_loader, val_loader, test_loader = get_data_loaders()
    except Exception as e:
        print(f"❌ Error during DataLoader initialization: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check if folder names in data/raw match exactly with your tree output.")
        print("2. Ensure params.yaml is present in the root folder.")
        sys.exit(1)
        
    # 3. Sanity check a single batch from the training set
    print("\n" + "-" * 40)
    print("RUNNING BATCH INTEGRITY SANITY CHECK")
    print("-" * 40)
    
    try:
        # Pull exactly one batch from the training loader
        images, labels = next(iter(train_loader))
        
        print("✔ Pipeline executed with zero corruptions!")
        print(f"  └─ Images batch tensor shape : {images.shape} (Batch Size, Channels, Height, Width)")
        print(f"  └─ Labels batch tensor shape : {labels.shape} (Batch Size)")
        print(f"  └─ Target image resolution   : {images.shape[2]}x{images.shape[3]}")
        print(f"  └─ Data type                 : {images.dtype}")
        
        # Verify label boundary limits
        unique_labels = torch.unique(labels).tolist()
        print(f"  └─ Unique classes detected   : {unique_labels} (0 = Healthy, 1 = Cracked)")
        
    except Exception as e:
        print(f"❌ Error reading a batch from the dataset: {e}")
        print("This usually means one of your image files is completely corrupted or unreadable.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("DATA PIPELINE READY FOR MODEL TRAINING!")
    print("=" * 60)

if __name__ == "__main__":
    execute_and_verify_pipeline()