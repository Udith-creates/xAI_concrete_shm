import os
import glob
from PIL import Image

def verify_dataset_structure():
    base_dir = "data/raw"
    datasets = {
        "CrackForest": os.path.join(base_dir, "CrackForest", "image"),
        "Mendeley Positive": os.path.join(base_dir, "Mendeley", "Positive"),
        "Mendeley Negative": os.path.join(base_dir, "Mendeley", "Negative"),
        "SDNET2018": os.path.join(base_dir, "SDNET2018")
    }
    
    print("=== X-ConcreteSHM Dataset Ingestion Report ===")
    for name, path in datasets.items():
        if os.path.exists(path):
            # Scan for standard image extensions
            extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG')
            files = []
            for ext in extensions:
                files.extend(glob.glob(os.path.join(path, "**", ext), recursive=True))
            
            print(f"✔ {name}: Found {len(files)} source images at '{path}'")
            
            # Spot check image integrity
            if len(files) > 0:
                try:
                    with Image.open(files[0]) as img:
                        print(f"  └─ Sample Integrity Pass: {os.path.basename(files[0])} | Format: {img.format} | Size: {img.size}")
                except Exception as e:
                    print(f"  └─ ⚠ Integrity Warning on sample file: {e}")
        else:
            print(f"❌ {name}: Directory not found at '{path}'. Please check placement.")

if __name__ == "__main__":
    verify_dataset_structure()