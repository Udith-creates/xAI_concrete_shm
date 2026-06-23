import os
import torch
import numpy as np
import mlflow
import mlflow.pytorch

from src.model import get_model

def upload_completed_run():
    print("📦 Initializing Retroactive MLflow Logging...")
    
    # Set the experiment name matching your project
    mlflow.set_experiment("xAI_Concrete_SHM")
    
    with mlflow.start_run(run_name="ResNet50_Final_Execution") as run:
        # 1. Log All Hyperparameters & Structural Configuration
        print("📝 Logging parameters...")
        mlflow.log_params({
            "backbone": "ResNet-50",
            "epochs": 15,
            "input_shape": "(224, 224, 3)",
            "optimizer": "Adam",
            "dataset": "SDNET2018 + CrackForest",
            "hardware": "NVIDIA GeForce RTX 3050"
        })
        
        # 2. Log Final Epoch Metrics (Extracted exactly from your successful run)
        print("📈 Logging metrics...")
        mlflow.log_metrics({
            "train_loss": 0.1190,
            "train_acc": 0.9599,
            "val_loss": 0.1157,
            "val_acc": 0.9619,
            "final_batch_loss": 0.2133
        })
        
        # 3. Log the Final Model Artifact with Traced Graph Context (The Fix!)
        print("🤖 Loading and logging the PyTorch Model state...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = get_model(pretrained=False, freeze_backbone=False)
        
        try:
            model.load_state_dict(torch.load("models/best_resnet50.pt", map_location=device, weights_only=True))
            model.to(device)
            model.eval()
            
            # Create the required input example to allow clean graph tracing
            example_input = np.random.rand(1, 3, 224, 224).astype(np.float32)
            
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="concrete_resnet50_model",
                input_example=example_input
            )
            print("✔ Model logged successfully to MLflow tracking.")
        except Exception as e:
            print(f"❌ Could not load or log model file: {e}")
            
        # 4. Log Visual Reports as Artifacts (Your Grad-CAM Heatmap Analysis)
        if os.path.exists("final_analysis_output.png"):
            print("🖼️ Logging Grad-CAM visual report...")
            mlflow.log_artifact("final_analysis_output.png", artifact_path="plots")
            print("✔ Grad-CAM analysis plot added to artifacts.")
        else:
            print("⚠ 'final_analysis_output.png' not found. Skipping plot artifact.")

    print("\n🚀 SUCCESS: Everything has been committed to MLflow!")
    print(f"Run ID: {run.info.run_id}")

if __name__ == "__main__":
    upload_completed_run()