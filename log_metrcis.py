import torch
import numpy as np
import mlflow
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    cohen_kappa_score, 
    roc_auc_score
)

from src.model import get_model
from src.data_pipeline import get_data_loaders

def log_comprehensive_metrics():
    print("📦 Initializing Comprehensive MLflow Evaluation...")
    
    # 1. Setup Device and Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(pretrained=False, freeze_backbone=False)
    
    try:
        model.load_state_dict(torch.load("models/best_resnet50.pt", map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        print("✔ Loaded model weights successfully.")
    except Exception as e:
        print(f"❌ Error loading model weights: {e}")
        return

    # 2. Get Test Loader
    _, _, test_loader = get_data_loaders()
    
    raw_probabilities = []
    all_preds = []
    all_labels = []

    print("🔄 Gathering predictions across the 10% holdout test set...")
    
    # 3. Collect Raw Outputs and True Labels
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Get probabilities via Softmax for ROC-AUC
            probs = torch.nn.functional.softmax(outputs, dim=1)
            # Probability of the positive class (Class 1: Cracked)
            crack_probs = probs[:, 1] 
            
            _, preds = torch.max(outputs, 1)
            
            raw_probabilities.extend(crack_probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 4. Calculate Mathematical Evaluation Matrix
    print("🧮 Calculating academic evaluation metrics...")
    
    metrics_dict = {
        "test_accuracy": accuracy_score(all_labels, all_preds),
        "test_precision": precision_score(all_labels, all_preds, pos_label=1),
        "test_recall_sensitivity": recall_score(all_labels, all_preds, pos_label=1),
        "test_f1_score": f1_score(all_labels, all_preds, pos_label=1),
        "test_specificity": recall_score(all_labels, all_preds, pos_label=0), # Specificity is Recall of the negative class
        "test_cohen_kappa": cohen_kappa_score(all_labels, all_preds),
        "test_roc_auc": roc_auc_score(all_labels, raw_probabilities)
    }

    # 5. Log Everything Cleanly to MLflow
    mlflow.set_experiment("xAI_Concrete_SHM")
    
    with mlflow.start_run(run_name="Comprehensive_Academic_Evaluation") as run:
        print(f"📝 Logging metrics to MLflow Run ID: {run.info.run_id}")
        
        # Log all calculated metrics
        mlflow.log_metrics(metrics_dict)
        
        # Log metadata parameters for study tracking
        mlflow.log_params({
            "evaluation_type": "Holdout Test Set",
            "positive_class": "Cracked (1)",
            "negative_class": "Healthy (0)"
        })
        
    print("\n🚀 SUCCESS: Full suite of evaluation metrics committed to MLflow!")
    for metric_name, value in metrics_dict.items():
        print(f"   └─ {metric_name.replace('_', ' ').title()}: {value:.4f}")

if __name__ == "__main__":
    log_comprehensive_metrics()