import os
import yaml
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import mlflow
import numpy as np

from src.model import get_model
from src.data_pipeline import get_data_loaders

def train_model():
    # 1. Load Configurations
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    epochs = config["train"]["epochs"]
    learning_rate = config["train"]["learning_rate"]
    
    # 2. Hardware Optimization Setup (Crucial for RTX 3050 6GB)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # Enable CuDNN benchmark for faster convolutions on fixed-size inputs (224x224)
        torch.backends.cudnn.benchmark = True
        print(f"🚀 Hardware Accelerator Detected: {torch.cuda.get_device_name(0)}")
        print(f"   └─ Mixed Precision (AMP) Enabled for maximum speed.")
    else:
        print("⚠ WARNING: CUDA not detected. Falling back to extremely slow CPU training.")

    # 3. Initialize Data, Model, Loss, and Optimizer
    train_loader, val_loader, _ = get_data_loaders(config_path="params.yaml")
    
    # Load ResNet-50 and send to GPU
    model = get_model(pretrained=True, freeze_backbone=False).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize the GradScaler for Mixed Precision Training
    scaler = torch.amp.GradScaler()

    # 4. Initialize MLflow Tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("X-ConcreteSHM-ResNet50")
    
    with mlflow.start_run(run_name="Optimized_AMP_Training"):
        # Log all hyperparameters
        mlflow.log_params(config["train"])
        mlflow.log_params({"dataset_target_size": config["data"]["target_size"]})
        
        best_val_acc = 0.0
        
        print("\n" + "="*50)
        print(f"🔥 STARTING TRAINING LOOP ({epochs} EPOCHS)")
        print("="*50)
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # --- TRAINING PHASE ---
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            # Use tqdm for a beautiful progress bar in the terminal
            train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            
            for inputs, labels in train_bar:
                # Move tensors to GPU immediately
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True) # Slightly faster than standard zero_grad
                
                # Forward pass with Automatic Mixed Precision (AMP)
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                # Backward pass and optimization using the Scaler
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                train_bar.set_postfix({'Loss': f"{loss.item():.4f}"})
                
            train_loss = running_loss / total
            train_acc = correct / total
            
            # --- VALIDATION PHASE ---
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]  ")
            
            with torch.no_grad():
                for inputs, labels in val_bar:
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
                    
            val_loss = val_loss / val_total
            val_acc = val_correct / val_total
            epoch_duration = time.time() - epoch_start_time
            
            # Print epoch summary
            print(f"📈 Epoch {epoch+1} Summary | Time: {epoch_duration:.1f}s")
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}\n")
            
            # Log metrics to MLflow
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            }, step=epoch)
            
            # Save the best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                os.makedirs("models", exist_ok=True)
                torch.save(model.state_dict(), "models/best_resnet50.pt")
                print("   [💾 Saved new best model weights]")

        # Log the final model artifact directly to MLflow
# Generate a dummy input to show MLflow the exact shape the model expects
        example_input = np.random.rand(1, 3, 224, 224).astype(np.float32)
        
        # Log the final model artifact directly to MLflow with the example
        mlflow.pytorch.log_model(
            model, 
            artifact_path="resnet50_model",
            input_example=example_input
        )

if __name__ == "__main__":
    train_model()