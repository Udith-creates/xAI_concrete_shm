import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from src.model import get_model
from src.explainability import GradCAM
from src.metrics import calculate_sci

def test_explainability():
    print("="*50)
    print("X-CONCRETE SHM: EXPLAINABILITY & SCI GENERATOR")
    print("="*50)

    # 1. Load the trained model weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(pretrained=False, freeze_backbone=False)
    
    try:
        # Added weights_only=True to silence the PyTorch security warning
        model.load_state_dict(torch.load("models/best_resnet50.pt", map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        print("✔ Loaded 96% Accuracy Model Weights successfully.")
    except Exception as e:
        print(f"❌ Could not load weights: {e}")
        return

    # 2. Pick a random test image from your raw data
    # (Update this path to point to any real cracked image in your dataset)
    test_image_path = "data/raw/CrackForest/image/001.jpg" 
    
    try:
        original_img = Image.open(test_image_path).convert("RGB")
    except:
        print(f"❌ Could not find test image at {test_image_path}. Please update the path in the script to a real image.")
        return

    # 3. Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(original_img).unsqueeze(0).to(device)

    # 4. Run the Model Prediction
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        class_idx = predicted.item()
        confidence = torch.nn.functional.softmax(output, dim=1)[0][class_idx].item()
    
    status_text = "Cracked" if class_idx == 1 else "Healthy"
    print(f"\n📊 AI Prediction: {status_text} (Confidence: {confidence*100:.1f}%)")

    # 5. Generate Grad-CAM Heatmap (Forcing it to run for debugging)
    print("\n🔍 Generating Grad-CAM Spatial Explanation...")
    # Hook into layer4, which is the final conv block in ResNet
    target_layer = model.backbone.layer4[-1].conv3
    grad_cam = GradCAM(model, target_layer)
    
    # We force the target_class to 1 (Cracked) so we can see what crack-like features it found
    heatmap = grad_cam.generate_heatmap(input_tensor, target_class=1)
    heatmap_resized = cv2.resize(heatmap, (original_img.width, original_img.height))
    
    # 6. Calculate Structural Condition Index (SCI)
    sci_score, defect_density, eng_status = calculate_sci(heatmap_resized, threshold=0.4, beta=2.5)
    
    print("\n🏗️ ENGINEERING ASSESSMENT:")
    print(f"   └─ Surface Defect Density: {defect_density*100:.2f}%")
    print(f"   └─ Final SCI Score:        {sci_score:.3f} / 1.000")
    print(f"   └─ Action Required:        {eng_status}")
    
    # 7. Visualization Plot
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title(f"Original Image\nAI Status: {status_text} ({confidence*100:.1f}%)")
    plt.imshow(original_img)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title(f"Grad-CAM Defect Tracker\nSCI: {sci_score:.2f} | Density: {defect_density*100:.1f}%")
    plt.imshow(original_img)
    plt.imshow(heatmap_resized, cmap='jet', alpha=0.5)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("final_analysis_output.png")
    print("\n💾 Saved visual report to 'final_analysis_output.png'")
    plt.show()

if __name__ == "__main__":
    test_explainability()