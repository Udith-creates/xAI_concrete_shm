import gradio as gr
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

from src.model import get_model
from src.metrics import calculate_sci

# Academically verified XAI Libraries
from pytorch_grad_cam import GradCAMPlusPlus, ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from captum.attr import IntegratedGradients

# 1. Initialize the Device and Model Globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(pretrained=False, freeze_backbone=False)

try:
    model.load_state_dict(torch.load("models/best_resnet50.pt", map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print("✔ Model weights loaded successfully for the web server.")
except Exception as e:
    print(f"❌ Failed to load model weights: {e}")

# Preprocessing Pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def analyze_concrete(input_image, xai_method):
    if input_image is None:
        return None, "Please upload an image.", "N/A"
        
    # Convert Gradio image to PIL and Tensor
    pil_img = Image.fromarray(input_image).convert("RGB")
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    
    # AI Prediction
    with torch.no_grad():
        output = model(input_tensor)
        confidence_scores = torch.nn.functional.softmax(output, dim=1)[0]
        predicted_class = torch.argmax(confidence_scores).item()
        confidence = confidence_scores[predicted_class].item()
        
    status_text = "Cracked" if predicted_class == 1 else "Healthy"
    prediction_str = f"**Status:** {status_text} (Confidence: {confidence*100:.1f}%)"

    # XAI Heatmap Generation
    target_layers = [model.backbone.layer4[-1]]
    heatmap_2d = None

    if xai_method == "Grad-CAM++":
        with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
            # target_category=1 forces it to look for cracks
            heatmap_2d = cam(input_tensor=input_tensor, targets=None)[0] 
            
    elif xai_method == "Score-CAM":
        with ScoreCAM(model=model, target_layers=target_layers) as cam:
            heatmap_2d = cam(input_tensor=input_tensor, targets=None)[0]
            
    elif xai_method == "Integrated Gradients (IG)":
        ig = IntegratedGradients(model)
        # Calculate pixel attributions relative to a black baseline
        attr, delta = ig.attribute(input_tensor, target=1, return_convergence_delta=True)
        attr = attr.squeeze().cpu().detach().numpy()
        
        # Process IG output: sum absolute values across RGB channels to get a 2D intensity map
        attr = np.sum(np.abs(attr), axis=0)
        # Normalize to 0-1
        if np.max(attr) > 0:
            heatmap_2d = attr / np.max(attr)
        else:
            heatmap_2d = np.zeros((224, 224))

    # Resize heatmap to match the original image display size
    heatmap_resized = cv2.resize(heatmap_2d, (pil_img.width, pil_img.height))
    
    # THE LOGIC GATE: Only calculate defects if the AI actually detected a crack
    if predicted_class == 1:
        sci_score, defect_density, eng_status = calculate_sci(heatmap_resized, threshold=0.4, beta=2.5)
    else:
        # If it's Healthy, mathematically enforce a perfect structural score
        sci_score = 1.000
        defect_density = 0.00
        eng_status = "Negligible - Normal Maintenance"
        
        # Zero out the heatmap so the user doesn't see "ghost" highlights
        heatmap_resized = np.zeros_like(heatmap_resized)
    
    sci_report = (
        f"**Surface Defect Density:** {defect_density*100:.2f}%\n\n"
        f"**Final SCI Score:** {sci_score:.3f} / 1.000\n\n"
        f"**Engineering Action:** {eng_status}"
    )

    # Create the visual overlay
    # Normalize original image to 0-1 float for the show_cam_on_image function
    img_float = np.array(pil_img) / 255.0 
    visual_output = show_cam_on_image(img_float, heatmap_resized, use_rgb=True)

    return visual_output, prediction_str, sci_report

# 3. Build the Gradio Web Interface
with gr.Blocks(theme=gr.themes.Soft()) as interface:
    gr.Markdown("# 🏗️ X-ConcreteSHM: Explainable AI for Structural Health Monitoring")
    gr.Markdown("Upload a concrete surface image, select your mathematical interpretability method, and generate the Structural Condition Index (SCI).")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Raw Concrete Input")
            method_dropdown = gr.Dropdown(
                choices=["Grad-CAM++", "Score-CAM", "Integrated Gradients (IG)"],
                value="Grad-CAM++",
                label="Explainability Algorithm"
            )
            submit_btn = gr.Button("Analyze Structure", variant="primary")
            
        with gr.Column():
            image_output = gr.Image(label="XAI Defect Tracker")
            prediction_output = gr.Markdown(label="AI Diagnosis")
            sci_output = gr.Markdown(label="Structural Condition Index (SCI)")

    submit_btn.click(
        fn=analyze_concrete,
        inputs=[image_input, method_dropdown],
        outputs=[image_output, prediction_output, sci_output]
    )

if __name__ == "__main__":
    interface.launch(share=False)