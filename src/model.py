import torch
import torch.nn as nn
import torchvision.models as models

class ConcreteHealthClassifier(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=False):
        super(ConcreteHealthClassifier, self).__init__()
        
        # Load the standard ResNet-50 backbone
        if pretrained:
            weights = models.ResNet50_Weights.DEFAULT
        else:
            weights = None
            
        self.backbone = models.resnet50(weights=weights)
        
        # Freeze early features if requested to prevent forgetting low-level textures
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # Intercept the original 1000-class linear layer input dimension
        in_features = self.backbone.fc.in_features
        
        # Replace the final fully connected layer with our binary head
        # 0: Healthy Concrete, 1: Structural Defect/Crack
        self.backbone.fc = nn.Linear(in_features, 2)

    def forward(self, x):
        return self.backbone(x)

def get_model(pretrained=True, freeze_backbone=False):
    """
    Instantiates and returns the modified ResNet-50 model.
    """
    model = ConcreteHealthClassifier(pretrained=pretrained, freeze_backbone=freeze_backbone)
    return model

if __name__ == "__main__":
    # Local architectural verification step
    model = get_model(pretrained=False)
    sample_input = torch.randn(1, 3, 224, 224)
    sample_output = model(sample_input)
    
    print("=== Neural Network Architecture Verification ===")
    print(f"✔ ResNet-50 Modification Pass!")
    print(f"  └─ Input Tensor Shape:  {sample_input.shape}")
    print(f"  └─ Output Logits Shape: {sample_output.shape} (Binary Classification)")