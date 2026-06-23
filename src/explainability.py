import torch
import torch.nn.functional as F
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks to intercept data flowing through the network
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor, target_class=1):
        """
        Generates the spatial heatmap highlighting the structural defect.
        Target class defaults to 1 (Cracked).
        """
        self.model.eval()
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Backward pass targeting the specific class score
        score = output[:, target_class].squeeze()
        score.backward(retain_graph=True)
        
        # Spatially average the gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Multiply activations by the gradient weights
        activations = self.activations.clone()
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        # Combine channels to create the final 2D heatmap
        heatmap = torch.mean(activations, dim=1).squeeze()
        
        # ReLU: We only care about features that *positively* contribute to the crack classification
        heatmap = F.relu(heatmap) 
        
        # Normalize between 0 and 1
        heatmap_max = torch.max(heatmap)
        if heatmap_max > 0:
            heatmap /= heatmap_max
            
        return heatmap.cpu().numpy()