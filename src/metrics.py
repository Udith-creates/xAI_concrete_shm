import numpy as np

def calculate_sci(heatmap, threshold=0.5, beta=2.5):
    """
    Calculates the Structural Condition Index (SCI) directly from Grad-CAM heatmaps.
    
    heatmap: 2D numpy array (values 0.0 to 1.0)
    threshold: Minimum activation intensity to be counted as a physical defect
    beta: Structural scaling penalty (2.5 is standard for concrete)
    """
    # 1. Isolate the defective pixels
    defect_pixels = np.sum(heatmap >= threshold)
    total_pixels = heatmap.size
    
    # 2. Calculate Surface Defect Density (Ds)
    defect_density = defect_pixels / total_pixels
    
    # 3. Apply the Exponential Decay Formula: SCI = e^(-beta * Ds)
    sci_score = float(np.exp(-beta * defect_density))
    
    # 4. Map to Civil Engineering Action States
    if sci_score >= 0.85:
        status = "Negligible - Normal Maintenance"
    elif sci_score >= 0.65:
        status = "Minor - Scheduled Monitoring"
    elif sci_score >= 0.40:
        status = "Moderate - Structural Rehabilitation Required"
    else:
        status = "Critical - Immediate Structural Intervention"
        
    return sci_score, defect_density, status