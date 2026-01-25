"""
Test script to verify gradient computation for G-NDF models.
"""
import torch
import numpy as np
from common.models import load_model

def test_gradient_computation():
    """Test that gradients are correctly computed through FK layer."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the smoothed G-NDF model
    model_path = "output/block_world5_SLL_smooth/model_final.pt"
    model = load_model(model_path, device)
    model.eval()
    
    # Test configuration
    q_test = torch.tensor([[0.0, 0.0]], dtype=torch.float32, device=device, requires_grad=True)
    
    # Forward pass
    ndf_value = model(q_test)
    
    # Backward pass
    ndf_value.backward()
    
    # Check gradient
    grad_q = q_test.grad
    
    print(f"Test configuration: q = {q_test.detach().cpu().numpy()}")
    print(f"NDF value: {ndf_value.item():.6f}")
    print(f"Gradient ∂(NDF)/∂q: {grad_q.cpu().numpy()}")
    print(f"Gradient norm: {torch.norm(grad_q).item():.6f}")
    
    # Test at multiple points
    print("\n" + "="*60)
    print("Testing gradient norms across C-space:")
    print("="*60)
    
    q_samples = torch.tensor([
        [0.0, 0.0],
        [np.pi/2, 0.0],
        [np.pi, 0.0],
        [0.0, np.pi/2],
        [np.pi/2, np.pi/2],
    ], dtype=torch.float32, device=device)
    
    for i, q in enumerate(q_samples):
        q_input = q.unsqueeze(0).requires_grad_(True)
        ndf_val = model(q_input)
        ndf_val.backward()
        grad_norm = torch.norm(q_input.grad).item()
        print(f"q = [{q[0].item():6.3f}, {q[1].item():6.3f}] | NDF = {ndf_val.item():7.4f} | ||∇NDF|| = {grad_norm:.4f}")

if __name__ == "__main__":
    test_gradient_computation()
