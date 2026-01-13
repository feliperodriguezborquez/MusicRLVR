
import torch
import sys

def check_environment():
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        
    # Simple tensor operation
    x = torch.rand(5, 3).to(device)
    y = torch.rand(5, 3).to(device)
    z = x + y
    
    print("\nTensor Test (Addition):")
    print(z)
    print("\nEnvironment check passed! MusicRLVR is ready to rock.")

if __name__ == "__main__":
    check_environment()
