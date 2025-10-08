import torch

def check_device():
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device count: {torch.cuda.device_count()}")
    else:
        device = torch.device("cpu")
        print("Running on CPU")
    
    # Example operation to check device usage
    x = torch.tensor([1.0, 2.0, 3.0])  # Simple tensor creation
    x = x.to(device)  # Move tensor to the selected device

    print(f"Tensor is on: {x.device}")
    
if __name__ == "__main__":
    check_device()
