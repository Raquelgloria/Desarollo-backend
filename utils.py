import torch

def detect_device():
    """
    Detecta si hay una GPU disponible y la utiliza, si no, usa la CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
