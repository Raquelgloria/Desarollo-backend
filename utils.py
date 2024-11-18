import torch

def check_device():
    """
    Verifica si hay una GPU disponible y retorna el dispositivo adecuado.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
