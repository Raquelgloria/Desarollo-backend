from transformers import MBartForConditionalGeneration, MBart50Tokenizer
import torch

def load_model_and_tokenizer(device):
    """
    Carga el modelo MBART y el tokenizador.
    """
    model_name = "facebook/mbart-large-50"
    model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer = MBart50Tokenizer.from_pretrained(model_name)
    return model, tokenizer, device
