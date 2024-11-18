from transformers import MBartForConditionalGeneration, MBart50Tokenizer

def load_model_and_tokenizer(model_name="facebook/mbart-large-50-many-to-many-mmt"):
    """
    Carga el modelo MBART y su tokenizador.
    
    Args:
    - model_name: el nombre del modelo preentrenado (por defecto "facebook/mbart-large-50-many-to-many-mmt").
    
    Returns:
    - model: el modelo MBART cargado.
    - tokenizer: el tokenizador correspondiente al modelo.
    """
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = MBart50Tokenizer.from_pretrained(model_name)
    
    return model, tokenizer
