from transformers import MBartForConditionalGeneration, MBart50Tokenizer

def load_model_and_tokenizer(model_name="facebook/mbart-large-50-many-to-many-mmt"):
    """
    Carga el modelo MBART y el tokenizador.
    """
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = MBart50Tokenizer.from_pretrained(model_name)
    return model, tokenizer
