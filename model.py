from transformers import MBartForConditionalGeneration, MBart50Tokenizer

def load_model_and_tokenizer(model_name, source_lang):
    """
    Carga el modelo y el tokenizador, configurando el idioma fuente.
    """
    tokenizer = MBart50Tokenizer.from_pretrained(model_name)
    tokenizer.src_lang = source_lang  # Idioma fuente
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    return model, tokenizer
