from transformers import MBartForConditionalGeneration, MBart50Tokenizer

def load_model_and_tokenizer(model_name="facebook/mbart-large-50-many-to-many-mmt", load_dir=None):
    """
    Carga un modelo MBART y su tokenizador. Si `load_dir` está definido, carga desde esa carpeta.
    """
    if load_dir:
        print(f"Cargando modelo desde {load_dir}...")
        model = MBartForConditionalGeneration.from_pretrained(load_dir)
        tokenizer = MBart50Tokenizer.from_pretrained(load_dir)
    else:
        print(f"Cargando modelo preentrenado {model_name}...")
        model = MBartForConditionalGeneration.from_pretrained(model_name)
        tokenizer = MBart50Tokenizer.from_pretrained(model_name)

    return model, tokenizer
