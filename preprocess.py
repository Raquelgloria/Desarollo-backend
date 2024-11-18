from transformers import MBart50Tokenizer
import torch

def preprocess_data(data, tokenizer, source_lang="en_XX", target_lang="es_XX"):
    """
    Preprocesa los datos para la entrada del modelo MBART.
    
    Args:
    - data: lista de diccionarios con "source" y "target".
    - tokenizer: el tokenizador MBART.
    - source_lang: el idioma fuente.
    - target_lang: el idioma destino.
    
    Returns:
    - inputs: tensores de entrada listos para pasar al modelo.
    """
    inputs = []
    for pair in data:
        source_text = pair['source']
        target_text = pair['target']
        
        # Tokenización
        source_tokens = tokenizer(source_text, truncation=True, padding='max_length', max_length=512, return_tensors="pt")
        target_tokens = tokenizer(target_text, truncation=True, padding='max_length', max_length=512, return_tensors="pt")

        # Crear un diccionario con los tensores de entrada y salida
        inputs.append({
            'input_ids': source_tokens['input_ids'].squeeze(0),
            'attention_mask': source_tokens['attention_mask'].squeeze(0),
            'labels': target_tokens['input_ids'].squeeze(0)
        })
    
    return inputs
