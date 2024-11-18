import torch
def translate(model, tokenizer, text, source_lang="en_XX", target_lang="es_XX"):
    model.eval()

    # Tokenizar el texto de entrada
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = inputs.to(model.device)
    
    # Configurar los idiomas fuente y destino
    model.config.src_lang = source_lang
    model.config.tgt_lang = target_lang
    
    # Generar la traducción
    with torch.no_grad():
        translated_ids = model.generate(inputs['input_ids'], max_length=512)
    
    translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
    
    return translated_text
