import torch

def translate_text(model, tokenizer, text, source_lang, target_lang):
    """
    Traduce texto usando MBART.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Preparar entrada
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    model.config.src_lang = source_lang
    model.config.tgt_lang = target_lang

    # Generar traducción
    with torch.no_grad():
        generated_ids = model.generate(
            inputs["input_ids"],
            max_length=512,
            decoder_start_token_id=tokenizer.lang_code_to_id[target_lang]
        )

    # Decodificar la traducción generada
    translated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return translated_text

