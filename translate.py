import torch

def translate_text(model, tokenizer, text, source_lang, target_lang, device):
    """
    Traduce un texto usando el modelo MBART entrenado.
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    model.config.src_lang = source_lang
    model.config.tgt_lang = target_lang

    with torch.no_grad():
        generated_ids = model.generate(
            inputs["input_ids"],
            decoder_start_token_id=tokenizer.lang_code_to_id[target_lang],
            max_length=512
        )
    
    translated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return translated_text
