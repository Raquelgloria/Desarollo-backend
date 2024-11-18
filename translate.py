import yaml
from model import load_model_and_tokenizer

def translate(text, target_lang):
    # Cargar configuración
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Cargar modelo y tokenizador
    model, tokenizer = load_model_and_tokenizer(config["model_name"], config["languages"]["source"])

    # Tokenizar la entrada
    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    # Generar la traducción
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[target_lang]
    )
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text

if __name__ == "__main__":
    text = "Once upon a time, there was a little girl named Goldilocks."
    print("Traducción:", translate(text, "es_XX"))
