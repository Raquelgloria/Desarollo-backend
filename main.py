from train import train_model
from translate import translate_text
from model import load_model_and_tokenizer
from datasets import Dataset

def main():
    try:
        # Entrenamiento del modelo
        print("Entrenando el modelo...")
        train_model()  # Aquí se entrena el modelo MBART con el dataset.

        # Cargar el modelo entrenado
        print("Cargando el modelo para traducción...")
        model, tokenizer = load_model_and_tokenizer()  # Carga el modelo MBART y el tokenizador.

        # Ejemplo de traducción
        source_text = "Hello, how are you?"
        translated_text = translate_text(model, tokenizer, source_text, source_lang="en_XX", target_lang="es_XX")

        print("Texto original:", source_text)
        print("Texto traducido:", translated_text)

    except Exception as e:
        print(f"Ocurrió un error: {e}")

if __name__ == "__main__":
    main()
