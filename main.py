from train import train_model
from translate import translate_text
from model import load_model_and_tokenizer
from datasets import Dataset

def main():
    try:
        # Entrenamiento del modelo
        save_dir = "./saved_model"
        entrenar = input("¿Deseas entrenar el modelo? (s/n): ").lower()

        if entrenar == "s":
            print("Entrenando el modelo...")
            train_model(save_dir=save_dir)

        # Cargar el modelo entrenado
        print("Cargando el modelo para traducción...")
        model, tokenizer = load_model_and_tokenizer(load_dir=save_dir)  # Carga el modelo MBART y el tokenizador.

        # Ejemplo de traducción
        source_text = "Hello, how are you?"
        translated_text = translate_text(model, tokenizer, source_text, source_lang="en_XX", target_lang="es_XX")

        print("Texto original:", source_text)
        print("Texto traducido:", translated_text)

    except Exception as e:
        print(f"Ocurrió un error: {e}")

if __name__ == "__main__":
    main()
