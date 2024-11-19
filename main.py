from train import train_model
from translate import translate_text

def main():
    try:
       
        entrenar = input("¿Deseas entrenar el modelo? (s/n): ").lower()

        if entrenar == "s":
            print("Entrenando el modelo...")
            train_model()

        source_text = "Are you okay?"
        translated_text = translate_text(source_text, source_lang="en_XX", target_lang="es_XX")
        
        print("Texto original:", source_text)
        print("Texto traducido:", translated_text[0])

    except Exception as e:
        print(f"Ocurrió un error: {e}")

if __name__ == "__main__":
    main()
