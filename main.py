import torch
from model import load_model_and_tokenizer
from dataset import load_dataset_from_moses
from train import train, create_dataloader
from translate import translate_text
from utils import detect_device
from transformers import AdamW

if __name__ == "__main__":
    # Configuración y carga de modelo
    device = detect_device()
    model, tokenizer, device = load_model_and_tokenizer(device)

    # Definir rutas de los archivos Moses
    source_file = "data/en.txt"  # Archivo con texto en inglés
    target_file = "data/es.txt"  # Archivo con texto en español
    
    # Crear el DataLoader para el entrenamiento
    train_dataloader = create_dataloader(source_file, target_file, tokenizer, batch_size=16)
    
    # Entrenamiento del modelo (descomenta para entrenar)
    # optimizer = AdamW(model.parameters(), lr=5e-5)
    # train(model, train_dataloader, optimizer, device, epochs=3)

    # Traducción del texto
    text_to_translate = "Hello, how are you?"
    translated_text = translate_text(model, tokenizer, text_to_translate, "en_XX", "es_XX", device)
    
    print(f"Texto original: {text_to_translate}")
    print(f"Texto traducido: {translated_text}")
