from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from dataset import load_dataset_from_moses
from preprocess import preprocess_data
from model import load_model_and_tokenizer
from train import train_model
from translate import translate
from utils import check_device

def main():
    source_file = 'data/en.txt'
    target_file = 'data/es.txt'

    # Cargar y preprocesar los datos
    data = load_dataset_from_moses(source_file, target_file)
    tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    
    # Preprocesar los datos
    preprocessed_data = preprocess_data(data, tokenizer)

    # Cargar el modelo MBART
    model, tokenizer = load_model_and_tokenizer("facebook/mbart-large-50-many-to-many-mmt")
    device = check_device()
    model.to(device)

    # Entrenar el modelo
    train_model(model, tokenizer, preprocessed_data, epochs=3)

    # Traducir una frase
    text = "Hello, how are you?"
    translated_text = translate(model, tokenizer, text, source_lang="en_XX", target_lang="es_XX")
    print(f"Translated Text: {translated_text}")

if __name__ == '__main__':
    main()
