import yaml
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from preprocess import preprocess_dataset
from model import load_model_and_tokenizer

def train():
    # Cargar configuración
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Cargar modelo y tokenizador
    model, tokenizer = load_model_and_tokenizer(config["model_name"], config["languages"]["source"])

    # Preprocesar datasets para cada idioma de destino
    train_datasets = []
    val_datasets = []
    for target_lang in config["languages"]["targets"]:
        dataset = preprocess_dataset("opus_books", config["languages"]["source"], target_lang, tokenizer)
        split = dataset.train_test_split(test_size=0.2)
        train_datasets.append(split["train"])
        val_datasets.append(split["test"])

    # Combinar los datasets de entrenamiento y validación
    train_dataset = sum(train_datasets, [])
    val_dataset = sum(val_datasets, [])

    # Configurar argumentos de entrenamiento
    training_args = Seq2SeqTrainingArguments(**config["training_args"])

    # Inicializar el entrenador
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    # Entrenar
    trainer.train()

if __name__ == "__main__":
    train()
