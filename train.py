import torch
from transformers import MBartForConditionalGeneration, MBart50Tokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, EarlyStoppingCallback
from dataset import get_dataset
from utils import tokenize_function
import os


def train_model(model_name="facebook/mbart-large-50-many-to-many-mmt",save_dir="./saved_model"):
    """
    Entrena el modelo MBART con el dataset.
    """
    # Obtener el dataset
    dataset = get_dataset()

    # Dividir el dataset en entrenamiento y validación (80% / 20%)
    train_dataset = dataset.train_test_split(test_size=0.2)["train"]
    eval_dataset = dataset.train_test_split(test_size=0.2)["test"]

    # Cargar modelo y tokenizador
    tokenizer = MBart50Tokenizer.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name)

    # Tokenizar el dataset
    tokenized_train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True
    )
    tokenized_eval_dataset = eval_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True
    )

    # Configurar el data collator para el Seq2SeqTrainer
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Configurar argumentos de entrenamiento
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        eval_strategy="no", # Sin evaluación , puede cambiarse a epoch

        #puede cambiarse a 3e-5 o 5e-5
        learning_rate=3e-5,

        #Modificar para agilizar entrenamiento  , 4 4 2 = 1 hra 30 min  .2 2 2 = 3hrs aprox
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=2, 
        gradient_accumulation_steps=2,  # Acumula gradientes durante 2 pasos antes de actualizarlos 


        weight_decay=0.01,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=True,
        dataloader_num_workers=4,
        logging_steps=100
        #fp16=torch.cuda.is_available()
    )

    # Crear y ejecutar el entrenador
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,  # Especificar el dataset de evaluación
        data_collator=data_collator,  # Usar el data collator
        processing_class=tokenizer,  # Usar el tokenizer
    )

    trainer.train()

  # Guardar el modelo entrenado
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"Guardando modelo en {save_dir}...")
    trainer.save_model(save_dir)  # Guarda el modelo, el tokenizador y el estado
    tokenizer.save_pretrained(save_dir)

    print("Entrenamiento finalizado y modelo guardado.")