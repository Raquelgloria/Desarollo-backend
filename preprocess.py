from datasets import load_dataset

def preprocess_dataset(dataset_name, source_lang, target_lang, tokenizer, max_length=128):
    """
    Preprocesa el dataset multilingüe para el entrenamiento.
    """
    dataset = load_dataset(dataset_name, f"{source_lang[:2]}-{target_lang[:2]}", split="train")

    def preprocess_function(examples):
        inputs = tokenizer(
            examples["translation"][source_lang[:2]],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )
        targets = tokenizer(
            examples["translation"][target_lang[:2]],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )
        inputs["labels"] = targets["input_ids"]
        return inputs

    return dataset.map(preprocess_function, batched=True)
