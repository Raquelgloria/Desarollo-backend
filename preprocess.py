from transformers import MBart50Tokenizer

def preprocess_data(dataset, tokenizer, max_length=512):
    """
    Preprocesa el dataset para convertirlo en tensores adecuados para el modelo MBART.
    """
    inputs = []
    for item in dataset:
        source = item["source"]
        target = item["target"]
        
        input_ids = tokenizer(source, return_tensors="pt", max_length=max_length, padding="max_length").input_ids.squeeze()
        target_ids = tokenizer(target, return_tensors="pt", max_length=max_length, padding="max_length").input_ids.squeeze()
        
        inputs.append({
            "input_ids": input_ids,
            "labels": target_ids
        })
    
    return inputs
