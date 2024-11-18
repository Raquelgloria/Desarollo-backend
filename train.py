import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
from preprocess import preprocess_data
from dataset import load_dataset_from_moses

def create_dataloader(source_file, target_file, tokenizer, batch_size=16):
    """
    Crea el DataLoader para entrenamiento.
    """
    dataset = load_dataset_from_moses(source_file, target_file)
    preprocessed_data = preprocess_data(dataset, tokenizer)
    
    return DataLoader(preprocessed_data, batch_size=batch_size, shuffle=True)

def train(model, train_dataloader, optimizer, device, epochs=3):
    """
    Entrena el modelo MBART.
    """
    model.train()
    for epoch in range(epochs):
        loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in loop:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())
