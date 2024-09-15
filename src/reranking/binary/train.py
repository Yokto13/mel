import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from pathlib import Path

import sys

sys.path.append("../../")

from reranking.binary.reranking_model import BinaryReranker
from utils.embeddings import create_attention_mask


def train(
    model: BinaryReranker,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    epochs: int,
    device: torch.device,
):
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1} - Training"):
            together_toks, attention_mask, labels = batch

            together_toks = together_toks.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            probs = model(together_toks, attention_mask)
            loss = criterion(probs.squeeze(), labels.float())

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1} - Validation"):
                together_toks, attention_mask, labels = batch

                together_toks = together_toks.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                probs = model(together_toks, attention_mask)

                loss = criterion(probs.squeeze(), labels.float())

                val_loss += loss.item()
                val_correct += ((probs > 0.5).squeeze() == labels).sum().item()

        val_loss /= len(val_dataloader)
        val_acc = val_correct / len(val_dataloader.dataset)

        print(
            f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}"
        )


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset_path = Path(
        "/lnet/work/home-students-external/farhan/troja/outputs/reranking_test/reranker_dataset.npz"
    )
    dataset = np.load(dataset_path)

    description_tokens = dataset["description_tokens"]
    link_tokens = dataset["link_tokens"]
    labels = dataset["y"]

    # Create attention masks
    # description_attn = create_attention_mask(description_tokens)
    # link_attn = create_attention_mask(link_tokens)

    # Convert to tensors
    description_tokens = torch.from_numpy(description_tokens)
    link_tokens = torch.from_numpy(link_tokens)
    labels = torch.from_numpy(labels)

    # Load pre-trained transformer model and tokenizer
    transformer = AutoModel.from_pretrained(
        "/lnet/work/home-students-external/farhan/troja/outputs/models/reranker"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "/lnet/work/home-students-external/farhan/troja/outputs/models/reranker"
    )

    together_tokens = []
    r_token = tokenizer.convert_tokens_to_ids("[R]")
    for desc, link in zip(description_tokens, link_tokens):
        together = torch.cat([desc, torch.tensor([r_token]), link])
        together_tokens.append(together)

    attention_mask = create_attention_mask(together_tokens)

    # Create dataset and dataloader
    dataset = TensorDataset(together_tokens, attention_mask, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    batch_size = 128
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    # Create BinaryReranker model
    model = BinaryReranker(transformer).to(device)

    # Set optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.BCELoss()

    # Train the model
    epochs = 10
    train(model, train_dataloader, val_dataloader, optimizer, criterion, epochs, device)

    # Save the trained model
    output_path = Path(
        "/lnet/work/home-students-external/farhan/troja/outputs/reranking_test/models/binary_reranker.pth"
    )
    torch.save(model.state_dict(), output_path)


if __name__ == "__main__":
    main()
