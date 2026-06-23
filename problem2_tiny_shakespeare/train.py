import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math
import os
import urllib.request

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Problem 2: Tiny Shakespeare Dataset
# =========================

# Create a folder to save all results
os.makedirs("results", exist_ok=True)

# Download the Tiny Shakespeare dataset if it is not already saved
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
filename = "tinyshakespeare.txt"

if not os.path.exists(filename):
    urllib.request.urlretrieve(url, filename)

# Read the dataset into memory
with open(filename, "r", encoding="utf-8") as f:
    shakespeare_text = f.read()

# Use part of the dataset to keep the training time reasonable
shakespeare_text = shakespeare_text[:200000]

print("Tiny Shakespeare text length:", len(shakespeare_text))
print(shakespeare_text[:500])


# =========================
# Text Generation Function
# =========================

def generate_text(model, dataset, start_text="To be", length=200):
    model.eval()

    chars = list(start_text)

    with torch.no_grad():
        for _ in range(length):
            input_ids = torch.tensor(
                [[dataset.char_to_idx.get(ch, 0) for ch in chars[-50:]]],
                dtype=torch.long
            ).to(device)

            output = model(input_ids)
            last_output = output[:, -1, :]

            probabilities = torch.softmax(last_output, dim=-1)
            next_id = torch.multinomial(probabilities, num_samples=1).item()

            next_char = dataset.idx_to_char[next_id]
            chars.append(next_char)

    return "".join(chars)

# =========================
# Problem 2 Experiments
# LSTM and GRU with sequence lengths 20 and 30
# =========================

P2_MODEL_TYPES = ["LSTM", "GRU"]
P2_SEQ_LENGTHS = [20, 30]

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.003
EMBED_SIZE = 64
HIDDEN_SIZE = 128
NUM_LAYERS = 1

problem2_results = []

for seq_len in P2_SEQ_LENGTHS:
    dataset = CharacterDataset(shakespeare_text, seq_len)
    vocab_size = len(dataset.chars)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    for model_type in P2_MODEL_TYPES:
        print("\n====================================")
        print("Training:", model_type, "Sequence Length:", seq_len)
        print("====================================")

        model = CharModel(
            vocab_size=vocab_size,
            model_type=model_type,
            embed_size=EMBED_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        train_losses = []
        val_losses = []
        val_accuracies = []

        start_time = time.time()

        for epoch in range(EPOCHS):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_accuracy = validate_model(model, val_loader, criterion)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            print(
                f"Epoch {epoch+1:02d}/{EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_accuracy:.2f}%"
            )

        total_time = time.time() - start_time
        perplexity = math.exp(val_losses[-1])
        sample_output = generate_text(model, dataset, start_text="To be", length=200)

        problem2_results.append({
            "model": model_type,
            "sequence_length": seq_len,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "final_val_accuracy": val_accuracies[-1],
            "perplexity": perplexity,
            "training_time_sec": total_time,
            "parameters": count_parameters(model),
            "model_size_mb": model_size_mb(model),
            "approx_complexity": approximate_complexity(
                model_type,
                seq_len,
                EMBED_SIZE,
                HIDDEN_SIZE
            ),
            "generated_text": sample_output
        })

# =========================
# Problem 2 Hyperparameter Experiments
# =========================

hyperparameter_settings = [
    {"model": "LSTM", "seq_len": 30, "hidden_size": 64, "num_layers": 1},
    {"model": "LSTM", "seq_len": 30, "hidden_size": 128, "num_layers": 1},
    {"model": "LSTM", "seq_len": 30, "hidden_size": 128, "num_layers": 2},
    {"model": "GRU", "seq_len": 30, "hidden_size": 64, "num_layers": 1},
    {"model": "GRU", "seq_len": 30, "hidden_size": 128, "num_layers": 1},
    {"model": "GRU", "seq_len": 30, "hidden_size": 128, "num_layers": 2},
]

hyperparameter_results = []

for setting in hyperparameter_settings:
    model_type = setting["model"]
    seq_len = setting["seq_len"]
    hidden_size = setting["hidden_size"]
    num_layers = setting["num_layers"]

    dataset = CharacterDataset(shakespeare_text, seq_len)
    vocab_size = len(dataset.chars)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("\n====================================")
    print("Hyperparameter Test")
    print(setting)
    print("====================================")

    model = CharModel(
        vocab_size=vocab_size,
        model_type=model_type,
        embed_size=EMBED_SIZE,
        hidden_size=hidden_size,
        num_layers=num_layers
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start_time = time.time()

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_accuracy = validate_model(model, val_loader, criterion)

        print(
            f"Epoch {epoch+1:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.2f}%"
        )

    total_time = time.time() - start_time
    perplexity = math.exp(val_loss)

    # Time one quick inference run
    inference_start = time.time()
    generated = generate_text(model, dataset, start_text="To be", length=200)
    inference_time = time.time() - inference_start

    hyperparameter_results.append({
        "model": model_type,
        "sequence_length": seq_len,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "final_train_loss": train_loss,
        "final_val_loss": val_loss,
        "final_val_accuracy": val_accuracy,
        "perplexity": perplexity,
        "training_time_sec": total_time,
        "inference_time_sec": inference_time,
        "parameters": count_parameters(model),
        "model_size_mb": model_size_mb(model),
        "generated_text": generated
    })

# =========================
# Problem 2 Sequence Length 50
# =========================

seq50_results = []

for model_type in ["LSTM", "GRU"]:
    seq_len = 50

    dataset = CharacterDataset(shakespeare_text, seq_len)
    vocab_size = len(dataset.chars)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("\n====================================")
    print("Training:", model_type, "Sequence Length: 50")
    print("====================================")

    model = CharModel(
        vocab_size=vocab_size,
        model_type=model_type,
        embed_size=EMBED_SIZE,
        hidden_size=128,
        num_layers=1
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start_time = time.time()

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_accuracy = validate_model(model, val_loader, criterion)

        print(
            f"Epoch {epoch+1:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.2f}%"
        )

    total_time = time.time() - start_time
    perplexity = math.exp(val_loss)
    generated = generate_text(model, dataset, start_text="To be", length=200)

    seq50_results.append({
        "model": model_type,
        "sequence_length": 50,
        "hidden_size": 128,
        "num_layers": 1,
        "final_train_loss": train_loss,
        "final_val_loss": val_loss,
        "final_val_accuracy": val_accuracy,
        "perplexity": perplexity,
        "training_time_sec": total_time,
        "parameters": count_parameters(model),
        "model_size_mb": model_size_mb(model),
        "generated_text": generated
    })

# =========================
# Final Problem 2 Tables
# =========================

problem2_df = pd.DataFrame(problem2_results)
hyperparameter_df = pd.DataFrame(hyperparameter_results)
seq50_df = pd.DataFrame(seq50_results)

print("Problem 2 LSTM vs GRU Results")
print(problem2_df)

print("Hyperparameter Comparison")
print(hyperparameter_df)

print("Sequence Length 50 Results")
print(seq50_df)

problem2_df.to_csv("results/problem2_lstm_gru_results.csv", index=False)
hyperparameter_df.to_csv("results/problem2_hyperparameter_results.csv", index=False)
seq50_df.to_csv("results/problem2_seq50_results.csv", index=False)

# =========================
# Problem 2 Plot
# =========================

plt.figure(figsize=(10, 6))

for model_type in ["LSTM", "GRU"]:
    subset = problem2_df[problem2_df["model"] == model_type]
    plt.plot(
        subset["sequence_length"],
        subset["final_val_accuracy"],
        marker="o",
        label=model_type
    )

plt.xlabel("Sequence Length")
plt.ylabel("Validation Accuracy (%)")
plt.title("Problem 2 Validation Accuracy: LSTM vs GRU")
plt.legend()
plt.grid(True)
plt.savefig("results/problem2_lstm_gru_validation_accuracy.png")
plt.show()

# =========================
# Show Generated Text Samples
# =========================

for i, row in problem2_df.iterrows():
    print("\n====================================")
    print(f"Model: {row['model']} | Seq Length: {row['sequence_length']}")
    print("Generated Text:")
    print(row["generated_text"])
