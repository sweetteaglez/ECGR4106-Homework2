# =========================
# ECGR 4106 Homework 2
# Name: Samantha Gonzalez
# Student ID: 801353957
# Character Prediction with RNN, LSTM, and GRU
# =========================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# =========================
# Problem 1 Text Sequence
# =========================

problem1_text = """
Next character prediction is a fundamental task in the field of natural language processing (NLP) that involves predicting the next character in a sequence of text based on the characters that precede it. This task is essential for various applications, including text auto-completion, spell checking, and even in the development of sophisticated AI models capable of generating human-like text.

At its core, next character prediction relies on statistical models or deep learning algorithms to analyze a given sequence of text and predict which character is most likely to follow. These predictions are based on patterns and relationships learned from large datasets of text during the training phase of the model.

One of the most popular approaches to next character prediction involves the use of Recurrent Neural Networks (RNNs), and more specifically, a variant called Long Short-Term Memory (LSTM) networks. RNNs are particularly well-suited for sequential data like text, as they can maintain information in 'memory' about previous characters to inform the prediction of the next character. LSTM networks enhance this capability by being able to remember long-term dependencies, making them even more effective for next character prediction tasks.

Training a model for next character prediction involves feeding it large amounts of text data, allowing it to learn the probability of each character's appearance following a sequence of characters. During this training process, the model adjusts its parameters to minimize the difference between its predictions and the actual outcomes, thus improving its predictive accuracy over time.

Once trained, the model can be used to predict the next character in a given piece of text by considering the sequence of characters that precede it. This can enhance user experience in text editing software, improve efficiency in coding environments with auto-completion features, and enable more natural interactions with AI-based chatbots and virtual assistants.

In summary, next character prediction plays a crucial role in enhancing the capabilities of various NLP applications, making text-based interactions more efficient, accurate, and human-like. Through the use of advanced machine learning models like RNNs and LSTMs, next character prediction continues to evolve, opening new possibilities for the future of text-based technology.
"""

print("Text length:", len(problem1_text))

# =========================
# Dataset and Model Classes
# =========================

class CharacterDataset(Dataset):
    def __init__(self, text, seq_length):
        self.seq_length = seq_length

        # Create the character vocabulary for this text
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}

        # Convert each character in the text into its number index
        self.data = torch.tensor(
            [self.char_to_idx[ch] for ch in text],
            dtype=torch.long
        )

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_length]
        y = self.data[idx+1:idx+self.seq_length+1]
        return x, y


class CharModel(nn.Module):
    def __init__(self, vocab_size, model_type="RNN", embed_size=64, hidden_size=128, num_layers=1):
        super(CharModel, self).__init__()

        self.model_type = model_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size)

        if model_type == "RNN":
            self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        elif model_type == "LSTM":
            self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        elif model_type == "GRU":
            self.rnn = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError("model_type must be RNN, LSTM, or GRU")

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, hidden = self.rnn(x)
        out = self.fc(out)
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    return param_size / (1024 ** 2)


def approximate_complexity(model_type, seq_len, embed_size, hidden_size):
    # This is a simple complexity estimate used only for comparison
    if model_type == "RNN":
        factor = 1
    elif model_type == "GRU":
        factor = 3
    elif model_type == "LSTM":
        factor = 4

    return factor * seq_len * hidden_size * (embed_size + hidden_size)

# =========================
# Training and Validation Functions
# =========================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        outputs = model(x)

        loss = criterion(
            outputs.reshape(-1, outputs.shape[-1]),
            y.reshape(-1)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate_model(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)

            loss = criterion(
                outputs.reshape(-1, outputs.shape[-1]),
                y.reshape(-1)
            )

            total_loss += loss.item()

            predictions = torch.argmax(outputs, dim=-1)
            correct += (predictions == y).sum().item()
            total += y.numel()

    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy

# =========================
# Problem 1 Experiments
# =========================

SEQ_LENGTHS = [10, 20, 30]
MODEL_TYPES = ["RNN", "LSTM", "GRU"]

BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.005
EMBED_SIZE = 64
HIDDEN_SIZE = 128

problem1_results = []

for seq_len in SEQ_LENGTHS:
    dataset = CharacterDataset(problem1_text, seq_len)
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

    for model_type in MODEL_TYPES:
        print("\n====================================")
        print("Training:", model_type, "Sequence Length:", seq_len)
        print("====================================")

        model = CharModel(
            vocab_size=vocab_size,
            model_type=model_type,
            embed_size=EMBED_SIZE,
            hidden_size=HIDDEN_SIZE
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

        problem1_results.append({
            "model": model_type,
            "sequence_length": seq_len,
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "final_val_accuracy": val_accuracies[-1],
            "training_time_sec": total_time,
            "parameters": count_parameters(model),
            "model_size_mb": model_size_mb(model),
            "approx_complexity": approximate_complexity(
                model_type,
                seq_len,
                EMBED_SIZE,
                HIDDEN_SIZE
            )
        })

# =========================
# Problem 1 Results Table
# =========================

problem1_df = pd.DataFrame(problem1_results)
problem1_df

# Save Problem 1 results
problem1_df.to_csv("results/problem1_results.csv", index=False)

plt.figure(figsize=(10, 6))
for model_type in MODEL_TYPES:
    subset = problem1_df[problem1_df["model"] == model_type]
    plt.plot(
        subset["sequence_length"],
        subset["final_val_accuracy"],
        marker="o",
        label=model_type
    )

plt.xlabel("Sequence Length")
plt.ylabel("Validation Accuracy (%)")
plt.title("Problem 1 Validation Accuracy Comparison")
plt.legend()
plt.grid(True)
plt.savefig("results/problem1_validation_accuracy_comparison.png")
plt.show()
