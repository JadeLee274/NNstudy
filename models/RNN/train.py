import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from nltk.tokenize import word_tokenize
from typing import *
from preprocess import *
from LSTMClassifier import *
from IMDBDAtaset import *


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--data_root",
        type=str,
        required=True,
    )
    args.add_argument(
        "--vocab_size",
        type=int,
        default=20000,
        help="The minimum of the frequency that the text words used"
    )
    args.add_argument(
        "--max_len",
        type=int,
        default=200,
        help="If the tokenized sentence length is less/more, then pad/cut"
    )
    args.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )
    args.add_argument(
        "--epochs",
        type=int,
        default=200,
    )
    args.add_argument(
        "--save_interval",
        type=int,
        default=1,
    )
    args.add_argument(
        "--checkpoint",
        type=int,
        default=0,
    )
    args.parse_args()

    dataframe = pd.read_csv(args.data_root)

    dataframe['review'] = dataframe['review'].apply(clean_text)
    dataframe['sentiment'] = dataframe['sentiment'].map(
        {'positive': 1, 'negative': 0,}
    )
    dataframe['tokens'] = dataframe['review'].apply(word_tokenize)
    
    word_to_index = sort_and_to_index(
        dataset=dataframe,
        size=args.vocab_size,
    )

    dataframe['indexed'] = dataframe['tokens'].apply(
        lambda tokens: [
            word_to_index.get(
                token, word_to_index["<OOV>"]
            ) for token in tokens
        ]
    )

    dataframe['indexed'] = dataframe['indexed'].apply(
        lambda x: pad_sequence(x, max_len=args.max_len)
    )

    train_data, test_data = train_test_split(
        dataset=dataframe,
        p=0.8
    )

    train_texts, train_labels = (
        train_data['indexed'].tolist(), train_data['sentiment'].tolist(),
    )

    test_texts, test_labels = (
        test_data['idexed'].tolist(), test_data['sentiment'].tolist(),
    )

    train_dataset = IMDBDataset(
        texts=train_texts,
        labels=train_labels,
    )

    test_dataset = IMDBDataset(
        texts=test_texts,
        labels=test_labels,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMClassifier(
        vocab_size=args.vocab_size
    ).to(device)

    criterion = nn.BCELoss()

    optimizer = optim.Adam(
        params=model.parameters(),
        lr=1e-3,
    )

    if args.checkpoint != 0:
        checkpoint = torch.load(
            f"./imdb_train_save/imdb_lstm_{args.checkpoint}.pt"
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Starting Training Loop from {args.checkpoint + 1}... \n")

    for epoch in range(args.checkpoint, args.epochs):
        model.train()
        total_loss = 0

        for texts, labels in train_loader:
            texts, lables = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} Loss = {avg_loss:.4e}")

        if (epoch + 1) % args.save_checkpoint == 0:
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                },
                f"./imdb_train_save/imdb_lstm_{epoch + 1}.pt",
            )
        print(f"Saving model at epoch {epoch + 1}")
    
    print("\nTraining Loop Finished \n\nStarting Test\n")

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts).squeeze()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    accuracy = 100 * (correct / total)

    print(f"Finished Testing with Accuracy {accuracy:.4f}%")
