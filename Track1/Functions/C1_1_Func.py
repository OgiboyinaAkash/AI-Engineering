import random

import torch
import torch.nn as nn


class NextWordModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(context_size * embedding_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.shape[0], -1)
        out = self.fc(x)
        return out


def predict_next(context_words, model, vocab, word_to_index, index_to_word):
    if any(word not in word_to_index for word in context_words):
        return random.choice(vocab)

    context_idx = torch.tensor([[word_to_index[word] for word in context_words]])
    output = model(context_idx)
    probabilities = torch.softmax(output[0], dim=0)
    predicted_idx = torch.multinomial(probabilities, num_samples=1).item()
    return index_to_word[predicted_idx]
