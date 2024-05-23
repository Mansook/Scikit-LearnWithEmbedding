import torch
import torch.nn as nn
import pandas as pd
from kobert_transformers import get_kobert_model
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score


df = pd.read_excel('embeddings.xlsx', header=None)
labels = pd.read_excel('labels.xlsx', header=None)

print(len(df))
print(len(labels))
