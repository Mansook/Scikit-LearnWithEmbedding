import torch
import torch.nn as nn
import pandas as pd
from kobert_transformers import get_kobert_model
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
# 데이터 로드
df = pd.read_excel('embeddings.xlsx', header=None)
labels = pd.read_excel('labels.xlsx', header=None)
embeddings = df.values


# 라벨 로드 및 정수형으로 변환 (예: 'positive' -> 1, 'negative' -> 0)
# labels는 예제이므로 실제 데이터에 맞게 변환 필요

# KoBERT 모델 불러오기
kobert_model = get_kobert_model()

# 간단한 분류 모델 정의 (KoBERT + 임베딩 입력을 위한 조정)


class KoBERTClassifier(nn.Module):
    def __init__(self, bert_model, num_classes):
        super(KoBERTClassifier, self).__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(768, num_classes)  # 768은 BERT의 hidden size

    def forward(self, inputs):
        # KoBERT 모델의 forward 메서드 호출
        outputs = self.bert(inputs)
        cls_token = outputs[0][:, 0, :]  # CLS 토큰 벡터
        logits = self.classifier(cls_token)
        return logits


# 분류할 클래스의 수
num_classes = 2  # 예를 들어 이진 분류

# KoBERT 분류 모델 인스턴스 생성
model = KoBERTClassifier(kobert_model, num_classes)

# 임베딩 데이터를 PyTorch 데이터셋으로 변환


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


# 데이터셋 및 데이터로더 생성
dataset = EmbeddingDataset(embeddings, labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# 모델 학습
model.train()
for epoch in range(1):  # 에포크 수 설정
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

print("모델 학습 완료")
torch.save(model.state_dict(), 'kobert_model.pth')
print("모델 저장 완료")
