import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm  # tqdm 라이브러리를 사용하여 진행 상태 바를 표시
import csv

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터셋 경로 및 파라미터 설정
train_data_dir = 'E:/Food101Classification-main/food101/train'
validation_data_dir = 'E:/Food101Classification-main/food101/test'
batch_size = 20
n_classes = 101

# 데이터 전처리
train_transforms = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 데이터 로더
train_dataset = ImageFolder(train_data_dir, transform=train_transforms)
val_dataset = ImageFolder(validation_data_dir, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 모델 생성
model = models.mobilenet_v2(pretrained='False')
model.classifier[1] = nn.Linear(model.last_channel, n_classes)

checkpoint_path = 'best_model.pth'
if os.path.isfile(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"체크포인트 '{checkpoint_path}'에서 모델 상태를 불러옴.")
else:
    print(f"체크포인트 '{checkpoint_path}'를 찾을 수 없습니다. 새로운 모델로 시작합니다.")



model.to(device)

# 손실 함수 및 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

# CSVLogger 정의
class CSVLogger:
    def __init__(self, filename, fieldnames):
        self.filename = filename
        self.fieldnames = fieldnames
        with open(self.filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()
    def log(self, **kwargs):
        with open(self.filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writerow(kwargs)

# ModelCheckpoint 정의
class ModelCheckpoint:
    def __init__(self, filepath, monitor='val_loss', verbose=0, save_best_only=False):
        self.filepath = filepath
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.best = float('inf')
    def save(self, model, current):
        if self.save_best_only:
            if current < self.best:
                self.best = current
                torch.save(model.state_dict(), self.filepath)
                if self.verbose:
                    print(f'Model improved from {self.best} to {current}. Saving model to {self.filepath}')
        else:
            torch.save(model.state_dict(), self.filepath)
            if self.verbose:
                print(f'Saving model to {self.filepath}')

# 로거 및 체크포인트 초기화
csv_logger = CSVLogger('history.log', fieldnames=['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])
checkpoint = ModelCheckpoint(filepath='best_model.pth', monitor='val_loss', verbose=1, save_best_only=True)

# 정확도 계산 함수
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)

# 학습 및 검증 루프
for epoch in range(10):
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    total = len(train_loader)
    pbar = tqdm(enumerate(train_loader), total=total, desc=f"Epoch {epoch+1}/{30}, Train")
    for batch_idx, (inputs, labels) in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_correct += calculate_accuracy(outputs, labels) * labels.size(0)
        train_total += labels.size(0)
        pbar.set_postfix({'Train Loss': train_loss / (batch_idx + 1), 'Train Acc': train_correct / train_total})


    train_loss /= total
    train_acc = train_correct / train_total


    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    total_val = len(val_loader)
    pbar_val = tqdm(enumerate(val_loader), total=total_val, desc=f"Epoch {epoch+1}/{30}, Validate")
    with torch.no_grad():
        for batch_idx, (inputs, labels) in pbar_val:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_correct += calculate_accuracy(outputs, labels) * labels.size(0)
            val_total += labels.size(0)
            pbar_val.set_postfix({'Val Loss': val_loss / (batch_idx + 1), 'Val Acc': val_correct / val_total})

    val_loss /= total_val
    val_acc = val_correct / val_total


    # 로깅 및 체크포인트 저장
    csv_logger.log(epoch=epoch, train_loss=train_loss, val_loss=val_loss, train_acc=train_acc, val_acc=val_acc)
    checkpoint.save(model, val_loss)

# 모델 저장
torch.save(model.state_dict(), 'model_trained2.pth')
