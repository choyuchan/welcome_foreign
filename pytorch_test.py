import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models import mobilenet_v2  # 모델은 필요에 따라 변경 가능

# 음식 리스트 생성 함수
def create_foodlist(path):
    list_ = []
    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            list_.append(name)
    list_.sort()  # 리스트 정렬
    return list_

# 모델 로드 (예: MobileNetV2)
n_classes = 101 
model = mobilenet_v2(pretrained=False)  # 모델 구조를 정의
model.classifier[1] = torch.nn.Linear(model.last_channel, n_classes)

model.load_state_dict(torch.load('best_model.pth'))  # 모델 가중치 로드
model.eval()  # 모델을 평가 모드로 설정

food_list = create_foodlist("food101/images")

# 이미지 예측 함수
def predict_class(model, images, show=True):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for img_path in images:
        img = Image.open(img_path).convert('RGB')
        img_t = transform(img).unsqueeze(0)

        with torch.no_grad():
            pred = model(img_t)
            index = pred.argmax(dim=1).item()
            pred_value = food_list[index]

        if show:
            plt.imshow(img)
            plt.axis('off')
            plt.title(pred_value)
            plt.show()

# 예측할 이미지 목록
images = ["images.jpg"]

# 예측 실행
print("PREDICTIONS BASED ON PICTURES UPLOADED")
predict_class(model, images, True)
