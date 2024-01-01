import torch

# GPU 사용 가능 여부 확인
if torch.cuda.is_available():
    print("GPU is available.")
    print(f"Current GPU Device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("GPU is not available, using CPU.")