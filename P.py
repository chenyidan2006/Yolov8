from ultralytics import YOLO
import torch
model = YOLO('D:\download/ultralytics-8.2.0/runs\detect/train5\weights\last.pt')
total_params = sum(p.numel() for p in model.parameters())
print(total_params)