import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        self.model = models.efficientnet_b0(weights='IMAGENET1K_V1')
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def process_image_for_ocr(image):
    """แปลงภาพเป็นขาว-ดำและหาพื้นที่ของตัวอักษร"""
    if isinstance(image, str):
        image = cv2.imread(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    char_regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > 100:  # กรองพื้นที่เล็กๆ ออก
            char_regions.append((x, y, w, h))
    
    char_regions.sort(key=lambda x: x[0])  # เรียงตามแกน x
    return image, char_regions

def predict_image(image_region, model, transform, class_names, device):
    """ทำนายตัวอักษรเดี่ยว"""
    image_pil = Image.fromarray(cv2.cvtColor(image_region, cv2.COLOR_BGR2RGB))
    image_tensor = transform(image_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        probability = torch.nn.functional.softmax(outputs, dim=1)
    
    predicted_class = class_names[predicted.item()]
    confidence = probability[0][predicted.item()].item()
    return predicted_class, confidence

def process_read_license(image, model_path, font_path=None):
    """ประมวลผลการอ่านป้ายทะเบียน"""
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                  'ก', 'ข', 'ค', 'ฆ', 'ง', 'จ', 'ฉ', 'ช', 'ฌ', 'ญ', 
                  'ฎ', 'ฐ', 'ฒ', 'ณ', 'ด', 'ต', 'ถ', 'ท', 'ธ', 'น', 
                  'บ', 'ป', 'ผ', 'พ', 'ฟ', 'ภ', 'ม', 'ย', 'ร', 'ล', 'ว', 
                  'ศ', 'ษ', 'ส', 'ห', 'ฬ', 'อ', 'ฮ']
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    
    image, char_regions = process_image_for_ocr(image)
    
    results = []
    for x, y, w, h in char_regions:
        char_image = image[y:y+h, x:x+w]
        predicted_class, confidence = predict_image(char_image, model, transform, class_names, device)
        # เพิ่มเงื่อนไขกรองเฉพาะผลลัพธ์ที่มีความแม่นยำมากกว่า 50%
        if confidence >= 0.8: 
            results.append((predicted_class, confidence, (x, y, w, h)))
    
    total_confidence = sum(conf for _, conf, _ in results) / len(results) if results else 0
    
    return results, total_confidence