import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import AutoModelForImageClassification
import os
import requests

# Настройки страницы
st.set_page_config(page_title="Архитектурный классификатор", layout="centered")
st.title("🏛️ Интеллектуальный классификатор культовой архитектуры")
st.write("Загрузите фотографию фасада здания, и нейросеть ConvNeXt-V2 определит его конфессиональную принадлежность.")

# === НАСТРОЙКА СКАЧИВАНИЯ ВЕСОВ ИЗ HUGGING FACE ===
MODEL_PATH = "temple_model_final.pth"

# ⚠️ ВСТАВЬ СЮДА СВОЮ ССЫЛКУ С HUGGING FACE МЕЖДУ КАВЫЧКАМИ:
HF_URL = "https://drive.google.com/file/d/1k-KEiXw-7ceV7FOpjL5Ow9VW1-Gd2_xp/view?usp=drive_link"

@st.cache_resource
def download_weights(url, output):
    if os.path.exists(output) and os.path.getsize(output) > 100000000:
        return
        
    with st.spinner("Загрузка чистых весов модели из репозитория Hugging Face..."):
        response = requests.get(url, stream=True)
        with open(output, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)

# Скачиваем веса
try:
    download_weights(HF_URL, MODEL_PATH)
except Exception as e:
    st.error(f"Ошибка скачивания весов: {e}")

# === КОД МОДЕЛИ ===
IMAGE_SIZE = 224
CLASSES = [
    "Античный храм", 
    "Буддийский храм / Пагода", 
    "Католический собор", 
    "Индуистский храм", 
    "Мечеть", 
    "Православный храм", 
    "Протестантская церковь", 
    "Синагога"
]

@st.cache_resource
def load_model():
    model_obj = AutoModelForImageClassification.from_pretrained(
        "facebook/convnextv2-large-1k-224", num_labels=8, ignore_mismatched_sizes=True
    )
    model_obj.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False))
    model_obj.eval()
    return model_obj

# Безопасная инициализация глобальной переменной model
model = None
try:
    model = load_model()
except Exception as e:
    st.error(f"Ошибка инициализации модели: {e}")

transform = A.Compose([
    A.LongestMaxSize(max_size=IMAGE_SIZE),
    A.PadIfNeeded(min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=0, fill=(255, 255, 255)),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

uploaded_file = st.file_uploader("Перетащите сюда фотографию храма...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Загруженный объект", use_container_width=True)
    
    if model is None:
        st.error("Критическая ошибка: Модель не была загружена. Пожалуйста, проверьте логи в 'Manage app'.")
    else:
        with st.spinner("Нейросеть извлекает архитектурные дескрипторы..."):
            img_np = np.array(image)
            augmented = transform(image=img_np)
            img_tensor = augmented['image'].unsqueeze(0)
            
            with torch.no_grad():
                outputs = model(img_tensor).logits
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                
            st.subheader("Результаты инференса:")
            results = sorted(zip(CLASSES, probabilities.tolist()), key=lambda x: x[1], reverse=True)
            
            for cls_name, prob in results:
                if prob > 0.01:
                    st.write(f"**{cls_name}**: {prob*100:.2f}%")
                    st.progress(prob)
