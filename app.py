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

# === НАСТРОЙКА СКАЧИВАНИЯ ВЕСОВ ИЗ GOOGLE DRIVE ===
MODEL_PATH = "temple_weights_gdrive_final.pth"
GOOGLE_DRIVE_URL = "https://drive.google.com/file/d/1k-KEiXw-7ceV7FOpjL5Ow9VW1-Gd2_xp/view?usp=sharing"

def extract_gdrive_id(url):
    if "id=" in url:
        return url.split("id=")[1].split("&")[0]
    elif "file/d/" in url:
        return url.split("file/d/")[1].split("/")[0]
    return url

@st.cache_resource
def download_weights_from_gdrive(url, output):
    # Если файл существует, но он подозрительно маленький (меньше 100 МБ) — это ошибка. Удаляем его!
    if os.path.exists(output) and os.path.getsize(output) < 100000000:
        os.remove(output)
        
    # Если файла нет (или мы его только что удалили как битый), запускаем скачивание
    if not os.path.exists(output):
        with st.spinner("Загрузка тяжелых весов модели из Google Диска (это займет около 3-5 минут)..."):
            file_id = extract_gdrive_id(url)
            # Прямая ссылка, которая заставляет Google пропустить страницу предупреждения о вирусах
            download_url = f"https://docs.google.com/uc?export=download&confirm=t&id={file_id}"
            
            session = requests.Session()
            response = session.get(download_url, stream=True)
            
            with open(output, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024*1024): # Качаем блоками по 1 МБ
                    if chunk:
                        f.write(chunk)

# Запускаем скачивание весов
try:
    download_weights_from_gdrive(GOOGLE_DRIVE_URL, MODEL_PATH)
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
    # weights_only=False отключает строгую безопасную проверку PyTorch 2.6 для кастомных весов
    model_obj.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False))
    model_obj.eval()
    return model_obj

# Безопасная инициализация модели
model = None
if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 100000000:
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Ошибка инициализации структуры весов модели: {e}")
else:
    st.info("Ожидание завершения скачивания файла весов...")

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
        st.error("Критическая ошибка: Файл весов поврежден или еще не скачался полностью.")
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
