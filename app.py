import streamlit as st
import torch
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import AutoModelForImageClassification
import os
import re

st.set_page_config(page_title="Архитектурный классификатор", layout="centered")
st.title("🏛️ Интеллектуальный классификатор культовой архитектуры")

MODEL_PATH = "temple_classifier_best.pth"
IMAGE_SIZE = 224
CLASSES = ["Античный храм", "Буддийский храм / Пагода", "Католический собор", 
           "Индуистский храм", "Мечеть", "Православный храм", "Протестантская церковь", "Синагога"]

GDRIVE_URL = "https://drive.google.com/file/d/1v3OAtV88fK7z4mGzly4tU6RzWco62r93/view?usp=sharing"

def extract_gdrive_id(url):
    match = re.search(r'/d/([a-zA-Z0-9-_]+)', url)
    return match.group(1) if match else url

@st.cache_resource
def load_model():
    # Проверяем, есть ли файл. Если файла нет или он сломан, качаем через встроенный инструмент PyTorch
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
        file_id = extract_gdrive_id(GDRIVE_URL)
        # Бронебойная прямая ссылка Google API
        direct_download_url = f"https://docs.google.com/uc?export=download&confirm=t&id={file_id}"
        
        # Встроенный в PyTorch загрузчик больших файлов (скачивает напрямую на диск без перегрузки RAM)
        torch.hub.download_url_to_file(direct_download_url, MODEL_PATH, progress=False)

    # Создаем саму модель
    model_obj = AutoModelForImageClassification.from_pretrained(
        "facebook/convnextv2-large-1k-224", num_labels=8, ignore_mismatched_sizes=True
    )
    # Загружаем скачанные веса
    model_obj.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False))
    model_obj.eval()
    return model_obj

# === ОДНОЭТАПНЫЙ ЗАПУСК ===
try:
    with st.spinner("⏳ Инициализация нейросети... При первом запуске идет фоновое скачивание весов модели (749 МБ). Пожалуйста, подождите 1–3 минуты и не закрывайте вкладку."):
        model = load_model()
    st.success("✅ Нейросеть ConvNeXt-V2 успешно активирована и готова!")
except Exception as e:
    st.error(f"Не удалось инициализировать веса модели: {e}")
    # Кнопка сброса на случай, если скачался битый файл
    if st.button("♻️ Сбросить файлы и попробовать снова"):
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        st.rerun()
    model = None

# Если модель успешно загрузилась (или скачалась и загрузилась)
if model is not None:
    st.write("---")
    transform = A.Compose([
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=0, fill=(255, 255, 255)),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    uploaded_file = st.file_uploader("Загрузите фотографию храма для классификации...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Загруженный объект", use_container_width=True)
        
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
