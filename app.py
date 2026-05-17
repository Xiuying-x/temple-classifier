import streamlit as st
import torch
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import AutoModelForImageClassification
import os

# Настройки страницы
st.set_page_config(page_title="Архитектурный классификатор", layout="centered")
st.title("🏛️ Интеллектуальный классификатор культовой архитектуры")

# Константы
MODEL_PATH = "temple_classifier_best.pth"
GITHUB_RELEASE_URL = "https://github.com/Xiuying-x/temple-classifier/releases/download/v1.0.0/temple_classifier_best.pth"
IMAGE_SIZE = 224
CLASSES = [
    "Античный храм", "Буддийский храм / Пагода", "Католический собор", 
    "Индуистский храм", "Мечеть", "Православный храм", "Протестантская церковь", "Синагога"
]

@st.cache_resource
def load_model():
    # Если файла нет или он сломан (меньше 1 МБ), скачиваем его заново
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        
        # Скачиваем напрямую из релизов GitHub без перегрузки RAM сервера
        torch.hub.download_url_to_file(GITHUB_RELEASE_URL, MODEL_PATH, progress=False)

    # Инициализируем архитектуру ConvNeXt-V2
    model_obj = AutoModelForImageClassification.from_pretrained(
        "facebook/convnextv2-large-1k-224", num_labels=8, ignore_mismatched_sizes=True
    )
    
    # Загружаем веса (weights_only=False отключает строгую проверку PyTorch)
    model_obj.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False))
    model_obj.eval()
    return model_obj

# Безопасный запуск модели
try:
    with st.spinner("⏳ Активация нейросети... При первом запуске идет скачивание весов. Пожалуйста, подождите."):
        model = load_model()
    st.success("✅ Нейросеть успешно активирована и готова!")
except Exception as e:
    st.error(f"Ошибка инициализации модели: {e}")
    model = None

# Если модель загружена успешно — выводим интерфейс классификации
if model is not None:
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
