import streamlit as st
import torch
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import AutoModelForImageClassification
import os

# Настройки интерфейса
st.set_page_config(page_title="Архитектурный классификатор", layout="centered")
st.title("🏛️ Интеллектуальный классификатор культовой архитектуры")
st.write("Загрузите фотографию фасада здания, и нейросеть ConvNeXt-V2 определит его конфессиональную принадлежность.")

# Локальное имя файла весов в репозитории
MODEL_PATH = "temple_classifier_best.pth"

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
    # Создаем архитектуру
    model_obj = AutoModelForImageClassification.from_pretrained(
        "facebook/convnextv2-large-1k-224", num_labels=8, ignore_mismatched_sizes=True
    )
    # Загружаем веса из локального файла
    model_obj.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False))
    model_obj.eval()
    return model_obj

# Проверяем, лежит ли файл весов в папке с проектом
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Файл весов `{MODEL_PATH}` не найден в репозитории!")
    st.info("Пожалуйста, загрузите файл весов (749 МБ) в свой репозиторий на GitHub.")
else:
    # Если файл на месте, активируем модель
    try:
        model = load_model()
        st.success("✅ Модель успешно загружена и готова к работе!")
    except Exception as e:
        st.error(f"Ошибка при инициализации весов: {e}")
        model = None

    # Настройка трансформации изображений
    transform = A.Compose([
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=0, fill=(255, 255, 255)),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # Форма загрузки фотографии пользователем
    uploaded_file = st.file_uploader("Перетащите сюда фотографию храма...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None and model is not None:
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
