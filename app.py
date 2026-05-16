import streamlit as st
import torch
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import AutoModelForImageClassification
import os

st.set_page_config(page_title="Архитектурный классификатор", layout="centered")
st.title("🏛️ Интеллектуальный классификатор культовой архитектуры")

MODEL_PATH = "temple_classifier_best.pth"
IMAGE_SIZE = 224
CLASSES = ["Античный храм", "Буддийский храм / Пагода", "Католический собор", 
           "Индуистский храм", "Мечеть", "Православный храм", "Протестантская церковь", "Синагога"]

@st.cache_resource
def load_model():
    model_obj = AutoModelForImageClassification.from_pretrained(
        "facebook/convnextv2-large-1k-224", num_labels=8, ignore_mismatched_sizes=True
    )
    model_obj.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False))
    model_obj.eval()
    return model_obj

# === ПРОВЕРКА НАЛИЧИЯ МОДЕЛИ В ПАМЯТИ СЕРВЕРА ===
if not os.path.exists(MODEL_PATH):
    st.warning("⚠️ Файл весов модели не найден на сервере.")
    st.write("Пожалуйста, загрузите файл `temple_classifier_best.pth` (749 МБ) с вашего компьютера, чтобы инициализировать нейросеть:")
    
    # Окошко для ручной загрузки весов
    weights_file = st.file_uploader("Перетащите сюда файл temple_classifier_best.pth", type=["pth"])
    
    if weights_file is not None:
        with st.spinner("Сохраняем веса в память сервера... Это займет около минуты."):
            with open(MODEL_PATH, "wb") as f:
                f.write(weights_file.getbuffer())
        st.success("Файл весов успешно сохранен!")
        st.rerun() # Перезапускаем сайт, чтобы он увидел файл

else:
    # Если файл уже на месте (или только что загружен)
    try:
        model = load_model()
        st.success("✅ Нейросеть ConvNeXt-V2 успешно активирована и готова к работе!")
    except Exception as e:
        st.error(f"Ошибка инициализации весов: {e}")
        model = None

    if model is not None:
        st.write("---")
        transform = A.Compose([
            A.LongestMaxSize(max_size=IMAGE_SIZE),
            A.PadIfNeeded(min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=0, fill=(255, 255, 255)),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

        uploaded_file = st.file_uploader("Шаг 2: Загрузите фотографию храма для классификации...", type=["jpg", "jpeg", "png"])

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
