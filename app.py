import streamlit as st
import torch
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import AutoModelForImageClassification
import os
import requests
import re

st.set_page_config(page_title="Архитектурный классификатор", layout="centered")
st.title("🏛️ Интеллектуальный классификатор культовой архитектуры")

MODEL_PATH = "temple_classifier_best.pth"
IMAGE_SIZE = 224
CLASSES = ["Античный храм", "Буддийский храм / Пагода", "Католический собор", 
           "Индуистский храм", "Мечеть", "Православный храм", "Протестантская церковь", "Синагога"]

# Ссылка на твой Google Диск с весами
GDRIVE_URL = "https://drive.google.com/file/d/1v3OAtV88fK7z4mGzly4tU6RzWco62r93/view?usp=sharing"

def extract_gdrive_id(url):
    match = re.search(r'/d/([a-zA-Z0-9-_]+)', url)
    return match.group(1) if match else url

@st.cache_resource
def load_model():
    model_obj = AutoModelForImageClassification.from_pretrained(
        "facebook/convnextv2-large-1k-224", num_labels=8, ignore_mismatched_sizes=True
    )
    model_obj.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False))
    model_obj.eval()
    return model_obj

# Инициализируем состояние скачивания в памяти Streamlit
if 'download_started' not in st.session_state:
    st.session_state.download_started = False

# === ПРОВЕРКА НАЛИЧИЯ МОДЕЛИ ===
if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
    st.warning("⚠️ Файл весов модели (749 МБ) отсутствует на сервере.")
    
    # Кнопка теперь просто меняет флаг в памяти, что исключает "залипание"
    if st.button("🚀 Начать скачивание весов на сервер") or st.session_state.download_started:
        st.session_state.download_started = True
        
        # Принудительно стираем старый поврежденный файл
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
            
        file_id = extract_gdrive_id(GDRIVE_URL)
        download_url = f"https://docs.google.com/uc?export=download&confirm=t&id={file_id}"
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Устанавливаем соединение с Google Диском...")
        
        try:
            session = requests.Session()
            response = session.get(download_url, stream=True)
            
            # Пробиваем проверку больших файлов Google
            token = None
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    token = value
                    break
            if token:
                download_url = f"https://docs.google.com/uc?export=download&confirm={token}&id={file_id}"
                response = session.get(download_url, stream=True)
            
            total_length = 785431000  
            downloaded = 0
            
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=4*1024*1024): 
                    if chunk:
                        f.write(chunk)
                        f.flush()
                        downloaded += len(chunk)
                        
                        percent = min(int((downloaded / total_length) * 100), 100)
                        progress_bar.progress(percent)
                        status_text.text(f"Загружено: {downloaded / (1024*1024):.1f} из 749.0 МБ ({percent}%)")
            
            st.success("🎉 Веса успешно скачаны!")
            st.session_state.download_started = False
            st.rerun()
            
        except Exception as e:
            st.error(f"💥 Критическая ошибка при скачивании: {e}")
            st.session_state.download_started = False
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)

else:
    # Если настоящий файл весов на месте — запускаем нейросеть
    try:
        model = load_model()
        st.success("✅ Нейросеть ConvNeXt-V2 успешно активирована и готова к работе!")
    except Exception as e:
        st.error(f"Ошибка инициализации весов: {e}")
        # Если файл битый, даем возможность удалить его кнопкой
        if st.button("Сбросить поврежденный файл весов"):
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
            st.rerun()
        model = None

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
