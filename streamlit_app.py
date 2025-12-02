import streamlit as st
import base64
import io
import zipfile
import time

from classifier import PotsdamSegmentationClassifier
from models.unetpp import UnetPP_EfficientNetB0

# ---------- Настройки страницы ----------
st.set_page_config(page_title="Сегментация аэрофотоснимков", layout="wide")

# ---------- Кэшируем загрузку модели ----------
def load_seg():
    seg = PotsdamSegmentationClassifier(
        model_path="models/best_unetpp_efficientnetb0.pth",
        tile=256,
        overlap=32,
        batch_size=8,
        google_drive_file_id="1gKCR8pXAUwfk1kflaz3YTYwHLwrVvQ5_"
    )
    seg.load_model(model_class=UnetPP_EfficientNetB0)
    return seg

seg = load_seg()

# ---------- Сайдбар: легенда ----------
st.sidebar.header("🎨 Легенда классов")
for cid, name in seg.class_names.items():
    color = seg.class_colors[cid]
    hex_color = '#%02x%02x%02x' % color
    st.sidebar.markdown(
        f"<div style='display:flex;align-items:center;'>"
        f"<div style='width:20px;height:20px;background:{hex_color};"
        f"border:1px solid #000;margin-right:8px;'></div>{name}</div>",
        unsafe_allow_html=True
    )

st.sidebar.markdown("---")
st.sidebar.info("Сегментация аэрофотоснимков с использованием UNet++ и EfficientNetB0. 🚀 PyTorch + Streamlit")

# ---------- Основной интерфейс ----------
st.title("Сегментация аэрофотоснимков")

uploaded_file = st.file_uploader(
    "📤 Загрузите изображение (TIF, TIFF, JPG, PNG)",
    type=["tif", "tiff", "jpg", "jpeg", "png"]
)

classes = st.multiselect(
    "🧭 Выберите классы для сегментации",
    options=list(seg.class_names.keys()),
    format_func=lambda x: seg.class_names[x],
    default=list(seg.class_names.keys())
)

if uploaded_file:
    st.success(f"✅ Файл выбран: {uploaded_file.name}")

    if st.button("🚀 Начать сегментацию"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        # ---------- Этап 1: чтение файла ----------
        status_text.text("📂 Чтение изображения...")
        contents = uploaded_file.read()
        progress_bar.progress(20)
        time.sleep(0.3)

        # ---------- Этап 2: сегментация ----------
        status_text.text("🧠 Выполняется сегментация...")
        results = seg.segment_all(contents, active_classes=classes)
        progress_bar.progress(70)
        time.sleep(0.3)

        # ---------- Этап 3: экспорт ----------
        status_text.text("📦 Подготовка результатов...")
        vis_bytes = base64.b64decode(results["visualization"])
        geotiff_bytes = base64.b64decode(results["geotiff"])
        tiff_bytes = base64.b64decode(results["tiff"])
        geojson_str = results["geojson"]
        progress_bar.progress(100)
        status_text.text("✅ Готово!")

        # ---------- Визуализация ----------
        st.subheader("🖼️ Визуализация сегментации")
        st.image(vis_bytes, caption="Цветовая маска", use_column_width=True)

        # ---------- Статистика ----------
        st.subheader("📊 Статистика по классам")
        stats = results["stats"]
        for cid, s in stats.items():
            st.write(f"• {s['name']}: {s['pixels']} пикселей ({s['percent']}%)")

        # ---------- Скачивание ----------
        st.subheader("📥 Скачать результаты")
        st.download_button("📥 GeoTIFF", geotiff_bytes,
                           file_name="результат_geotiff.tif", mime="image/tiff")
        st.download_button("📥 TIFF (маска)", tiff_bytes,
                           file_name="результат.tiff", mime="image/tiff")
        st.download_button("📥 PNG (визуализация)", vis_bytes,
                           file_name="маска.png", mime="image/png")
        st.download_button("📥 GeoJSON (векторизация)", geojson_str,
                           file_name="результат.geojson", mime="application/geo+json")

        # ---------- ZIP архив ----------
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            zf.writestr("маска.png", vis_bytes)
            zf.writestr("маска_geotiff.tif", geotiff_bytes)
            zf.writestr("маска.tiff", tiff_bytes)
            zf.writestr("маска.geojson", geojson_str)
        zip_buf.seek(0)

        st.download_button("📥 ZIP-архив (все форматы)", zip_buf.getvalue(),
                           file_name="результаты_сегментации.zip", mime="application/zip")
