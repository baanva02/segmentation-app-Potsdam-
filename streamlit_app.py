import streamlit as st
import base64
import io
import zipfile
import time

from classifier import PotsdamSegmentationClassifier
from models.unetpp import UnetPP_EfficientNetB0


# ---------- Кэшируем загрузку модели ----------
@st.cache_resource
def load_seg():
    seg = PotsdamSegmentationClassifier(
        model_path="models/best_unetpp_efficientnetb0.pth",
        tile=256,
        overlap=32,
        batch_size=8,
        google_drive_file_id="1gKCR8pXAUwfk1kflaz3YTYwHLwrVvQ5_"  # можно вынести в ENV
    )
    seg.load_model(model_class=UnetPP_EfficientNetB0)
    return seg


seg = load_seg()

st.set_page_config(page_title="Potsdam Segmentation", layout="wide")

# ---------- Сайдбар: легенда ----------
st.sidebar.header("🎨 Легенда")
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
st.sidebar.info("Классификация аэрофотоснимков (UNet++ + EfficientNetB0). 🚀 PyTorch & Streamlit")

# ---------- Основной интерфейс ----------
st.title("Классификация аэрофотоснимков (UNet++ + EfficientNetB0)")

uploaded_file = st.file_uploader(
    "Загрузите растровое изображение",
    type=["tif", "tiff", "jpg", "jpeg", "png"]
)

classes = st.multiselect(
    "Выберите классы для сегментации",
    options=list(seg.class_names.keys()),
    format_func=lambda x: seg.class_names[x],
    default=list(seg.class_names.keys())
)

if uploaded_file:
    st.write("Файл выбран:", uploaded_file.name)

    if st.button("Сегментировать"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        # ---------- Этап 1: чтение файла ----------
        status_text.text("Чтение файла...")
        contents = uploaded_file.read()
        progress_bar.progress(20)
        time.sleep(0.3)

        # ---------- Этап 2: сегментация ----------
        status_text.text("Сегментация изображения...")
        results = seg.segment_all(contents, active_classes=classes)
        progress_bar.progress(70)
        time.sleep(0.3)

        # ---------- Этап 3: экспорт результатов ----------
        status_text.text("Формирование выходных форматов...")
        vis_bytes = base64.b64decode(results["visualization"])
        geotiff_bytes = base64.b64decode(results["geotiff"])
        tiff_bytes = base64.b64decode(results["tiff"])
        geojson_str = results["geojson"]
        progress_bar.progress(100)
        status_text.text("Готово ✅")

        # ---------- Визуализация ----------
        st.subheader("Результат сегментации (PNG)")
        st.image(vis_bytes, caption="Визуализация", use_column_width=True)

        # ---------- Статистика ----------
        st.subheader("Статистика по классам")
        stats = results["stats"]
        for cid, s in stats.items():
            st.write(f"{s['name']}: {s['pixels']} px ({s['percent']}%)")

        # ---------- Кнопки скачивания ----------
        st.subheader("Скачать результат")
        st.download_button("📥 GeoTIFF", geotiff_bytes,
                           file_name="result_geotiff.tif", mime="image/tiff")
        st.download_button("📥 TIFF", tiff_bytes,
                           file_name="result.tiff", mime="image/tiff")
        st.download_button("📥 PNG маска", vis_bytes,
                           file_name="mask.png", mime="image/png")
        st.download_button("📥 GeoJSON", geojson_str,
                           file_name="result.geojson", mime="application/geo+json")

        # ---------- ZIP со всеми форматами ----------
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            zf.writestr("mask.png", vis_bytes)
            zf.writestr("mask_geotiff.tif", geotiff_bytes)
            zf.writestr("mask.tiff", tiff_bytes)
            zf.writestr("mask.geojson", geojson_str)
        zip_buf.seek(0)

        st.download_button("📥 ZIP (все форматы)", zip_buf.getvalue(),
                           file_name="segmentation_results.zip", mime="application/zip")
