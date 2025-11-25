from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os

from classifier import PotsdamSegmentationClassifier
from models.unetpp import UnetPP_EfficientNetB0  # класс модели из unetpp.py

# Путь к модели (локальный кеш)
MODEL_PATH = "models/best_unetpp_efficientnetb0.pth"
# ID файла на Google Drive (можно задать прямо здесь или через ENV)
GDRIVE_FILE_ID = os.getenv("GDRIVE_FILE_ID", "1gKCR8pXAUwfk1kflaz3YTYwHLwrVvQ5_")

app = FastAPI(title="Potsdam Segmentation Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Глобальный сегментатор
seg: PotsdamSegmentationClassifier = None

@app.on_event("startup")
async def startup():
    """Инициализация при запуске"""
    global seg
    seg = PotsdamSegmentationClassifier(
        model_path=MODEL_PATH,
        tile=256,
        overlap=32,
        batch_size=8,
        google_drive_file_id=GDRIVE_FILE_ID
    )
    seg.load_model(model_class=UnetPP_EfficientNetB0)
    print("✅ PyTorch модель загружена и прогрета")

@app.post("/segment/")
async def segment(
    file: UploadFile = File(...),
    format: str = Query("geotiff", description="Формат результата: geotiff | tiff | geojson | png"),
    classes: List[int] = Query(None, description="Список активных классов (например: 1&classes=2)")
):
    """
    Сегментация аэрофотоснимка.
    - PNG-визуализация всегда возвращается.
    - Пользователь выбирает формат результата: GeoTIFF, TIFF, GeoJSON или PNG-маска.
    - Можно указать список активных классов для фильтрации.
    """
    contents = await file.read()
    try:
        results = seg.segment_all(contents, active_classes=classes)

        # PNG-визуализация всегда
        response = {"visualization": results["visualization"], "stats": results["stats"]}

        # Возвращаем выбранный формат
        if format == "geotiff":
            response["geotiff"] = results["geotiff"]
        elif format == "tiff":
            response["tiff"] = results["tiff"]
        elif format == "geojson":
            response["geojson"] = results["geojson"]
        elif format == "png":
            response["mask_png"] = results["visualization"]
        else:
            return JSONResponse({"error": "Неверный формат"}, status_code=400)

        return JSONResponse(response)

    except Exception as e:
        return JSONResponse({"error": f"Ошибка сегментации: {str(e)}"}, status_code=500)

@app.get("/classes")
async def get_classes():
    """Возвращает список классов и их цвета (легенду)"""
    if seg:
        return {"classes": seg.class_names, "colors": seg.class_colors}
    return {"error": "Сегментатор не инициализирован"}

@app.get("/health")
async def health_check():
    """Проверка состояния"""
    return {"status": "ok", "model_loaded": seg is not None and seg.model is not None}
