from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import base64, tempfile, zipfile

from classifier import PotsdamSegmentationClassifier
from models.unetpp import UnetPP_EfficientNetB0

MODEL_PATH = "models/best_unetpp_efficientnetb0.pth"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

seg: Optional[PotsdamSegmentationClassifier] = None
last_results: Optional[dict] = None  # хранит последний результат для скачивания

@app.on_event("startup")
async def startup():
    global seg
    seg = PotsdamSegmentationClassifier(MODEL_PATH, tile=256, overlap=32, batch_size=8)
    seg.load_model(model_class=UnetPP_EfficientNetB0)
    print("PyTorch модель загружена и прогрета")

# ---------- Фронт ----------
@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <html>
    <head>
        <title>Классификация аэрофотоснимков (модель UNet++ + EfficientNetB0)</title>
        <meta charset="utf-8"/>
        <style>
            body { font-family: Arial, sans-serif; display: flex; margin: 0; padding: 0; background: #fff; color: #222; }
            .sidebar { width: 280px; background-color: #f7f7f7; padding: 20px; border-right: 1px solid #ddd; box-sizing: border-box; }
            .legend-item { display: flex; align-items: center; margin-bottom: 10px; gap: 10px; }
            .color-box { width: 20px; height: 20px; border: 1px solid #000; }
            .main { flex-grow: 1; padding: 30px; box-sizing: border-box; }
            .upload-box { border: 2px dashed #aaa; padding: 24px; text-align: center; background-color: #fafafa; border-radius: 6px; }
            input[type="file"] { margin-top: 10px; }
            select, button { margin-top: 16px; padding: 8px 12px; font-size: 14px; }
            .section-title { margin-top: 0; }
            .about { font-size: 13px; color: #555; line-height: 1.4; }
            .results { margin-top: 24px; }
            .downloads { margin-top: 16px; }
            .downloads button { margin-right: 8px; }
            .progress-container { margin-top: 16px; }
            .hidden { display: none; }
            progress { width: 100%; height: 16px; }
            img.preview { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }
            .class-checks label { margin-right: 12px; }
        </style>
    </head>
    <body>
        <div class="sidebar">
            <h3 class="section-title">🎨 Легенда</h3>
            <div class="legend-item"><div class="color-box" style="background:brown;"></div><span>Здания</span></div>
            <div class="legend-item"><div class="color-box" style="background:gray;"></div><span>Дороги</span></div>
            <div class="legend-item"><div class="color-box" style="background:purple;"></div><span>Автомобили</span></div>
            <div class="legend-item"><div class="color-box" style="background:limegreen;"></div><span>Деревья</span></div>
            <div class="legend-item"><div class="color-box" style="background:green;"></div><span>Трава</span></div>
            <hr/>
            <div class="about">
                <p><b>О приложении</b><br/>Классификация аэрофотоснимков (модель UNet++ + EfficientNetB0).</p>
                <p>🚀 PyTorch & FastAPI</p>
            </div>
        </div>
        <div class="main">
            <h2 class="section-title">Классификация аэрофотоснимков (модель UNet++ + EfficientNetB0)</h2>

            <form id="segForm">
                <div class="upload-box">
                    <p>Загрузите растровое изображение</p>
                    <input id="fileInput" name="file" type="file" accept=".tif,.tiff,.jpg,.jpeg,.png" required/>
                    <p style="font-size:12px; color:#666;">Limit 200MB per file • TIF, TIFF, JPG, PNG, JPEG</p>
                </div>

                <div class="class-checks">
                    <label for="classes">Выберите классы:</label><br/>
                    <label><input type="checkbox" name="classes" value="1" checked/> Здания</label>
                    <label><input type="checkbox" name="classes" value="2" checked/> Дороги</label>
                    <label><input type="checkbox" name="classes" value="5" checked/> Автомобили</label>
                    <label><input type="checkbox" name="classes" value="4" checked/> Деревья</label>
                    <label><input type="checkbox" name="classes" value="3" checked/> Трава</label>
                </div>
                <br/>
                <button type="submit">Сегментировать (PNG сначала)</button>
            </form>

            <div class="progress-container hidden" id="progressBox">
                <p>Идёт сегментация…</p>
                <progress id="progressBar" max="100" value="0"></progress>
            </div>

            <div class="results hidden" id="resultsBox">
                <h3>Результат сегментации (PNG)</h3>
                <img id="previewImg" class="preview" src="" alt="Визуализация"/>
                <div id="statsBox"></div>

                <div class="downloads">
                    <h3>Скачать результат</h3>
                    <button onclick="downloadFmt('geotiff')">GeoTIFF</button>
                    <button onclick="downloadFmt('tiff')">TIFF</button>
                    <button onclick="downloadFmt('geojson')">GeoJSON</button>
                    <button onclick="downloadFmt('zip')">ZIP (все форматы)</button>
                </div>
            </div>
        </div>

        <script>
            const form = document.getElementById('segForm');
            const progressBox = document.getElementById('progressBox');
            const progressBar = document.getElementById('progressBar');
            const resultsBox = document.getElementById('resultsBox');
            const previewImg = document.getElementById('previewImg');
            const statsBox = document.getElementById('statsBox');

            form.addEventListener('submit', async (e) => {
                e.preventDefault();

                const file = document.getElementById('fileInput').files[0];
                if (!file) { alert('Выберите файл'); return; }

                const clsInputs = document.querySelectorAll('input[name="classes"]:checked');
                const classes = Array.from(clsInputs).map(i => i.value);

                const formData = new FormData();
                formData.append('file', file);
                formData.append('format', 'png'); // сначала визуализируем
                classes.forEach(c => formData.append('classes', c));

                progressBox.classList.remove('hidden');
                resultsBox.classList.add('hidden');
                progressBar.value = 10;

                try {
                    const resp = await fetch('/segment/', { method: 'POST', body: formData });
                    progressBar.value = 70;

                    if (!resp.ok) {
                        const err = await resp.json().catch(()=>({}));
                        alert(err.error || 'Ошибка сегментации');
                        progressBar.value = 0;
                        progressBox.classList.add('hidden');
                        return;
                    }

                    const data = await resp.json();
                    previewImg.src = 'data:image/png;base64,' + data.visualization;

                    const stats = data.stats || {};
                    let statsHtml = '<h4>Статистика по классам</h4><ul>';
                    Object.keys(stats).forEach(cid => {
                        const s = stats[cid];
                        statsHtml += `<li>${s.name}: ${s.pixels} px (${s.percent}%)</li>`;
                    });
                    statsHtml += '</ul>';
                    statsBox.innerHTML = statsHtml;

                    progressBar.value = 100;
                    progressBox.classList.add('hidden');
                    resultsBox.classList.remove('hidden');

                } catch (err) {
                    alert('Ошибка сети или сервера: ' + err);
                    progressBar.value = 0;
                    progressBox.classList.add('hidden');
                }
            });

            async function downloadFmt(fmt) {
                try {
                    const resp = await fetch('/download/' + fmt);
                    if (!resp.ok) {
                        const err = await resp.json().catch(() => ({}));
                        alert(err.error || 'Нет результата для скачивания');
                        return;
                    }
                    const blob = await resp.blob();
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    let fname = 'result';
                    if (fmt === 'geotiff') fname = 'result_geotiff.tif';
                    else if (fmt === 'tiff') fname = 'result.tiff';
                    else if (fmt === 'geojson') fname = 'result.geojson';
                    else if (fmt === 'zip') fname = 'segmentation_results.zip';
                    a.download = fname;
                    document.body.appendChild(a);
                    a.click();
                    a.remove();
                    URL.revokeObjectURL(url);
                } catch (e) {
                    alert('Ошибка скачивания: ' + e);
                }
            }
        </script>
    </body>
    </html>
    """

# ---------- Сегментация: всегда возвращаем визуализацию и сохраняем полный результат ----------
@app.post("/segment/")
async def segment(
    file: UploadFile = File(...),
    format: str = Form("png"),
    classes: List[int] = Form(None)
):
    global last_results
    contents = await file.read()
    try:
        results = seg.segment_all(contents, active_classes=classes)
        last_results = results  # сохраняем для дальнейших скачиваний

        return JSONResponse({
            "visualization": results["visualization"],
            "stats": results["stats"]
        })

    except Exception as e:
        return JSONResponse({"error": f"Ошибка сегментации: {str(e)}"}, status_code=500)

# ---------- Скачивание результата в выбранном формате ----------
@app.get("/download/{fmt}")
async def download(fmt: str):
    global last_results
    if not last_results:
        return JSONResponse({"error": "Нет результата для скачивания. Сначала выполните сегментацию."}, status_code=400)

    try:
        if fmt == "geotiff":
            data = base64.b64decode(last_results["geotiff"])
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
            tmp.write(data); tmp.close()
            return FileResponse(tmp.name, media_type="image/tiff", filename="result_geotiff.tif")

        elif fmt == "tiff":
            data = base64.b64decode(last_results["tiff"])
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tiff")
            tmp.write(data); tmp.close()
            return FileResponse(tmp.name, media_type="image/tiff", filename="result.tiff")

        elif fmt == "png":
            data = base64.b64decode(last_results["visualization"])
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tmp.write(data); tmp.close()
            return FileResponse(tmp.name, media_type="image/png", filename="mask.png")

        elif fmt == "geojson":
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".geojson")
            tmp.write(last_results["geojson"].encode("utf-8")); tmp.close()
            return FileResponse(tmp.name, media_type="application/geo+json", filename="result.geojson")

        elif fmt == "zip":
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
            with zipfile.ZipFile(tmp.name, "w") as zf:
                zf.writestr("mask.png", base64.b64decode(last_results["visualization"]))
                zf.writestr("mask_geotiff.tif", base64.b64decode(last_results["geotiff"]))
                zf.writestr("mask.tiff", base64.b64decode(last_results["tiff"]))
                zf.writestr("mask.geojson", last_results["geojson"])
            return FileResponse(tmp.name, media_type="application/zip", filename="segmentation_results.zip")

        else:
            return JSONResponse({"error": "Неверный формат скачивания"}, status_code=400)

    except Exception as e:
        return JSONResponse({"error": f"Ошибка скачивания: {str(e)}"}, status_code=500)

# ---------- Служебные эндпоинты ----------
@app.get("/classes")
async def get_classes():
    if seg:
        return {"classes": seg.class_names, "colors": seg.class_colors}
    return {"error": "Сегментатор не инициализирован"}

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": seg is not None and seg.model is not None}
