import os
import io
import json
import base64
from typing import Tuple, List, Dict, Optional

import numpy as np
from PIL import Image

import torch
import gdown
import rasterio
from rasterio.io import MemoryFile
from rasterio.features import shapes
from rasterio.transform import from_bounds


class PotsdamSegmentationClassifier:
    """Сегментатор аэрофотоснимков Potsdam на PyTorch с автозагрузкой модели с Google Drive"""

    def __init__(
        self,
        model_path: str = "models/best_unetpp_efficientnetb0.pth",
        tile: int = 256,
        overlap: int = 32,
        batch_size: int = 8,
        google_drive_file_id: Optional[str] = "1gKCR8pXAUwfk1kflaz3YTYwHLwrVvQ5_",  # твой ID
    ):
        self.model_path = model_path
        self.tile = tile
        self.overlap = overlap
        self.batch_size = batch_size
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.google_drive_file_id = google_drive_file_id

        # Легенда Potsdam (фиксированные цвета)
        self.class_colors = {
            0: (0, 0, 0),          # Background
            1: (139, 69, 19),      # Buildings
            2: (128, 128, 128),    # Roads
            3: (0, 128, 0),        # Low vegetation
            4: (0, 255, 0),        # Trees
            5: (128, 0, 128),      # Cars
        }
        self.class_names = {
            0: "Background",
            1: "Building",
            2: "Road",
            3: "Low vegetation",
            4: "Tree",
            5: "Car",
        }

        os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)

    # ---------- Загрузка модели с Google Drive ----------
    def _download_model_from_gdrive(self, file_id: str, dest_path: str) -> None:
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        gdown.download(url, dest_path, quiet=False)

    def ensure_model(self) -> None:
        # если файл существует, но это HTML (начинается с "<"), удаляем
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                head = f.read(10)
            if head.startswith(b"<"):
                print("⚠️ Найден HTML вместо модели, удаляем...")
                os.remove(self.model_path)

        # если файла нет — качаем заново
        if not os.path.exists(self.model_path):
            if not self.google_drive_file_id:
                raise RuntimeError("Файл модели отсутствует и google_drive_file_id не задан.")
            print("🔄 Модель не найдена — загружаем с Google Drive...")
            self._download_model_from_gdrive(self.google_drive_file_id, self.model_path)
            print(f"✅ Модель загружена: {self.model_path}")

    # ---------- Инициализация модели ----------
    def load_model(self, model_class):
        if self.model is not None:
            return
        if model_class is None:
            raise RuntimeError("Для загрузки модели нужен model_class")

        self.ensure_model()

        m = model_class()
        checkpoint = torch.load(self.model_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            sd = checkpoint["model_state_dict"]
        else:
            sd = checkpoint

        try:
            m.load_state_dict(sd, strict=True)
        except RuntimeError as e:
            print("⚠️ Несовпадение state_dict при strict=True.\n", str(e))
            m.load_state_dict(sd, strict=False)

        self.model = m
        self.model.eval()
        self.model.to(self.device)

    # ---------- Входные данные ----------
    def process_raster(self, file_bytes: bytes) -> Tuple[np.ndarray, dict, object, object]:
        """
        Возвращает:
        - data: np.ndarray [H, W, 3] float32 в диапазоне [0,1]
        - profile: исходный профиль растрового файла (или заглушка)
        - transform: аффинное преобразование (или заглушка для PNG/JPG)
        - crs: система координат (None для PNG/JPG)
        """
        try:
            with MemoryFile(file_bytes) as memfile:
                with memfile.open() as src:
                    profile = src.profile.copy()
                    transform, crs = src.transform, src.crs
                    count = profile.get("count", 1)
                    if count >= 3:
                        data = src.read([1, 2, 3]).transpose(1, 2, 0).astype(np.float32)
                    else:
                        one = src.read(1).astype(np.float32)
                        data = np.stack([one, one, one], axis=-1)
                    data = np.nan_to_num(data, nan=0.0, posinf=255.0, neginf=0.0)
                    maxv = np.max(data) if np.isfinite(np.max(data)) else 255.0
                    data = (data / max(1.0, maxv)).clip(0.0, 1.0)
                    return data, profile, transform, crs
        except Exception:
            img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            data = np.array(img, dtype=np.float32) / 255.0
            h, w = data.shape[:2]
            profile = {"driver": "GTiff", "dtype": "uint8", "count": 1,
                       "height": h, "width": w, "compress": "lzw"}
            transform = from_bounds(0, 0, w, h, w, h)
            return data, profile, transform, None

    # ---------- Тайловка ----------
    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    def _coords(self, H: int, W: int) -> List[Tuple[int, int]]:
        stride = max(1, self.tile - self.overlap)
        xs = list(range(0, max(W - self.tile, 0) + 1, stride))
        ys = list(range(0, max(H - self.tile, 0) + 1, stride))
        if len(xs) == 0:
            xs = [0]
        if len(ys) == 0:
            ys = [0]
        if xs[-1] != max(W - self.tile, 0):
            xs.append(max(W - self.tile, 0))
        if ys[-1] != max(H - self.tile, 0):
            ys.append(max(H - self.tile, 0))
        return [(y, x) for y in ys for x in xs]

    def segment(self, data: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Модель не загружена")

        H, W, _ = data.shape
        coords = self._coords(H, W)
        class_map = np.zeros((H, W), dtype=np.uint8)

        for (y, x) in coords:
            patch = data[y:y + self.tile, x:x + self.tile, :]
            inp = self._to_tensor(patch).to(self.device)
            with torch.no_grad():
                out = self.model(inp)  # [B, C, H, W]
                probs = torch.softmax(out, dim=1).cpu().numpy()
                pred = probs.argmax(axis=1)[0]  # [H, W]
            h, w = patch.shape[:2]
            class_map[y:y + h, x:x + w] = pred[:h, :w]

        return class_map

    # ---------- Экспорт ----------
    def _visualize_bytes(self, class_map: np.ndarray) -> bytes:
        h, w = class_map.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for cid, color in self.class_colors.items():
            rgb[class_map == cid] = color
        img = Image.fromarray(rgb)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        buf.seek(0)
        return buf.getvalue()

    def _write_geotiff_bytes(self, class_map: np.ndarray, profile: dict, transform: object, crs: object) -> bytes:
        out_profile = profile.copy()
        out_profile.update({"count": 1, "dtype": "uint8", "compress": "lzw"})
        out_profile.pop("nodata", None)
        with MemoryFile() as memfile:
            with memfile.open(**out_profile) as dst:
                dst.write(class_map.astype(np.uint8), 1)
                if transform is not None:
                    dst.transform = transform
                if crs is not None:
                    dst.crs = crs
            return memfile.read()

    def _write_tiff_bytes(self, class_map: np.ndarray) -> bytes:
        img = Image.fromarray(class_map.astype(np.uint8), mode="L")
        buf = io.BytesIO()
        img.save(buf, format="TIFF", compression="tiff_lzw")
        buf.seek(0)
        return buf.getvalue()

    def _vectorize_geojson(self, class_map: np.ndarray, transform: object, active_classes: Optional[List[int]]) -> str:
        """
        Генерация GeoJSON с полигонами по маске.
        Для PNG/JPG (crs=None) геометрия будет в пиксельных координатах (по transform-заглушке).
        """
        mask = class_map.astype(np.int32)
        features = []

        target_classes = active_classes if active_classes else sorted(self.class_names.keys())

        for cid in target_classes:
            bin_mask = (mask == cid).astype(np.uint8)
            if bin_mask.sum() == 0:
                continue
            for geom, val in shapes(bin_mask, transform=transform):
                if val != 1:
                    continue
                features.append({
                    "type": "Feature",
                    "geometry": geom,
                    "properties": {
                        "class_id": int(cid),
                        "class_name": self.class_names.get(cid, str(cid)),
                        "color": self.class_colors.get(cid, (0, 0, 0))
                    }
                })

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        return json.dumps(geojson)

    # ---------- Комплексный метод ----------
    def segment_all(self, file_bytes: bytes, active_classes: Optional[List[int]] = None) -> Dict[str, object]:
        """
        Возвращает словарь:
        - visualization: base64 PNG визуализации (по цветам)
        - geotiff: base64 GeoTIFF (классы как uint8)
        - tiff: base64 TIFF (uint8 маска)
        - geojson: строка JSON с полигонами (по активным классам)
        - stats: статистика по классам
        """
        data, profile, transform, crs = self.process_raster(file_bytes)
        class_map = self.segment(data)

        vis_bytes = self._visualize_bytes(class_map)
        vis_b64 = base64.b64encode(vis_bytes).decode("utf-8")

        geotiff_bytes = self._write_geotiff_bytes(class_map, profile, transform, crs)
        geotiff_b64 = base64.b64encode(geotiff_bytes).decode("utf-8")

        tiff_bytes = self._write_tiff_bytes(class_map)
        tiff_b64 = base64.b64encode(tiff_bytes).decode("utf-8")

        geojson_str = self._vectorize_geojson(class_map, transform, active_classes)

        stats = {}
        total = int(class_map.size)
        for cid, name in self.class_names.items():
            pixels = int((class_map == cid).sum())
            percent = round(100.0 * pixels / total, 4) if total > 0 else 0.0
            stats[int(cid)] = {
                "name": name,
                "pixels": pixels,
                "percent": percent
            }

        return {
            "visualization": vis_b64,
            "geotiff": geotiff_b64,
            "tiff": tiff_b64,
            "geojson": geojson_str,
            "stats": stats
        }
