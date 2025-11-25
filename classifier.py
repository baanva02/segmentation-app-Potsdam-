import os
import io
import json
import base64
from typing import Dict, Tuple, List

import numpy as np
from PIL import Image

import torch
import requests
import rasterio
from rasterio.io import MemoryFile
from rasterio.features import shapes


class PotsdamSegmentationClassifier:
    """Сегментатор аэрофотоснимков Potsdam на PyTorch с автозагрузкой модели с Google Drive"""

    def __init__(
        self,
        model_path: str = "models/best_unetpp_efficientnetb0.pth",
        tile: int = 256,
        overlap: int = 32,
        batch_size: int = 8,
        google_drive_file_id: str = None,
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
            0: (0, 0, 0),          # Background - чёрный
            1: (139, 69, 19),      # Buildings - коричневый
            2: (128, 128, 128),    # Roads - серый
            3: (0, 128, 0),        # Low vegetation - зелёный
            4: (0, 255, 0),        # Trees - ярко-зелёный
            5: (128, 0, 128),      # Cars - фиолетовый
        }
        self.class_names = {
            0: "Background",
            1: "Building",
            2: "Road",
            3: "Low vegetation",
            4: "Tree",
            5: "Car"
        }

        os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)

    # ---------- Загрузка модели с Google Drive ----------
    def _download_model_from_gdrive(self, file_id: str, dest_path: str) -> None:
        session = requests.Session()
        base_url = "https://docs.google.com/uc?export=download"

        response = session.get(base_url, params={"id": file_id}, stream=True)
        token = None
        for k, v in response.cookies.items():
            if k.startswith("download_warning"):
                token = v
                break

        if token:
            response = session.get(base_url, params={"id": file_id, "confirm": token}, stream=True)

        if response.status_code != 200:
            raise RuntimeError(f"Ошибка загрузки модели: HTTP {response.status_code}")

        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(32768):
                if chunk:
                    f.write(chunk)

    def ensure_model(self) -> None:
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
            transform = rasterio.transform.from_bounds(0, 0, w, h, w, h)
            return data, profile, transform, None

    # ---------- Тайловка ----------
    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    def _coords(self, H: int, W: int) -> List[Tuple[int, int]]:
        stride = max(1, self.tile - self.overlap)
        xs = list(range(0, max(W - self.tile, 0) + 1, stride))
        ys = list(range(0, max(H - self.tile, 0) + 1, stride))
        if len(xs) == 0: xs = [0]
        if len(ys) == 0: ys = [0]
        if xs[-1] != max(W - self.tile, 0): xs.append(max(W - self.tile, 0))
        if ys[-1] != max(H - self.tile, 0): ys.append(max(H - self.tile, 0))
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
                out = self.model(inp)
                probs = torch.softmax(out, dim=1).cpu().numpy()
                pred = probs.argmax(axis=1)[0]
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
        buf