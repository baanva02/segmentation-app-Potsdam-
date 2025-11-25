import numpy as np
import io, json, base64
from typing import Dict, Tuple, List
from PIL import Image

import torch
from rasterio.io import MemoryFile
from rasterio.features import shapes
import rasterio


class PotsdamSegmentationClassifier:
    """Сегментатор аэрофотоснимков Potsdam на PyTorch"""

    def __init__(self, model_path: str, tile: int = 256, overlap: int = 32, batch_size: int = 8):
        self.model_path = model_path
        self.tile = tile
        self.overlap = overlap
        self.batch_size = batch_size
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Легенда Potsdam (фиксированные цвета)
        self.class_colors = {
            0: (0, 0, 0),          # Background - чёрный
            1: (139, 69, 19),      # Buildings - коричневый (saddle brown)
            2: (128, 128, 128),    # Roads - серый
            3: (0, 128, 0),        # Low vegetation (трава) - зелёный
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


    # ---------- Загрузка модели ----------
    def load_model(self, model_class):
        if self.model is not None:
            return
        if model_class is None:
            raise RuntimeError("Для загрузки модели нужен model_class")

        m = model_class()
        checkpoint = torch.load(self.model_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            sd = checkpoint["model_state_dict"]
        else:
            sd = checkpoint

        try:
            m.load_state_dict(sd, strict=True)
        except RuntimeError as e:
            print("Внимание: несовпадение state_dict при strict=True.\n", str(e))
            m.load_state_dict(sd, strict=False)

        self.model = m
        self.model.eval()
        self.model.to(self.device)

    # ---------- Входные данные ----------
    def process_raster(self, file_bytes: bytes) -> Tuple[np.ndarray, dict, object, object]:
        try:
            with MemoryFile(file_bytes) as memfile:
                with memfile.open() as src:
                    profile, transform, crs = src.profile.copy(), src.transform, src.crs
                    # Берём первые три канала, нормализуем
                    data = src.read([1, 2, 3]).transpose(1, 2, 0).astype(np.float32)
                    data = np.nan_to_num(data, nan=0.0, posinf=255.0, neginf=0.0)
                    data /= 255.0
                    return data, profile, transform, crs
        except Exception:
            # Обычный растровый формат без геопривязки (PNG/JPG/TIFF)
            img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            data = np.array(img, dtype=np.float32) / 255.0
            h, w = data.shape[:2]
            profile = {"driver": "GTiff", "dtype": "uint8", "count": 1, "height": h, "width": w, "compress": "lzw"}
            transform = rasterio.transform.from_bounds(0, 0, w, h, w, h)
            return data, profile, transform, None

    # ---------- Тайловка ----------
    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    def _coords(self, H: int, W: int):
        stride = self.tile - self.overlap
        xs = list(range(0, max(W - self.tile, 0) + 1, stride))
        ys = list(range(0, max(H - self.tile, 0) + 1, stride))
        if xs[-1] != W - self.tile:
            xs.append(W - self.tile)
        if ys[-1] != H - self.tile:
            ys.append(H - self.tile)
        return [(y, x) for y in ys for x in xs]

    def segment(self, data: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Модель не загружена")

        H, W, _ = data.shape
        coords = self._coords(H, W)
        mask = np.zeros((H, W), dtype=np.uint8)

        for (y, x) in coords:
            patch = data[y:y + self.tile, x:x + self.tile, :]
            inp = self._to_tensor(patch).to(self.device)
            with torch.no_grad():
                out = self.model(inp)  # (1, K, tile, tile)
                probs = torch.softmax(out, dim=1).cpu().numpy()
                pred = probs.argmax(axis=1)[0]
            mask[y:y + self.tile, x:x + self.tile] = pred[:patch.shape[0], :patch.shape[1]]

        return mask

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
                dst.write(class_map, 1)
                if transform:
                    dst.transform = transform
                if crs:
                    dst.crs = crs
            return memfile.read()

    def _write_tiff_bytes(self, class_map: np.ndarray) -> bytes:
        # Обычный TIFF без геопривязки
        img = Image.fromarray(class_map.astype(np.uint8), mode="L")
        buf = io.BytesIO()
        img.save(buf, format="TIFF", compression="tiff_lzw")
        buf.seek(0)
        return buf.read()

    def _geojson(self, class_map: np.ndarray, transform: object, crs: object) -> str:
        features = []
        if transform is not None:
            from scipy.ndimage import median_filter
            cm = median_filter(class_map, size=3)
            for geom, value in shapes(cm.astype(np.int32), transform=transform):
                if int(value) > 0:
                    features.append({
                        "type": "Feature",
                        "properties": {
                            "class": int(value),
                            "class_name": self.class_names.get(int(value), "Unknown")
                        },
                        "geometry": geom
                    })
        geojson = {"type": "FeatureCollection", "features": features}
        if crs:
            try:
                epsg = crs.to_epsg()
                if epsg:
                    geojson["crs"] = {"type": "name", "properties": {"name": f"EPSG:{epsg}"}}
            except Exception:
                pass
        return json.dumps(geojson)

    def _stats(self, class_map: np.ndarray) -> Dict[int, Dict[str, float]]:
        counts = np.bincount(class_map.flatten(), minlength=len(self.class_names))
        total = counts.sum() if counts.sum() > 0 else 1
        stats = {}
        for cid, count in enumerate(counts):
            if count > 0:
                stats[cid] = {
                    "name": self.class_names.get(cid, "Unknown"),
                    "pixels": int(count),
                    "percent": round(100 * count / total, 2)
                }
        return stats

    # ---------- Полный пайплайн ----------
    def segment_all(self, file_bytes: bytes, active_classes: List[int] = None) -> Dict[str, str]:
        data, profile, transform, crs = self.process_raster(file_bytes)
        class_map = self.segment(data)

        # Фильтрация классов (всё, что не в списке, уходит в 0/Background)
        if active_classes is not None:
            class_map = np.where(np.isin(class_map, active_classes), class_map, 0)

        viz_bytes = self._visualize_bytes(class_map)
        geotiff_bytes = self._write_geotiff_bytes(class_map, profile, transform, crs)
        tiff_bytes = self._write_tiff_bytes(class_map)
        geojson_str = self._geojson(class_map, transform, crs)
        stats = self._stats(class_map)

        return {
            "visualization": base64.b64encode(viz_bytes).decode("utf-8"),
            "geotiff": base64.b64encode(geotiff_bytes).decode("utf-8"),
            "tiff": base64.b64encode(tiff_bytes).decode("utf-8"),
            "geojson": geojson_str,
            "stats": stats
        }
