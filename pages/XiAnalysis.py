import sys
import os
import math
import json
import base64
import traceback
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import tifffile as tiff

from PySide6.QtCore import Qt, QRect, QPoint, Signal, QObject, QThread, Slot, QUrl
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QSplitter,
    QLabel,
    QPushButton,
    QTextEdit,
    QProgressBar,
    QSlider,
    QComboBox,
    QGroupBox,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QMessageBox,
    QScrollArea,
    QDialog,
    QListWidget,
    QInputDialog,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QTabWidget,
    QSizePolicy,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


DEFAULT_DIR = "/home/alonso/Desktop/"
DEFAULT_VIDEO_DIR = os.path.join(DEFAULT_DIR, "videos")


# ============================================================
# Utility functions
# ============================================================

def ensure_gray(img):
    """Return grayscale image. If RGB/RGBA/multichannel, take channel 0."""
    img = np.asarray(img)
    if img.ndim == 3:
        return img[..., 0]
    return img


def to_uint8(img):
    """Convert image to uint8 for OpenCV drawing/tracking preview."""
    img = np.asarray(img)
    if img.dtype == np.uint16:
        return (img >> 8).astype(np.uint8)

    if img.dtype == np.uint8:
        return img

    arr = img.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(arr[np.isfinite(arr)], [2, 98]) if np.isfinite(arr).any() else (0, 1)
    if hi <= lo:
        hi = lo + 1e-6
    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    return (arr * 255 + 0.5).astype(np.uint8)


def image_to_qpixmap_gray(img_u8):
    """Create QPixmap from a uint8 grayscale numpy image."""
    img_u8 = np.ascontiguousarray(img_u8)
    h, w = img_u8.shape
    qimg = QImage(img_u8.data, w, h, img_u8.strides[0], QImage.Format_Grayscale8)
    return QPixmap.fromImage(qimg.copy())


def encode_png_u8(img_u8):
    ok, buf = cv2.imencode(".png", img_u8)
    if not ok:
        raise RuntimeError("No se pudo codificar PNG.")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def decode_png_to_u8(b64):
    arr = np.frombuffer(base64.b64decode(b64.encode("ascii")), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError("No se pudo decodificar PNG guardado.")
    return img


def sanitize_inner_rel(inners, bw, bh):
    fixed = []
    for inn in inners:
        name = inn.get("name")
        ix, iy, iw, ih = map(int, inn["rect"])
        ix = max(0, min(ix, max(0, bw - 1)))
        iy = max(0, min(iy, max(0, bh - 1)))
        iw = max(1, min(iw, max(1, bw - ix)))
        ih = max(1, min(ih, max(1, bh - iy)))
        fixed.append({"name": name, "rect": (ix, iy, iw, ih)})
    return fixed


def dedup_inner_rects(inners):
    seen = set()
    dedup = []
    for inn in inners:
        key = tuple(map(int, inn["rect"]))
        if key in seen:
            ix, iy, iw, ih = key
            ix += 2
            iy += 2
            inn = {**inn, "rect": (ix, iy, iw, ih)}
            key = (ix, iy, iw, ih)
        seen.add(key)
        dedup.append(inn)
    return dedup


def build_saved_payload_from_rois(img0_u8, rois_local):
    payload = {"image_shape": list(img0_u8.shape), "rois": []}

    for roi in rois_local:
        bx, by, bw, bh = map(int, roi["rect"])
        big_tpl = img0_u8[by:by + bh, bx:bx + bw].copy()

        inners_out = []
        for inn in roi.get("inners", []):
            ix, iy, iw, ih = map(int, inn["rect"])
            ix = max(0, min(ix, max(0, bw - 1)))
            iy = max(0, min(iy, max(0, bh - 1)))
            iw = max(1, min(iw, max(1, bw - ix)))
            ih = max(1, min(ih, max(1, bh - iy)))
            inner_tpl = big_tpl[iy:iy + ih, ix:ix + iw].copy()
            inners_out.append({
                "name": inn.get("name") or None,
                "rect": [int(ix), int(iy), int(iw), int(ih)],
                "inner_template_png_b64": encode_png_u8(inner_tpl),
            })

        payload["rois"].append({
            "name": roi["name"],
            "rect": [int(bx), int(by), int(bw), int(bh)],
            "inners": inners_out,
            "template_png_b64": encode_png_u8(big_tpl),
        })

    return payload


def rois_from_saved_on_new_img0(saved_payload, img0_u8, img0_f32):
    rois_local = []

    for r in saved_payload.get("rois", []):
        name = r["name"]
        bx0, by0, bw, bh = map(int, r["rect"])
        big_tpl_u8 = decode_png_to_u8(r["template_png_b64"]).astype(np.float32)

        if img0_f32.shape[0] < big_tpl_u8.shape[0] or img0_f32.shape[1] < big_tpl_u8.shape[1]:
            nx, ny = bx0, by0
        else:
            res_big = cv2.matchTemplate(img0_f32, big_tpl_u8, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(res_big)
            nx, ny = int(max_loc[0]), int(max_loc[1])

        big_now = img0_u8[ny:ny + bh, nx:nx + bw].copy().astype(np.float32)

        inners_rel = []
        for inn in r.get("inners", []):
            ix, iy, iw, ih = map(int, inn["rect"])

            if inn.get("inner_template_png_b64"):
                inner_tpl = decode_png_to_u8(inn["inner_template_png_b64"]).astype(np.float32)
                if (
                    inner_tpl.shape[0] > 0 and inner_tpl.shape[1] > 0 and
                    big_now.shape[0] >= inner_tpl.shape[0] and
                    big_now.shape[1] >= inner_tpl.shape[1]
                ):
                    res_in = cv2.matchTemplate(big_now, inner_tpl, cv2.TM_CCOEFF_NORMED)
                    _, _, _, in_loc = cv2.minMaxLoc(res_in)
                    ix, iy = int(in_loc[0]), int(in_loc[1])

            inners_rel.append({"name": inn.get("name"), "rect": (ix, iy, iw, ih)})

        inners_rel = sanitize_inner_rel(dedup_inner_rects(inners_rel), bw, bh)
        rois_local.append({"name": name, "rect": (nx, ny, bw, bh), "inners": inners_rel})

    return rois_local


def align_metadata_to_frames(metadata, n_frames):
    """Align metadata DataFrame/CSV to frames."""
    if metadata is None:
        return None

    if not isinstance(metadata, pd.DataFrame):
        try:
            metadata = pd.read_csv(metadata)
        except Exception:
            return None

    cols_norm = {c: c.strip() for c in metadata.columns}
    metadata = metadata.rename(columns=cols_norm)
    lower_map = {c.lower(): c for c in metadata.columns}

    if "frame" in lower_map:
        col = lower_map["frame"]
        md = metadata.copy()
        try:
            md[col] = md[col].astype(int)
        except Exception:
            try:
                md[col] = pd.to_numeric(md[col], errors="coerce").astype("Int64")
            except Exception:
                return None
        md = md.rename(columns={col: "frame"})
        return md

    if len(metadata) == n_frames:
        md = metadata.copy()
        md.insert(0, "frame", np.arange(n_frames, dtype=int))
        return md

    for candidate in ["Timestamp", "timestamp", "time_stamp", "TimeStamp"]:
        if candidate in metadata.columns and len(metadata) == n_frames:
            md = metadata.copy()
            md.insert(0, "frame", np.arange(n_frames, dtype=int))
            return md

    return None


def list_tiff_files(folder):
    files = []
    for ext in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        files.extend(Path(folder).glob(ext))
    return sorted([str(p) for p in files])


def compute_mean_in_tracked_rois(processed_stack, roi_tracks, metadata=None, out_dir=None):
    n_frames, height, width = processed_stack.shape
    rows = []

    for roi in roi_tracks:
        name = roi["name"]
        for frame_id, x, y, w, h in roi["coords"]:
            if frame_id < 0 or frame_id >= n_frames:
                continue

            y1, y2 = max(0, y), min(y + h, height)
            x1, x2 = max(0, x), min(x + w, width)
            roi_data = processed_stack[frame_id][y1:y2, x1:x2]
            mean_val = float(np.nanmean(roi_data)) if roi_data.size > 0 else np.nan

            rows.append({
                "frame": int(frame_id),
                "roi_name": name,
                "mean_value": mean_val,
            })

    df_long = pd.DataFrame(rows).dropna(subset=["mean_value"])

    if df_long.empty:
        return pd.DataFrame(), None

    df_wide = df_long.pivot(index="frame", columns="roi_name", values="mean_value").reset_index()

    md_aligned = align_metadata_to_frames(metadata, n_frames)
    if md_aligned is not None:
        roi_cols = set(df_wide.columns) - {"frame"}
        rename_map = {}
        for c in md_aligned.columns:
            if c != "frame" and c in roi_cols:
                rename_map[c] = f"meta_{c}"
        if rename_map:
            md_aligned = md_aligned.rename(columns=rename_map)
        df_final = df_wide.merge(md_aligned, on="frame", how="left")
    else:
        df_final = df_wide

    csv_path = None
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "roi_mean_values_with_metadata.csv")
        df_final.to_csv(csv_path, index=False)


    return df_final, csv_path


def finite_percentile_limits(values, p_low=20.0, p_high=80.0, symmetric=False):
    """
    Robust vmin/vmax for visualization.
    values can be a full image or only target ROI pixels.
    """
    vals = np.asarray(values, dtype=np.float32).ravel()
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return -1.0 if symmetric else 0.0, 1.0

    if symmetric:
        lo = float(np.percentile(vals, max(0.0, min(50.0, p_low))))
        hi = float(np.percentile(vals, min(100.0, max(50.0, p_high))))
        m = max(abs(lo), abs(hi), 1e-6)
        return -m, m

    vmin = float(np.percentile(vals, p_low))
    vmax = float(np.percentile(vals, p_high))

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1e-6

    return vmin, vmax


def mask_from_roi_tracks(shape_hw, roi_tracks, frame_idx):
    """
    Build a boolean mask from tracked ROIs for the selected frame.
    If a big ROI has inner ROIs, the inner ROIs are used for target scaling.
    Otherwise the big ROI is used.
    """
    if not roi_tracks:
        return None

    H, W = shape_hw
    mask = np.zeros((H, W), dtype=bool)

    for roi in roi_tracks:
        coords = roi.get("coords", [])
        if not coords:
            continue

        selected = None
        for c in coords:
            if int(c[0]) == int(frame_idx):
                selected = c
                break

        if selected is None:
            # Fallback: same index position if available
            if 0 <= frame_idx < len(coords):
                selected = coords[frame_idx]
            else:
                selected = coords[0]

        _, x0, y0, w0, h0 = map(int, selected)
        inners = roi.get("inners_rel", [])

        if inners:
            for inner in inners:
                ix, iy, iw, ih = map(int, inner["rect"])
                x1 = max(0, min(W, x0 + ix))
                y1 = max(0, min(H, y0 + iy))
                x2 = max(0, min(W, x1 + iw))
                y2 = max(0, min(H, y1 + ih))
                if x2 > x1 and y2 > y1:
                    mask[y1:y2, x1:x2] = True
        else:
            x1 = max(0, min(W, x0))
            y1 = max(0, min(H, y0))
            x2 = max(0, min(W, x0 + w0))
            y2 = max(0, min(H, y0 + h0))
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = True

    return mask if mask.any() else None


def mask_from_saved_rois(shape_hw, saved_rois):
    """
    Fallback mask from saved ROI payload when tracking has not been run yet.
    This only corresponds to the original saved coordinates, not motion-tracked coordinates.
    """
    if not saved_rois or not saved_rois.get("rois"):
        return None

    H, W = shape_hw
    mask = np.zeros((H, W), dtype=bool)

    for roi in saved_rois.get("rois", []):
        x0, y0, w0, h0 = map(int, roi.get("rect", [0, 0, 0, 0]))
        inners = roi.get("inners", [])

        if inners:
            for inner in inners:
                ix, iy, iw, ih = map(int, inner.get("rect", [0, 0, 0, 0]))
                x1 = max(0, min(W, x0 + ix))
                y1 = max(0, min(H, y0 + iy))
                x2 = max(0, min(W, x1 + iw))
                y2 = max(0, min(H, y1 + ih))
                if x2 > x1 and y2 > y1:
                    mask[y1:y2, x1:x2] = True
        else:
            x1 = max(0, min(W, x0))
            y1 = max(0, min(H, y0))
            x2 = max(0, min(W, x0 + w0))
            y2 = max(0, min(H, y0 + h0))
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = True

    return mask if mask.any() else None


def auto_object_mask_from_image(img, threshold_pct=20, keep_mode="brighter", clean_px=5, largest_only=True):
    """
    Build an automatic object mask from an image.

    This is intended for visualization only. It is useful when the processed image
    has artificial high values in low-signal background pixels.
    """
    if img is None:
        return None

    arr = ensure_gray(img).astype(np.float32)
    if arr.ndim != 2:
        arr = ensure_gray(arr).astype(np.float32)

    finite = np.isfinite(arr)
    vals = arr[finite]
    if vals.size == 0:
        return None

    threshold_pct = float(np.clip(threshold_pct, 0, 100))
    thr = float(np.percentile(vals, threshold_pct))

    if keep_mode == "darker":
        mask = (arr <= thr) & finite
    else:
        mask = (arr >= thr) & finite

    if not mask.any():
        return None

    mask_u8 = (mask.astype(np.uint8) * 255)

    clean_px = int(clean_px)
    if clean_px > 0:
        k = max(1, clean_px)
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

    if largest_only:
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask_u8 > 0).astype(np.uint8), 8)
        if n_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            if areas.size > 0:
                largest_label = 1 + int(np.argmax(areas))
                mask_u8 = ((labels == largest_label).astype(np.uint8) * 255)

    out = mask_u8 > 0
    return out if out.any() else None


def combine_masks(mask_a, mask_b, mode="or"):
    """Combine two boolean masks with safe fallbacks."""
    if mask_a is None:
        return mask_b
    if mask_b is None:
        return mask_a
    if mask_a.shape != mask_b.shape:
        return mask_a
    if mode == "and":
        out = mask_a & mask_b
    else:
        out = mask_a | mask_b
    return out if out.any() else None


# ============================================================
# Matplotlib canvas
# ============================================================

class MplCanvas(FigureCanvas):
    def __init__(self, figsize=(6, 5), dpi=100):
        self.figure = Figure(figsize=figsize, dpi=dpi)
        self.ax = self.figure.add_subplot(111)
        super().__init__(self.figure)


# ============================================================
# ROI selection widgets
# ============================================================

class RoiImageWidget(QWidget):
    roi_created = Signal(dict)

    def __init__(self, img_u8, scale=3.0, parent=None):
        super().__init__(parent)
        self.scale = float(scale)
        self.start_point = None
        self.current_rect = None
        self.rois = []
        self.set_image(img_u8)

    def set_image(self, img_u8):
        """Replace displayed image without recreating the widget."""
        self.img_u8 = np.asarray(img_u8)
        self.pixmap_original = image_to_qpixmap_gray(self.img_u8)
        self.pixmap_scaled = self.pixmap_original.scaled(
            int(self.pixmap_original.width() * self.scale),
            int(self.pixmap_original.height() * self.scale),
            Qt.IgnoreAspectRatio,
            Qt.FastTransformation,
        )
        self.setMinimumSize(self.pixmap_scaled.size())
        self.resize(self.pixmap_scaled.size())
        self.updateGeometry()
        self.update()

    def add_existing_rois(self, rois):
        self.rois = list(rois)
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_point = event.position().toPoint()
            self.current_rect = QRect(self.start_point, self.start_point)
            self.update()

    def mouseMoveEvent(self, event):
        if self.start_point is not None:
            end = event.position().toPoint()
            self.current_rect = QRect(self.start_point, end).normalized()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.start_point is not None:
            end = event.position().toPoint()
            rect_scaled = QRect(self.start_point, end).normalized()
            self.start_point = None
            self.current_rect = None

            if rect_scaled.width() < 2 or rect_scaled.height() < 2:
                self.update()
                return

            x = int(math.floor(rect_scaled.x() / self.scale))
            y = int(math.floor(rect_scaled.y() / self.scale))
            w = int(math.ceil(rect_scaled.width() / self.scale))
            h = int(math.ceil(rect_scaled.height() / self.scale))

            H, W = self.img_u8.shape
            x = max(0, min(x, W - 1))
            y = max(0, min(y, H - 1))
            w = max(1, min(w, W - x))
            h = max(1, min(h, H - y))

            roi = {"name": None, "rect": (x, y, w, h)}
            self.roi_created.emit(roi)
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.pixmap_scaled)

        pen = QPen(QColor(255, 0, 0), 2)
        painter.setPen(pen)
        painter.setFont(QFont("Arial", 10))

        for idx, roi in enumerate(self.rois, start=1):
            x, y, w, h = map(int, roi["rect"])
            rect = QRect(
                int(x * self.scale),
                int(y * self.scale),
                int(w * self.scale),
                int(h * self.scale),
            )
            painter.drawRect(rect)
            label = roi.get("name") or str(idx)
            if "ref_frame" in roi:
                label = f"{label} [f{int(roi.get('ref_frame', 0))}]"
            painter.drawText(rect.x() + 3, max(12, rect.y() - 3), label)

        if self.current_rect is not None:
            painter.drawRect(self.current_rect)

        painter.end()


class RoiSelectionDialog(QDialog):
    def __init__(
        self,
        img_u8,
        title="Select ROIs",
        scale=3.0,
        name_required=True,
        parent=None,
        frame_loader=None,
        n_frames=1,
        initial_frame=0,
    ):
        """
        ROI selector.

        If frame_loader is provided, the dialog becomes a sequence navigator.
        ROIs are drawn on the currently visible frame and each ROI stores ref_frame.
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1150, 850)
        self.scale = scale
        self.name_required = name_required
        self.rois = []

        self.frame_loader = frame_loader
        self.n_frames = max(1, int(n_frames or 1))
        self.current_frame = max(0, min(int(initial_frame or 0), self.n_frames - 1))
        self.is_sequence = self.frame_loader is not None and self.n_frames > 1

        if self.frame_loader is not None:
            self.img_u8 = self.load_frame_u8(self.current_frame)
        else:
            self.img_u8 = np.asarray(img_u8)

        layout = QHBoxLayout(self)

        self.image_widget = RoiImageWidget(self.img_u8, scale=scale)
        self.image_widget.roi_created.connect(self.on_roi_created)

        self.scroll = QScrollArea()
        self.scroll.setWidget(self.image_widget)
        self.scroll.setWidgetResizable(False)

        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel.setFixedWidth(330)

        help_lines = [
            "Arrastra con el mouse para dibujar ROI.",
            "Usa las barras para desplazarte.",
        ]
        if self.is_sequence:
            help_lines.extend([
                "Navega con el slider para elegir el mejor frame.",
                "Cada ROI guarda el frame visible como template de tracking.",
                "Teclas: ←/→ para cambiar frame.",
            ])
        help_lines.append("Al soltar el mouse te pedirá nombre.")
        help_text = QLabel("\n".join(help_lines))
        help_text.setWordWrap(True)

        self.frame_label = QLabel()
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(self.n_frames - 1)
        self.frame_slider.setValue(self.current_frame)
        self.frame_slider.setEnabled(self.is_sequence)
        self.frame_slider.valueChanged.connect(self.on_frame_slider_changed)

        self.btn_prev = QPushButton("← Previous frame")
        self.btn_prev.clicked.connect(lambda: self.go_to_frame(self.current_frame - 1))
        self.btn_prev.setEnabled(self.is_sequence)

        self.btn_next = QPushButton("Next frame →")
        self.btn_next.clicked.connect(lambda: self.go_to_frame(self.current_frame + 1))
        self.btn_next.setEnabled(self.is_sequence)

        frame_buttons = QHBoxLayout()
        frame_buttons.addWidget(self.btn_prev)
        frame_buttons.addWidget(self.btn_next)

        self.roi_list = QListWidget()

        self.btn_undo = QPushButton("Undo last ROI")
        self.btn_undo.clicked.connect(self.undo_last)

        self.btn_clear_frame = QPushButton("Clear ROIs in current frame")
        self.btn_clear_frame.clicked.connect(self.clear_current_frame)
        self.btn_clear_frame.setEnabled(self.is_sequence)

        self.btn_clear = QPushButton("Clear all")
        self.btn_clear.clicked.connect(self.clear_all)

        self.btn_done = QPushButton("Done")
        self.btn_done.clicked.connect(self.accept)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)

        panel_layout.addWidget(QLabel("Frame navigation"))
        panel_layout.addWidget(self.frame_label)
        panel_layout.addWidget(self.frame_slider)
        panel_layout.addLayout(frame_buttons)
        panel_layout.addWidget(help_text)
        panel_layout.addWidget(QLabel("ROIs"))
        panel_layout.addWidget(self.roi_list, stretch=1)
        panel_layout.addWidget(self.btn_undo)
        panel_layout.addWidget(self.btn_clear_frame)
        panel_layout.addWidget(self.btn_clear)
        panel_layout.addWidget(self.btn_done)
        panel_layout.addWidget(self.btn_cancel)

        layout.addWidget(self.scroll, stretch=1)
        layout.addWidget(panel)

        self.refresh_frame_label()
        self.refresh_list()

    def load_frame_u8(self, frame_idx):
        if self.frame_loader is None:
            return np.asarray(self.img_u8)
        frame_idx = max(0, min(int(frame_idx), self.n_frames - 1))
        img = self.frame_loader(frame_idx)
        return to_uint8(ensure_gray(img))

    def refresh_frame_label(self):
        if self.is_sequence:
            self.frame_label.setText(f"Current frame: {self.current_frame} / {self.n_frames - 1}")
        else:
            self.frame_label.setText("Current frame: single image")

    def on_frame_slider_changed(self, value):
        self.go_to_frame(int(value), update_slider=False)

    def go_to_frame(self, frame_idx, update_slider=True):
        if not self.is_sequence:
            return

        frame_idx = max(0, min(int(frame_idx), self.n_frames - 1))
        if frame_idx == self.current_frame and self.image_widget.img_u8 is not None:
            self.refresh_frame_label()
            self.refresh_list()
            return

        try:
            self.current_frame = frame_idx
            self.img_u8 = self.load_frame_u8(frame_idx)
            self.image_widget.set_image(self.img_u8)
            if update_slider and self.frame_slider.value() != frame_idx:
                self.frame_slider.blockSignals(True)
                self.frame_slider.setValue(frame_idx)
                self.frame_slider.blockSignals(False)
            self.refresh_frame_label()
            self.refresh_list()
        except Exception as e:
            QMessageBox.critical(self, "Frame load error", str(e))

    def keyPressEvent(self, event):
        if self.is_sequence and event.key() in (Qt.Key_Left, Qt.Key_A):
            self.go_to_frame(self.current_frame - 1)
            return
        if self.is_sequence and event.key() in (Qt.Key_Right, Qt.Key_D):
            self.go_to_frame(self.current_frame + 1)
            return
        super().keyPressEvent(event)

    def on_roi_created(self, roi):
        default_name = f"ROI_{len(self.rois) + 1}"
        name, ok = QInputDialog.getText(self, "ROI name", "Name:", text=default_name)

        if not ok:
            return

        name = name.strip()

        if self.name_required and not name:
            name = default_name

        roi["name"] = name if name else None
        roi["ref_frame"] = int(self.current_frame)
        self.rois.append(roi)
        self.refresh_list()

    def _visible_rois_for_current_frame(self):
        if not self.is_sequence:
            return self.rois
        return [r for r in self.rois if int(r.get("ref_frame", 0)) == int(self.current_frame)]

    def refresh_list(self):
        self.roi_list.clear()
        for idx, roi in enumerate(self.rois, start=1):
            x, y, w, h = roi["rect"]
            name = roi.get("name") or f"ROI_{idx}"
            ref = int(roi.get("ref_frame", 0))
            self.roi_list.addItem(f"{idx}. {name}: frame={ref}, x={x}, y={y}, w={w}, h={h}")
        self.image_widget.add_existing_rois(self._visible_rois_for_current_frame())

    def undo_last(self):
        if self.rois:
            self.rois.pop()
            self.refresh_list()

    def clear_current_frame(self):
        if not self.is_sequence:
            return
        self.rois = [r for r in self.rois if int(r.get("ref_frame", 0)) != int(self.current_frame)]
        self.refresh_list()

    def clear_all(self):
        self.rois.clear()
        self.refresh_list()

    def get_rois(self):
        return self.rois

def fit_scale_to_screen(img_w, img_h, requested=3.0, screen_frac=0.8):
    app = QApplication.instance()
    screen = app.primaryScreen().availableGeometry()
    max_w = int(screen.width() * screen_frac)
    max_h = int(screen.height() * screen_frac)

    if img_w * requested <= max_w and img_h * requested <= max_h:
        return requested

    scale_w = max_w / max(1, img_w)
    scale_h = max_h / max(1, img_h)
    return max(1.0, min(scale_w, scale_h))


# ============================================================
# Tracking worker
# ============================================================

class TrackingWorker(QObject):
    progress = Signal(int, str)
    log = Signal(str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, tiff_files, processed_stack, metadata, rois_local, video_path, saved_rois_enabled=True):
        super().__init__()
        self.tiff_files = tiff_files
        self.processed_stack = processed_stack
        self.metadata = metadata
        self.rois_local = rois_local
        self.video_path = video_path
        self.saved_rois_enabled = saved_rois_enabled

    @Slot()
    def run(self):
        try:
            result = self.track_and_measure()
            self.finished.emit(result)
        except Exception:
            self.error.emit(traceback.format_exc())

    def track_and_measure(self):
        if not self.tiff_files:
            raise ValueError("No hay TIFF files.")

        if self.processed_stack is None:
            raise ValueError("No hay processed_stack.")

        n_frames_proc = self.processed_stack.shape[0]
        n_tiffs = len(self.tiff_files)
        n_frames = min(n_frames_proc, n_tiffs)

        if n_frames <= 0:
            raise ValueError("No hay frames comunes entre processed_stack y TIFF folder.")

        if n_tiffs != n_frames_proc:
            self.log.emit(
                f"Advertencia: processed_stack tiene {n_frames_proc} frames, TIFF folder tiene {n_tiffs}. "
                f"Usaré {n_frames}."
            )

        img0_raw = ensure_gray(tiff.imread(self.tiff_files[0]))
        H, W = img0_raw.shape

        roi_tracks = []
        total_rois = len(self.rois_local)

        for r_idx, roi in enumerate(self.rois_local, start=1):
            name = roi["name"]
            x, y, w, h = map(int, roi["rect"])

            # IMPORTANT: the ROI may have been drawn on any frame selected by the user.
            # Use that frame as template reference instead of always using frame 0.
            ref_frame = int(roi.get("ref_frame", 0))
            ref_frame = max(0, min(ref_frame, n_frames - 1))
            img_ref_raw = ensure_gray(tiff.imread(self.tiff_files[ref_frame]))
            template = img_ref_raw[y:y + h, x:x + w].astype(np.float32)

            if template.size == 0:
                self.log.emit(f"ROI vacía omitida: {name} en frame {ref_frame}")
                continue

            self.log.emit(f"ROI '{name}': template tomado del frame {ref_frame}, rect=({x},{y},{w},{h})")
            coords = []

            for i in range(n_frames):
                img = ensure_gray(tiff.imread(self.tiff_files[i])).astype(np.float32)

                if img.shape[0] < template.shape[0] or img.shape[1] < template.shape[1]:
                    coords.append((i, x, y, w, h))
                else:
                    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
                    _, _, _, max_loc = cv2.minMaxLoc(res)
                    coords.append((i, int(max_loc[0]), int(max_loc[1]), w, h))

                done = ((r_idx - 1) * n_frames + i + 1) / max(1, total_rois * n_frames)
                self.progress.emit(int(done * 50), f"Tracking ROI {r_idx}/{total_rois}, frame {i + 1}/{n_frames}")

            roi_tracks.append({
                "name": name,
                "coords": coords,
                "inners_rel": roi.get("inners", []),
                "ref_frame": ref_frame,
            })

        # Video output
        os.makedirs(os.path.dirname(self.video_path), exist_ok=True)
        out = cv2.VideoWriter(self.video_path, cv2.VideoWriter_fourcc(*"XVID"), 10, (W, H))

        for i in range(n_frames):
            img = ensure_gray(tiff.imread(self.tiff_files[i]))
            frame8 = to_uint8(img)
            frame_bgr = cv2.cvtColor(frame8, cv2.COLOR_GRAY2BGR)

            for roi in roi_tracks:
                _, x0, y0, w0, h0 = roi["coords"][i]
                cv2.rectangle(frame_bgr, (x0, y0), (x0 + w0, y0 + h0), (0, 0, 255), 2)
                cv2.putText(
                    frame_bgr,
                    roi["name"],
                    (x0, max(10, y0 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

                for idx, inner in enumerate(roi.get("inners_rel", []), start=1):
                    ix, iy, iw, ih = map(int, inner["rect"])
                    cx, cy = x0 + ix, y0 + iy
                    cv2.rectangle(frame_bgr, (cx, cy), (cx + iw, cy + ih), (0, 255, 255), 2)
                    label_text = inner.get("name") or str(idx)
                    cv2.putText(
                        frame_bgr,
                        label_text,
                        (cx, max(10, cy - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

            out.write(frame_bgr)
            self.progress.emit(50 + int(((i + 1) / n_frames) * 25), f"Writing video {i + 1}/{n_frames}")

        out.release()

        # Save ROI payload from first frame, using tracked coords at frame 0
        img0_u8 = to_uint8(img0_raw)
        saved_rois_input = []
        for r in roi_tracks:
            _, bx, by, bw, bh = r["coords"][0]
            inners_rel = sanitize_inner_rel(r.get("inners_rel", []), bw, bh)
            saved_rois_input.append({
                "name": r["name"],
                "rect": (bx, by, bw, bh),
                "inners": inners_rel,
            })

        saved_payload = build_saved_payload_from_rois(img0_u8, saved_rois_input)

        # Measurement tracks: inner ROIs if present; otherwise big ROI
        measure_tracks = []
        for roi in roi_tracks:
            big_name = roi["name"]
            inners = roi.get("inners_rel", [])

            if inners:
                for idx, inner in enumerate(inners, start=1):
                    ix, iy, iw, ih = map(int, inner["rect"])
                    series = []
                    for frame_id, x0, y0, w0, h0 in roi["coords"]:
                        series.append((frame_id, x0 + ix, y0 + iy, iw, ih))
                    metric_name = inner.get("name") or f"{big_name}_small{idx}"
                    measure_tracks.append({"name": metric_name, "coords": series})
            else:
                measure_tracks.append({"name": big_name, "coords": roi["coords"]})

        out_dir = os.path.dirname(self.video_path)
        df_final, csv_path = compute_mean_in_tracked_rois(
            self.processed_stack,
            measure_tracks,
            self.metadata,
            out_dir=out_dir,
        )

        # Save tracking JSON
        tracking_json_path = os.path.join(out_dir, "roi_tracks.json")
        serializable_tracks = []
        for roi in roi_tracks:
            serializable_tracks.append({
                "name": roi["name"],
                "ref_frame": int(roi.get("ref_frame", 0)),
                "coords": [list(map(int, c)) for c in roi["coords"]],
                "inners_rel": [
                    {"name": inn.get("name"), "rect": list(map(int, inn["rect"]))}
                    for inn in roi.get("inners_rel", [])
                ],
            })

        with open(tracking_json_path, "w", encoding="utf-8") as f:
            json.dump(serializable_tracks, f, ensure_ascii=False, indent=2)

        saved_rois_path = os.path.join(out_dir, "saved_rois.json")
        with open(saved_rois_path, "w", encoding="utf-8") as f:
            json.dump(saved_payload, f, ensure_ascii=False, indent=2)

        self.progress.emit(100, "Done")

        return {
            "roi_tracks": roi_tracks,
            "saved_payload": saved_payload,
            "video_path": self.video_path,
            "csv_path": csv_path,
            "tracking_json_path": tracking_json_path,
            "saved_rois_path": saved_rois_path,
            "df_final": df_final,
        }


# ============================================================
# Main app
# ============================================================

class RoiTrackingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spectral ROI Tracking - PySide6")
        self.resize(1350, 850)

        # State replaces st.session_state
        self.processed_stack = None
        self.processed_path = None
        self.metadata = None
        self.metadata_path = None
        self.tiff_folder = None
        self.tiff_files = []
        self.roi_tracks = None
        self.saved_rois = None
        self.result_df = None
        self.video_path = None
        self.csv_path = None
        self.tracking_json_path = None
        self.saved_rois_path = None
        self.last_export_dir = None

        self.thread = None
        self.worker = None

        self.init_ui()

    def init_ui(self):
        root = QWidget()
        main_layout = QVBoxLayout(root)

        title = QLabel("Spectral analysis of Ximea cameras")
        title.setStyleSheet("font-size: 22px; font-weight: bold;")
        subtitle = QLabel("AG-Santos Neurovascular Research Laboratory")
        subtitle.setStyleSheet("font-size: 13px; color: gray;")

        main_layout.addWidget(title)
        main_layout.addWidget(subtitle)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter, stretch=1)

        # Left panel
        left = QWidget()
        left_layout = QVBoxLayout(left)

        files_group = QGroupBox("Input files")
        files_layout = QGridLayout(files_group)

        self.processed_edit = QLineEdit()
        self.processed_edit.setReadOnly(True)
        self.btn_load_processed = QPushButton("Load processed stack .npy")
        self.btn_load_processed.clicked.connect(self.load_processed_stack)

        self.tiff_edit = QLineEdit()
        self.tiff_edit.setReadOnly(True)
        self.btn_load_tiff = QPushButton("Load TIFF folder")
        self.btn_load_tiff.clicked.connect(self.load_tiff_folder)

        self.metadata_edit = QLineEdit()
        self.metadata_edit.setReadOnly(True)
        self.btn_load_metadata = QPushButton("Load metadata CSV")
        self.btn_load_metadata.clicked.connect(self.load_metadata_csv)

        files_layout.addWidget(QLabel("Processed .npy:"), 0, 0)
        files_layout.addWidget(self.processed_edit, 0, 1)
        files_layout.addWidget(self.btn_load_processed, 0, 2)

        files_layout.addWidget(QLabel("TIFF folder:"), 1, 0)
        files_layout.addWidget(self.tiff_edit, 1, 1)
        files_layout.addWidget(self.btn_load_tiff, 1, 2)

        files_layout.addWidget(QLabel("Metadata CSV:"), 2, 0)
        files_layout.addWidget(self.metadata_edit, 2, 1)
        files_layout.addWidget(self.btn_load_metadata, 2, 2)

        left_layout.addWidget(files_group)

        # Viewer controls
        viewer_group = QGroupBox("Viewer")
        viewer_layout = QGridLayout(viewer_group)

        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems(["Superimposed", "Processed", "Original"])
        self.view_mode_combo.currentIndexChanged.connect(self.update_viewer)

        self.display_scale_combo = QComboBox()
        self.display_scale_combo.addItems([
            "Frame percentiles",
            "Target ROI percentiles",
            "Target minus background",
            "Background-subtracted full frame",
            "Mask outside target",
        ])
        self.display_scale_combo.setCurrentText("Target ROI percentiles")
        self.display_scale_combo.setToolTip(
            "Frame percentiles = escala por todo el frame.\n"
            "Target ROI percentiles = escala usando solo la máscara visual/ROI/target.\n"
            "Target minus background = resta la mediana de fuera de la máscara y escala por el target.\n"
            "Background-subtracted full frame = resta la mediana del fondo y escala por todo el frame.\n"
            "Mask outside target = oculta todo lo que está fuera de la ROI."
        )
        self.display_scale_combo.currentIndexChanged.connect(self.update_viewer)

        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(["coolwarm", "seismic", "turbo", "viridis", "gray"])
        self.cmap_combo.currentIndexChanged.connect(self.update_viewer)

        self.mask_source_combo = QComboBox()
        self.mask_source_combo.addItems([
            "Off",
            "ROI/saved ROI",
            "Auto object from original TIFF",
            "Auto object from processed",
            "ROI OR auto original",
            "ROI AND auto original",
        ])
        self.mask_source_combo.setCurrentText("Auto object from original TIFF")
        self.mask_source_combo.setToolTip(
            "Define qué píxeles se usan como objeto/target para escalar el colormap y/o ocultar fondo.\n"
            "Auto original suele funcionar bien cuando el fondo tiene poca señal y produce valores artificialmente altos."
        )
        self.mask_source_combo.currentIndexChanged.connect(self.update_viewer)

        self.object_keep_combo = QComboBox()
        self.object_keep_combo.addItems(["Keep brighter pixels", "Keep darker pixels"])
        self.object_keep_combo.setToolTip(
            "Keep brighter pixels: excluye fondo oscuro o de baja señal.\n"
            "Keep darker pixels: úsalo si tu tejido/objeto es más oscuro que el fondo."
        )
        self.object_keep_combo.currentIndexChanged.connect(self.update_viewer)

        self.object_threshold_slider = QSlider(Qt.Horizontal)
        self.object_threshold_slider.setRange(0, 100)
        self.object_threshold_slider.setValue(20)
        self.object_threshold_slider.setToolTip(
            "Umbral percentil para la máscara automática.\n"
            "Con Keep brighter: valores > percentil quedan como objeto.\n"
            "Con Keep darker: valores < percentil quedan como objeto."
        )
        self.object_threshold_slider.valueChanged.connect(self.on_object_threshold_slider_changed)
        self.object_threshold_label = QLabel("Object threshold: 20%")

        self.mask_clean_spin = QSpinBox()
        self.mask_clean_spin.setRange(0, 25)
        self.mask_clean_spin.setValue(5)
        self.mask_clean_spin.setToolTip("Limpieza morfológica de la máscara automática en píxeles. 0 = sin limpieza.")
        self.mask_clean_spin.valueChanged.connect(self.update_viewer)

        self.largest_component_checkbox = QCheckBox("Largest object only")
        self.largest_component_checkbox.setChecked(True)
        self.largest_component_checkbox.setToolTip("Evita que manchas pequeñas del fondo entren en la escala visual.")
        self.largest_component_checkbox.stateChanged.connect(self.update_viewer)

        self.low_pct_spin = QDoubleSpinBox()
        self.low_pct_spin.setRange(0.0, 49.9)
        self.low_pct_spin.setDecimals(1)
        self.low_pct_spin.setSingleStep(1.0)
        self.low_pct_spin.setValue(5.0)
        self.low_pct_spin.valueChanged.connect(self.on_low_pct_spin_changed)

        self.low_pct_slider = QSlider(Qt.Horizontal)
        self.low_pct_slider.setRange(0, 49)
        self.low_pct_slider.setValue(5)
        self.low_pct_slider.setToolTip("Percentil bajo para vmin del colormap.")
        self.low_pct_slider.valueChanged.connect(self.on_low_pct_slider_changed)

        self.high_pct_spin = QDoubleSpinBox()
        self.high_pct_spin.setRange(50.1, 100.0)
        self.high_pct_spin.setDecimals(1)
        self.high_pct_spin.setSingleStep(1.0)
        self.high_pct_spin.setValue(95.0)
        self.high_pct_spin.valueChanged.connect(self.on_high_pct_spin_changed)

        self.high_pct_slider = QSlider(Qt.Horizontal)
        self.high_pct_slider.setRange(51, 100)
        self.high_pct_slider.setValue(95)
        self.high_pct_slider.setToolTip("Percentil alto para vmax del colormap.")
        self.high_pct_slider.valueChanged.connect(self.on_high_pct_slider_changed)

        self.overlay_alpha_spin = QDoubleSpinBox()
        self.overlay_alpha_spin.setRange(0.05, 1.0)
        self.overlay_alpha_spin.setDecimals(2)
        self.overlay_alpha_spin.setSingleStep(0.05)
        self.overlay_alpha_spin.setValue(0.70)
        self.overlay_alpha_spin.valueChanged.connect(self.update_viewer)

        self.hide_outside_roi_checkbox = QCheckBox("Hide outside mask/target")
        self.hide_outside_roi_checkbox.setChecked(True)
        self.hide_outside_roi_checkbox.stateChanged.connect(self.update_viewer)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self.update_viewer)

        self.frame_label = QLabel("Frame: 0")

        viewer_layout.addWidget(QLabel("View:"), 0, 0)
        viewer_layout.addWidget(self.view_mode_combo, 0, 1)

        viewer_layout.addWidget(QLabel("Processed scale:"), 1, 0)
        viewer_layout.addWidget(self.display_scale_combo, 1, 1)

        viewer_layout.addWidget(QLabel("Colormap:"), 2, 0)
        viewer_layout.addWidget(self.cmap_combo, 2, 1)

        viewer_layout.addWidget(QLabel("Visual mask source:"), 3, 0)
        viewer_layout.addWidget(self.mask_source_combo, 3, 1)

        viewer_layout.addWidget(QLabel("Auto mask keeps:"), 4, 0)
        viewer_layout.addWidget(self.object_keep_combo, 4, 1)

        viewer_layout.addWidget(self.object_threshold_label, 5, 0)
        viewer_layout.addWidget(self.object_threshold_slider, 5, 1)

        viewer_layout.addWidget(QLabel("Mask clean px:"), 6, 0)
        viewer_layout.addWidget(self.mask_clean_spin, 6, 1)

        viewer_layout.addWidget(self.largest_component_checkbox, 7, 0, 1, 2)

        low_row = QWidget()
        low_layout = QHBoxLayout(low_row)
        low_layout.setContentsMargins(0, 0, 0, 0)
        low_layout.addWidget(self.low_pct_spin)
        low_layout.addWidget(self.low_pct_slider, stretch=1)
        viewer_layout.addWidget(QLabel("Low percentile:"), 8, 0)
        viewer_layout.addWidget(low_row, 8, 1)

        high_row = QWidget()
        high_layout = QHBoxLayout(high_row)
        high_layout.setContentsMargins(0, 0, 0, 0)
        high_layout.addWidget(self.high_pct_spin)
        high_layout.addWidget(self.high_pct_slider, stretch=1)
        viewer_layout.addWidget(QLabel("High percentile:"), 9, 0)
        viewer_layout.addWidget(high_row, 9, 1)

        viewer_layout.addWidget(QLabel("Overlay alpha:"), 10, 0)
        viewer_layout.addWidget(self.overlay_alpha_spin, 10, 1)

        viewer_layout.addWidget(self.hide_outside_roi_checkbox, 11, 0, 1, 2)

        viewer_layout.addWidget(self.frame_label, 12, 0)
        viewer_layout.addWidget(self.frame_slider, 12, 1)

        left_layout.addWidget(viewer_group)

        # ROI controls
        roi_group = QGroupBox("ROI tracking")
        roi_layout = QGridLayout(roi_group)

        self.roi_mode_combo = QComboBox()
        self.roi_mode_combo.addItems(["New ROIs", "Use saved ROIs in session"])

        self.scale_spin = QSpinBox()
        self.scale_spin.setMinimum(1)
        self.scale_spin.setMaximum(12)
        self.scale_spin.setValue(3)

        self.video_name_edit = QLineEdit("tracking_output.avi")

        self.btn_select_track = QPushButton("Select ROIs & Track")
        self.btn_select_track.clicked.connect(self.select_rois_and_track)
        self.btn_select_track.setEnabled(False)

        self.btn_save_rois = QPushButton("Save ROIs JSON")
        self.btn_save_rois.clicked.connect(self.save_rois_json)
        self.btn_save_rois.setEnabled(False)

        self.btn_load_rois = QPushButton("Load ROIs JSON")
        self.btn_load_rois.clicked.connect(self.load_rois_json)

        roi_layout.addWidget(QLabel("ROI mode:"), 0, 0)
        roi_layout.addWidget(self.roi_mode_combo, 0, 1)

        roi_layout.addWidget(QLabel("ROI UI scale:"), 1, 0)
        roi_layout.addWidget(self.scale_spin, 1, 1)

        roi_layout.addWidget(QLabel("Video name:"), 2, 0)
        roi_layout.addWidget(self.video_name_edit, 2, 1)

        roi_layout.addWidget(self.btn_select_track, 3, 0, 1, 2)
        roi_layout.addWidget(self.btn_save_rois, 4, 0)
        roi_layout.addWidget(self.btn_load_rois, 4, 1)

        left_layout.addWidget(roi_group)

        # Progress and logs
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress_label = QLabel("Idle")

        left_layout.addWidget(self.progress)
        left_layout.addWidget(self.progress_label)

        self.logs = QTextEdit()
        self.logs.setReadOnly(True)
        self.logs.setMinimumHeight(140)
        left_layout.addWidget(QLabel("Logs"))
        left_layout.addWidget(self.logs, stretch=1)
        left_layout.addStretch(1)

        # Put the long control column inside a scroll area so it does not get cut
        # on wide/short screens.
        left.setMinimumWidth(390)
        left.setMaximumWidth(540)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left)
        left_scroll.setMinimumWidth(410)
        left_scroll.setMaximumWidth(570)
        splitter.addWidget(left_scroll)

        # Right workspace optimized for a horizontal screen:
        # central image viewer + right results/export panel.
        workspace = QSplitter(Qt.Horizontal)

        viewer_panel = QWidget()
        viewer_layout = QVBoxLayout(viewer_panel)
        viewer_title = QLabel("Viewer")
        viewer_title.setStyleSheet("font-size: 16px; font-weight: bold;")
        viewer_layout.addWidget(viewer_title)
        self.image_canvas = MplCanvas(figsize=(8.5, 6))
        self.image_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        viewer_layout.addWidget(self.image_canvas, stretch=1)

        results_panel = QWidget()
        results_layout = QVBoxLayout(results_panel)
        results_title = QLabel("Results and export")
        results_title.setStyleSheet("font-size: 16px; font-weight: bold;")
        results_layout.addWidget(results_title)

        export_group = QGroupBox("Export outputs")
        export_layout = QGridLayout(export_group)

        self.output_paths_text = QTextEdit()
        self.output_paths_text.setReadOnly(True)
        self.output_paths_text.setMinimumHeight(95)
        self.output_paths_text.setText("No outputs yet. Run tracking first.")

        self.btn_export_data_csv = QPushButton("Export data CSV")
        self.btn_export_data_csv.clicked.connect(self.export_data_csv)
        self.btn_export_data_csv.setEnabled(False)

        self.btn_export_data_xlsx = QPushButton("Export data Excel")
        self.btn_export_data_xlsx.clicked.connect(self.export_data_xlsx)
        self.btn_export_data_xlsx.setEnabled(False)

        self.btn_export_plot_png = QPushButton("Export graph PNG")
        self.btn_export_plot_png.clicked.connect(self.export_plot_png)
        self.btn_export_plot_png.setEnabled(False)

        self.btn_export_plot_pdf = QPushButton("Export graph PDF")
        self.btn_export_plot_pdf.clicked.connect(self.export_plot_pdf)
        self.btn_export_plot_pdf.setEnabled(False)

        self.btn_export_video = QPushButton("Export video AVI")
        self.btn_export_video.clicked.connect(self.export_video_copy)
        self.btn_export_video.setEnabled(False)

        self.btn_open_video = QPushButton("Open video")
        self.btn_open_video.clicked.connect(self.open_video_file)
        self.btn_open_video.setEnabled(False)

        self.btn_export_all = QPushButton("Export all to folder")
        self.btn_export_all.clicked.connect(self.export_all_outputs)
        self.btn_export_all.setEnabled(False)

        self.btn_open_output_folder = QPushButton("Open output folder")
        self.btn_open_output_folder.clicked.connect(self.open_output_folder)
        self.btn_open_output_folder.setEnabled(False)

        export_layout.addWidget(QLabel("Last outputs:"), 0, 0, 1, 4)
        export_layout.addWidget(self.output_paths_text, 1, 0, 1, 4)
        export_layout.addWidget(self.btn_export_data_csv, 2, 0)
        export_layout.addWidget(self.btn_export_data_xlsx, 2, 1)
        export_layout.addWidget(self.btn_export_plot_png, 2, 2)
        export_layout.addWidget(self.btn_export_plot_pdf, 2, 3)
        export_layout.addWidget(self.btn_export_video, 3, 0)
        export_layout.addWidget(self.btn_open_video, 3, 1)
        export_layout.addWidget(self.btn_export_all, 3, 2)
        export_layout.addWidget(self.btn_open_output_folder, 3, 3)

        results_layout.addWidget(export_group)

        self.results_tabs = QTabWidget()

        plot_tab = QWidget()
        plot_layout = QVBoxLayout(plot_tab)
        self.plot_canvas = MplCanvas(figsize=(7, 3.5))
        self.plot_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        plot_layout.addWidget(self.plot_canvas, stretch=1)

        table_tab = QWidget()
        table_layout = QVBoxLayout(table_tab)
        self.table = QTableWidget()
        self.table.setMinimumHeight(250)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        table_layout.addWidget(self.table, stretch=1)

        self.results_tabs.addTab(plot_tab, "Graph")
        self.results_tabs.addTab(table_tab, "Data table")
        results_layout.addWidget(self.results_tabs, stretch=1)

        workspace.addWidget(viewer_panel)
        workspace.addWidget(results_panel)
        workspace.setStretchFactor(0, 3)
        workspace.setStretchFactor(1, 2)

        splitter.addWidget(workspace)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([450, 1450])
        workspace.setSizes([900, 600])

        self.setCentralWidget(root)
        self.draw_empty_viewer()
        self.draw_empty_plot()

    def log(self, text):
        stamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{stamp}] {text}")

    def set_running(self, running):
        self.btn_load_processed.setEnabled(not running)
        self.btn_load_tiff.setEnabled(not running)
        self.btn_load_metadata.setEnabled(not running)
        self.btn_select_track.setEnabled((not running) and self.ready_to_track())
        self.btn_load_rois.setEnabled(not running)
        self.btn_save_rois.setEnabled((not running) and self.saved_rois is not None)
        self.refresh_output_buttons(running=running)

    def ready_to_track(self):
        return self.processed_stack is not None and bool(self.tiff_files)

    def refresh_ready_state(self):
        self.btn_select_track.setEnabled(self.ready_to_track())
        self.btn_save_rois.setEnabled(self.saved_rois is not None)
        self.refresh_output_buttons(running=False)

    def draw_empty_viewer(self):
        self.image_canvas.ax.clear()
        self.image_canvas.ax.text(0.5, 0.5, "Load processed .npy and TIFF folder", ha="center", va="center")
        self.image_canvas.ax.axis("off")
        self.image_canvas.draw()

    def draw_empty_plot(self):
        self.plot_canvas.ax.clear()
        self.plot_canvas.ax.text(0.5, 0.5, "ROI time series will appear here", ha="center", va="center")
        self.plot_canvas.ax.axis("off")
        self.plot_canvas.draw()

    def refresh_output_buttons(self, running=False):
        """Enable export/open buttons only when the corresponding outputs exist."""
        if not hasattr(self, "btn_export_data_csv"):
            return

        has_df = self.result_df is not None and not self.result_df.empty
        has_video = bool(self.video_path) and os.path.exists(self.video_path)
        has_any = has_df or has_video or bool(self.csv_path)
        enabled = not running

        self.btn_export_data_csv.setEnabled(enabled and has_df)
        self.btn_export_data_xlsx.setEnabled(enabled and has_df)
        self.btn_export_plot_png.setEnabled(enabled and has_df)
        self.btn_export_plot_pdf.setEnabled(enabled and has_df)
        self.btn_export_video.setEnabled(enabled and has_video)
        self.btn_open_video.setEnabled(enabled and has_video)
        self.btn_export_all.setEnabled(enabled and has_any)
        self.btn_open_output_folder.setEnabled(enabled and self.current_output_dir() is not None)

    def current_output_dir(self):
        for path in [self.csv_path, self.video_path, self.tracking_json_path, self.saved_rois_path, self.last_export_dir]:
            if path:
                if os.path.isdir(path):
                    return path
                parent = os.path.dirname(path)
                if parent and os.path.isdir(parent):
                    return parent
        return None

    def update_output_paths_text(self):
        if not hasattr(self, "output_paths_text"):
            return

        lines = []
        if self.csv_path:
            lines.append(f"CSV: {self.csv_path}")
        if self.video_path:
            lines.append(f"Video: {self.video_path}")
        if self.tracking_json_path:
            lines.append(f"Tracking JSON: {self.tracking_json_path}")
        if self.saved_rois_path:
            lines.append(f"Saved ROIs JSON: {self.saved_rois_path}")
        if self.last_export_dir:
            lines.append(f"Last export folder: {self.last_export_dir}")

        if not lines:
            lines = ["No outputs yet. Run tracking first."]

        self.output_paths_text.setPlainText("\n".join(lines))
        self.refresh_output_buttons(running=False)

    def suggest_output_name(self, filename):
        folder = self.current_output_dir() or DEFAULT_VIDEO_DIR or DEFAULT_DIR
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, filename)

    def export_data_csv(self):
        if self.result_df is None or self.result_df.empty:
            QMessageBox.information(self, "No data", "No hay datos para exportar.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export ROI data as CSV",
            self.suggest_output_name("roi_mean_values_export.csv"),
            "CSV files (*.csv);;All files (*.*)",
        )
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"

        try:
            self.result_df.to_csv(path, index=False)
            self.csv_path = path
            self.last_export_dir = os.path.dirname(path)
            self.update_output_paths_text()
            self.log(f"Exported data CSV: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export error", str(e))

    def export_data_xlsx(self):
        if self.result_df is None or self.result_df.empty:
            QMessageBox.information(self, "No data", "No hay datos para exportar.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export ROI data as Excel",
            self.suggest_output_name("roi_mean_values_export.xlsx"),
            "Excel files (*.xlsx);;All files (*.*)",
        )
        if not path:
            return
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"

        try:
            self.result_df.to_excel(path, index=False)
            self.last_export_dir = os.path.dirname(path)
            self.update_output_paths_text()
            self.log(f"Exported data Excel: {path}")
        except Exception as e:
            QMessageBox.critical(
                self,
                "Export error",
                f"No se pudo exportar Excel. Instala openpyxl si falta:\n\npython3 -m pip install openpyxl\n\nDetalle:\n{e}",
            )

    def export_plot_png(self):
        self.export_plot_image(fmt="png")

    def export_plot_pdf(self):
        self.export_plot_image(fmt="pdf")

    def export_plot_image(self, fmt="png"):
        if self.result_df is None or self.result_df.empty:
            QMessageBox.information(self, "No graph", "No hay gráfico para exportar.")
            return

        fmt = fmt.lower().strip(".")
        path, _ = QFileDialog.getSaveFileName(
            self,
            f"Export ROI graph as {fmt.upper()}",
            self.suggest_output_name(f"roi_time_series.{fmt}"),
            f"{fmt.upper()} files (*.{fmt});;All files (*.*)",
        )
        if not path:
            return
        if not path.lower().endswith(f".{fmt}"):
            path += f".{fmt}"

        try:
            self.plot_canvas.figure.savefig(path, dpi=250, bbox_inches="tight")
            self.last_export_dir = os.path.dirname(path)
            self.update_output_paths_text()
            self.log(f"Exported graph {fmt.upper()}: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export error", str(e))

    def export_video_copy(self):
        if not self.video_path or not os.path.exists(self.video_path):
            QMessageBox.information(self, "No video", "No hay video generado para exportar.")
            return

        base = os.path.basename(self.video_path) or "tracking_output.avi"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export tracking video",
            self.suggest_output_name(base),
            "AVI files (*.avi);;All files (*.*)",
        )
        if not path:
            return

        try:
            shutil.copy2(self.video_path, path)
            self.video_path = path
            self.last_export_dir = os.path.dirname(path)
            self.update_output_paths_text()
            self.log(f"Exported video copy: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export error", str(e))

    def export_all_outputs(self):
        default_folder = self.suggest_output_name("roi_tracking_export")
        folder = QFileDialog.getExistingDirectory(self, "Select export folder", default_folder)
        if not folder:
            return
        os.makedirs(folder, exist_ok=True)

        exported = []
        try:
            if self.result_df is not None and not self.result_df.empty:
                csv_out = os.path.join(folder, "roi_mean_values_export.csv")
                self.result_df.to_csv(csv_out, index=False)
                exported.append(csv_out)

                try:
                    xlsx_out = os.path.join(folder, "roi_mean_values_export.xlsx")
                    self.result_df.to_excel(xlsx_out, index=False)
                    exported.append(xlsx_out)
                except Exception as e:
                    self.log(f"Excel export skipped: {e}")

                png_out = os.path.join(folder, "roi_time_series.png")
                pdf_out = os.path.join(folder, "roi_time_series.pdf")
                self.plot_canvas.figure.savefig(png_out, dpi=250, bbox_inches="tight")
                self.plot_canvas.figure.savefig(pdf_out, dpi=250, bbox_inches="tight")
                exported.extend([png_out, pdf_out])

            for src, name in [
                (self.video_path, "tracking_output.avi"),
                (self.csv_path, "roi_mean_values_original.csv"),
                (self.tracking_json_path, "roi_tracks.json"),
                (self.saved_rois_path, "saved_rois.json"),
            ]:
                if src and os.path.exists(src):
                    dst = os.path.join(folder, name)
                    if os.path.abspath(src) != os.path.abspath(dst):
                        shutil.copy2(src, dst)
                    exported.append(dst)

            summary_path = os.path.join(folder, "export_summary.txt")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write("ROI tracking export\n")
                f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
                for item in exported:
                    f.write(item + "\n")
            exported.append(summary_path)

            self.last_export_dir = folder
            self.update_output_paths_text()
            self.log(f"Exported all outputs to: {folder}")
            QMessageBox.information(self, "Export complete", f"Exportados {len(exported)} archivos en:\n{folder}")
        except Exception as e:
            QMessageBox.critical(self, "Export error", str(e))

    def open_video_file(self):
        if not self.video_path or not os.path.exists(self.video_path):
            QMessageBox.information(self, "No video", "No hay video generado.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.video_path))

    def open_output_folder(self):
        folder = self.current_output_dir()
        if not folder:
            QMessageBox.information(self, "No folder", "Todavía no hay carpeta de salida.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    def load_processed_stack(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select processed stack .npy",
            DEFAULT_DIR,
            "NumPy files (*.npy);;All files (*.*)",
        )
        if not path:
            return

        try:
            data = np.load(path, allow_pickle=False, mmap_mode="r")
            if data.ndim != 3:
                raise ValueError(f"El processed stack debe tener shape T,H,W. Recibido: {data.shape}")

            self.processed_stack = data
            self.processed_path = path
            self.processed_edit.setText(path)
            self.log(f"Processed stack loaded: {path}")
            self.log(f"Shape={data.shape}, dtype={data.dtype}")

            self.update_frame_slider_limits()
            self.update_viewer()
            self.refresh_ready_state()

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.log(f"ERROR loading processed stack: {e}")

    def load_tiff_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder with TIFF frames", DEFAULT_DIR)
        if not folder:
            return

        files = list_tiff_files(folder)
        if not files:
            QMessageBox.warning(self, "No TIFF files", "No encontré .tif/.tiff en esa carpeta.")
            return

        self.tiff_folder = folder
        self.tiff_files = files
        self.tiff_edit.setText(folder)
        self.log(f"TIFF folder loaded: {folder}")
        self.log(f"TIFF files found: {len(files)}")

        self.update_frame_slider_limits()
        self.update_viewer()
        self.refresh_ready_state()

    def load_metadata_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select metadata CSV",
            DEFAULT_DIR,
            "CSV files (*.csv);;All files (*.*)",
        )
        if not path:
            return

        try:
            md = pd.read_csv(path)
            self.metadata = md
            self.metadata_path = path
            self.metadata_edit.setText(path)
            self.log(f"Metadata loaded: {path}")
            self.log(f"Metadata shape: {md.shape}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.log(f"ERROR loading metadata: {e}")

    def update_frame_slider_limits(self):
        max_frames = 0
        if self.processed_stack is not None:
            max_frames = max(max_frames, int(self.processed_stack.shape[0]))
        if self.tiff_files:
            max_frames = max(max_frames, len(self.tiff_files))

        if max_frames > 0:
            self.frame_slider.setEnabled(True)
            self.frame_slider.setMinimum(0)
            self.frame_slider.setMaximum(max_frames - 1)
            self.frame_slider.setValue(0)
        else:
            self.frame_slider.setEnabled(False)
            self.frame_slider.setMaximum(0)

    def on_low_pct_slider_changed(self, value):
        self.low_pct_spin.blockSignals(True)
        self.low_pct_spin.setValue(float(value))
        self.low_pct_spin.blockSignals(False)
        self.update_viewer()

    def on_high_pct_slider_changed(self, value):
        self.high_pct_spin.blockSignals(True)
        self.high_pct_spin.setValue(float(value))
        self.high_pct_spin.blockSignals(False)
        self.update_viewer()

    def on_low_pct_spin_changed(self, value):
        self.low_pct_slider.blockSignals(True)
        self.low_pct_slider.setValue(int(round(float(value))))
        self.low_pct_slider.blockSignals(False)
        self.update_viewer()

    def on_high_pct_spin_changed(self, value):
        self.high_pct_slider.blockSignals(True)
        self.high_pct_slider.setValue(int(round(float(value))))
        self.high_pct_slider.blockSignals(False)
        self.update_viewer()

    def on_object_threshold_slider_changed(self, value):
        self.object_threshold_label.setText(f"Object threshold: {int(value)}%")
        self.update_viewer()

    def get_current_target_mask(self, frame_idx, shape_hw):
        """
        Target mask used only for visualization.
        Priority:
        1) Tracked ROIs after running tracking.
        2) Saved ROIs JSON/session as fallback.
        """
        mask = mask_from_roi_tracks(shape_hw, self.roi_tracks, frame_idx)
        if mask is not None:
            return mask
        return mask_from_saved_rois(shape_hw, self.saved_rois)

    def get_current_visual_mask(self, frame_idx, shape_hw, img_proc=None, img_tiff=None):
        """
        Mask used for display scaling/overlay.
        This can be ROI-based or automatic from the raw TIFF/processed image.
        It does not change measurements or exported CSV.
        """
        mode = self.mask_source_combo.currentText() if hasattr(self, "mask_source_combo") else "ROI/saved ROI"
        if mode == "Off":
            return None

        roi_mask = self.get_current_target_mask(frame_idx, shape_hw)

        keep_text = self.object_keep_combo.currentText() if hasattr(self, "object_keep_combo") else "Keep brighter pixels"
        keep_mode = "darker" if "darker" in keep_text.lower() else "brighter"
        threshold_pct = int(self.object_threshold_slider.value()) if hasattr(self, "object_threshold_slider") else 20
        clean_px = int(self.mask_clean_spin.value()) if hasattr(self, "mask_clean_spin") else 5
        largest_only = bool(self.largest_component_checkbox.isChecked()) if hasattr(self, "largest_component_checkbox") else True

        auto_original = None
        if img_tiff is not None:
            auto_original = auto_object_mask_from_image(
                img_tiff,
                threshold_pct=threshold_pct,
                keep_mode=keep_mode,
                clean_px=clean_px,
                largest_only=largest_only,
            )
            if auto_original is not None and auto_original.shape != shape_hw:
                auto_original = None

        auto_processed = None
        if img_proc is not None:
            auto_processed = auto_object_mask_from_image(
                img_proc,
                threshold_pct=threshold_pct,
                keep_mode=keep_mode,
                clean_px=clean_px,
                largest_only=largest_only,
            )
            if auto_processed is not None and auto_processed.shape != shape_hw:
                auto_processed = None

        if mode == "ROI/saved ROI":
            return roi_mask
        if mode == "Auto object from original TIFF":
            return auto_original
        if mode == "Auto object from processed":
            return auto_processed
        if mode == "ROI OR auto original":
            return combine_masks(roi_mask, auto_original, mode="or")
        if mode == "ROI AND auto original":
            return combine_masks(roi_mask, auto_original, mode="and")

        return roi_mask

    def prepare_processed_for_display(self, img_proc, frame_idx, img_tiff=None):
        """
        Returns: display_img, vmin, vmax, title_suffix, target_mask.
        This controls visualization only; it does not change exported measurements.
        """
        img = np.asarray(img_proc, dtype=np.float32)
        img = np.nan_to_num(img, nan=np.nan, posinf=np.nan, neginf=np.nan)

        p_low = float(self.low_pct_spin.value())
        p_high = float(self.high_pct_spin.value())
        if p_high <= p_low:
            p_high = min(100.0, p_low + 1.0)

        scale_mode = self.display_scale_combo.currentText()
        target_mask = self.get_current_visual_mask(frame_idx, img.shape, img_proc=img, img_tiff=img_tiff)
        has_target = target_mask is not None and target_mask.any()

        display_img = img.copy()
        title_suffix = scale_mode
        symmetric = False

        if has_target:
            title_suffix += f" | mask={100.0 * float(np.mean(target_mask)):.1f}%"

        if scale_mode == "Frame percentiles":
            vals = display_img[np.isfinite(display_img)]

        elif scale_mode == "Target ROI percentiles":
            if has_target:
                vals = display_img[target_mask]
                title_suffix += " | using tracked/saved ROI"
            else:
                vals = display_img[np.isfinite(display_img)]
                title_suffix += " | no ROI found, fallback frame"

        elif scale_mode == "Target minus background":
            if has_target and np.any(~target_mask):
                background_value = float(np.nanmedian(display_img[~target_mask]))
                display_img = display_img - background_value
                vals = display_img[target_mask]
                symmetric = True
                title_suffix += f" | bg median={background_value:.4g}"
            else:
                background_value = float(np.nanmedian(display_img))
                display_img = display_img - background_value
                vals = display_img[np.isfinite(display_img)]
                symmetric = True
                title_suffix += " | no ROI, global median subtracted"

        elif scale_mode == "Background-subtracted full frame":
            if has_target and np.any(~target_mask):
                background_value = float(np.nanmedian(display_img[~target_mask]))
            else:
                background_value = float(np.nanmedian(display_img))
            display_img = display_img - background_value
            vals = display_img[np.isfinite(display_img)]
            symmetric = True
            title_suffix += f" | bg median={background_value:.4g}"

        elif scale_mode == "Mask outside target":
            if has_target:
                vals = display_img[target_mask]
                display_img = np.ma.masked_where(~target_mask, display_img)
                title_suffix += " | outside hidden"
            else:
                vals = display_img[np.isfinite(display_img)]
                title_suffix += " | no ROI found"
        else:
            vals = display_img[np.isfinite(display_img)]

        # Independent checkbox: useful with Target ROI percentiles or Target minus background.
        if self.hide_outside_roi_checkbox.isChecked() and has_target:
            display_img = np.ma.masked_where(~target_mask, display_img)
            title_suffix += " | outside hidden"

        vmin, vmax = finite_percentile_limits(vals, p_low=p_low, p_high=p_high, symmetric=symmetric)
        return display_img, vmin, vmax, title_suffix, target_mask

    def draw_target_contours(self, target_mask):
        """Draw target ROI outline on matplotlib axes."""
        if target_mask is None or not target_mask.any():
            return
        try:
            self.image_canvas.ax.contour(target_mask.astype(np.uint8), levels=[0.5], colors="yellow", linewidths=1.2)
        except Exception:
            pass

    def update_viewer(self):
        idx = self.frame_slider.value()
        self.frame_label.setText(f"Frame: {idx}")

        if self.processed_stack is None and not self.tiff_files:
            self.draw_empty_viewer()
            return

        mode = self.view_mode_combo.currentText()
        cmap = self.cmap_combo.currentText()
        alpha = float(self.overlay_alpha_spin.value())

        self.image_canvas.figure.clear()
        self.image_canvas.ax = self.image_canvas.figure.add_subplot(111)

        try:
            img_proc = None
            img_tiff = None
            rgb_disp = None

            if self.processed_stack is not None and idx < self.processed_stack.shape[0]:
                img_proc = np.asarray(self.processed_stack[idx], dtype=np.float32)

            if self.tiff_files and idx < len(self.tiff_files):
                img_tiff = tiff.imread(self.tiff_files[idx])

            target_mask = None

            if mode == "Processed":
                if img_proc is None:
                    self.image_canvas.ax.text(0.5, 0.5, "No processed frame", ha="center", va="center")
                else:
                    display_img, vmin, vmax, title_suffix, target_mask = self.prepare_processed_for_display(img_proc, idx, img_tiff=img_tiff)
                    im = self.image_canvas.ax.imshow(display_img, cmap=cmap, vmin=vmin, vmax=vmax)
                    self.draw_target_contours(target_mask)
                    self.image_canvas.ax.set_title(f"Processed | {title_suffix} | vmin={vmin:.4g}, vmax={vmax:.4g}")
                    self.image_canvas.figure.colorbar(im, ax=self.image_canvas.ax, fraction=0.046, pad=0.04)

            elif mode == "Original":
                if img_tiff is None:
                    self.image_canvas.ax.text(0.5, 0.5, "No TIFF frame", ha="center", va="center")
                else:
                    if img_tiff.ndim == 2:
                        vmin = float(np.percentile(img_tiff, 2))
                        vmax = float(np.percentile(img_tiff, 98))
                        if vmax <= vmin:
                            vmax = vmin + 1e-6
                        self.image_canvas.ax.imshow(img_tiff, cmap="gray", vmin=vmin, vmax=vmax)
                    else:
                        rgb = img_tiff[:, :, :3].astype(np.float32)
                        p2, p98 = np.percentile(rgb[np.isfinite(rgb)], (2, 98))
                        rgb_disp = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)
                        self.image_canvas.ax.imshow(rgb_disp)
                    base_shape = ensure_gray(img_tiff).shape
                    target_mask = self.get_current_visual_mask(idx, base_shape, img_proc=img_proc, img_tiff=img_tiff)
                    self.draw_target_contours(target_mask)
                    mask_info = "" if target_mask is None else f" | mask={100.0 * float(np.mean(target_mask)):.1f}%"
                    self.image_canvas.ax.set_title(f"Original TIFF image{mask_info}")

            else:  # Superimposed
                if img_tiff is None or img_proc is None:
                    self.image_canvas.ax.text(0.5, 0.5, "Need processed + TIFF", ha="center", va="center")
                else:
                    if img_tiff.ndim == 2:
                        vmin_t = float(np.percentile(img_tiff, 2))
                        vmax_t = float(np.percentile(img_tiff, 98))
                        if vmax_t <= vmin_t:
                            vmax_t = vmin_t + 1e-6
                        self.image_canvas.ax.imshow(img_tiff, cmap="gray", vmin=vmin_t, vmax=vmax_t)
                    else:
                        rgb = img_tiff[:, :, :3].astype(np.float32)
                        p2, p98 = np.percentile(rgb[np.isfinite(rgb)], (2, 98))
                        rgb_disp = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)
                        self.image_canvas.ax.imshow(rgb_disp)

                    display_img, vmin, vmax, title_suffix, target_mask = self.prepare_processed_for_display(img_proc, idx, img_tiff=img_tiff)
                    im = self.image_canvas.ax.imshow(display_img, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha)
                    self.draw_target_contours(target_mask)
                    self.image_canvas.ax.set_title(f"Overlay | {title_suffix} | vmin={vmin:.4g}, vmax={vmax:.4g}")
                    self.image_canvas.figure.colorbar(im, ax=self.image_canvas.ax, fraction=0.046, pad=0.04)

            self.image_canvas.ax.axis("off")
            self.image_canvas.figure.tight_layout()
            self.image_canvas.draw()

        except Exception as e:
            self.log(f"ERROR viewer: {e}")
            self.image_canvas.ax.text(0.5, 0.5, f"Viewer error:\n{e}", ha="center", va="center")
            self.image_canvas.ax.axis("off")
            self.image_canvas.draw()

    def select_rois_and_track(self):
        if not self.ready_to_track():
            QMessageBox.warning(self, "Missing input", "Carga primero processed .npy y carpeta TIFF.")
            return

        try:
            img0_raw = ensure_gray(tiff.imread(self.tiff_files[0]))
            img0_u8 = to_uint8(img0_raw)
            img0_f32 = img0_u8.astype(np.float32)
            H, W = img0_u8.shape

            mode = self.roi_mode_combo.currentText()
            rois_local = []

            if mode == "Use saved ROIs in session":
                if not self.saved_rois or not self.saved_rois.get("rois"):
                    QMessageBox.information(self, "No saved ROIs", "No hay ROIs guardados. Dibujaré ROIs nuevos.")
                    mode = "New ROIs"
                else:
                    rois_local = rois_from_saved_on_new_img0(self.saved_rois, img0_u8, img0_f32)
                    self.log(f"Saved ROIs relocated: {len(rois_local)}")

            if mode == "New ROIs":
                requested_scale = float(self.scale_spin.value())
                scale = fit_scale_to_screen(W, H, requested=requested_scale, screen_frac=0.85)

                n_select_frames = len(self.tiff_files)
                if self.processed_stack is not None:
                    n_select_frames = min(n_select_frames, int(self.processed_stack.shape[0]))
                n_select_frames = max(1, n_select_frames)

                def large_roi_frame_loader(frame_idx):
                    return to_uint8(ensure_gray(tiff.imread(self.tiff_files[int(frame_idx)])))

                initial_frame = 0
                if self.frame_slider.isEnabled():
                    initial_frame = max(0, min(int(self.frame_slider.value()), n_select_frames - 1))

                dlg = RoiSelectionDialog(
                    img0_u8,
                    title=f"Select LARGE tracking ROIs - navigate frames - scale x{scale:.2f}",
                    scale=scale,
                    name_required=True,
                    parent=self,
                    frame_loader=large_roi_frame_loader,
                    n_frames=n_select_frames,
                    initial_frame=initial_frame,
                )

                if dlg.exec() != QDialog.Accepted:
                    return

                rois_local = dlg.get_rois()

                if not rois_local:
                    QMessageBox.information(self, "No ROIs", "No seleccionaste ROIs.")
                    return

                # Inner ROIs, one dialog per big ROI.
                # The crop is taken from the same reference frame used to draw the large ROI.
                for roi in rois_local:
                    name = roi["name"]
                    x, y, w, h = map(int, roi["rect"])
                    ref_frame = int(roi.get("ref_frame", 0))
                    ref_frame = max(0, min(ref_frame, len(self.tiff_files) - 1))
                    ref_img_u8 = to_uint8(ensure_gray(tiff.imread(self.tiff_files[ref_frame])))
                    crop = ref_img_u8[y:y + h, x:x + w]
                    roi["inners"] = []

                    if crop.size == 0:
                        self.log(f"ROI '{name}' tiene crop vacío en frame {ref_frame}; omito small ROIs.")
                        continue

                    answer = QMessageBox.question(
                        self,
                        "Small ROIs",
                        f"¿Quieres dibujar ROIs pequeños dentro de '{name}'?",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.Yes,
                    )

                    if answer != QMessageBox.Yes:
                        continue

                    target_min_width = 800
                    desired_scale = max(2, min(8, math.ceil(target_min_width / max(1, w))))
                    inner_scale = fit_scale_to_screen(w, h, requested=desired_scale, screen_frac=0.85)

                    inner_dlg = RoiSelectionDialog(
                        crop,
                        title=f"Small ROIs inside '{name}' - reference frame {ref_frame} - scale x{inner_scale:.2f}",
                        scale=inner_scale,
                        name_required=False,
                        parent=self,
                    )

                    if inner_dlg.exec() == QDialog.Accepted:
                        inners = inner_dlg.get_rois()
                        # Rename key format: RoiSelectionDialog returns name/rect already relative to crop
                        roi["inners"] = sanitize_inner_rel(dedup_inner_rects(inners), w, h)

            if not rois_local:
                QMessageBox.information(self, "No ROIs", "No hay ROIs para tracking.")
                return

            # Set output video path
            video_name = self.video_name_edit.text().strip() or "tracking_output.avi"
            if os.path.isabs(video_name) or os.path.dirname(video_name):
                video_path = video_name
            else:
                os.makedirs(DEFAULT_VIDEO_DIR, exist_ok=True)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base, ext = os.path.splitext(video_name)
                if not ext:
                    ext = ".avi"
                video_path = os.path.join(DEFAULT_VIDEO_DIR, f"{base}_{stamp}{ext}")

            self.video_path = video_path
            self.start_tracking_worker(rois_local, video_path)

        except Exception as e:
            QMessageBox.critical(self, "ROI error", str(e))
            self.log(f"ERROR ROI selection: {e}")

    def start_tracking_worker(self, rois_local, video_path):
        self.set_running(True)
        self.progress.setValue(0)
        self.progress_label.setText("Starting tracking...")

        self.thread = QThread(self)
        self.worker = TrackingWorker(
            tiff_files=self.tiff_files,
            processed_stack=self.processed_stack,
            metadata=self.metadata,
            rois_local=rois_local,
            video_path=video_path,
        )

        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_worker_progress)
        self.worker.log.connect(self.log)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.error.connect(self.on_worker_error)

        self.worker.finished.connect(self.thread.quit)
        self.worker.error.connect(self.thread.quit)

        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    @Slot(int, str)
    def on_worker_progress(self, value, message):
        self.progress.setValue(value)
        self.progress_label.setText(message)

    @Slot(dict)
    def on_worker_finished(self, result):
        self.roi_tracks = result.get("roi_tracks")
        self.saved_rois = result.get("saved_payload")
        self.result_df = result.get("df_final")
        self.video_path = result.get("video_path")
        self.csv_path = result.get("csv_path")
        self.tracking_json_path = result.get("tracking_json_path")
        self.saved_rois_path = result.get("saved_rois_path")
        self.last_export_dir = self.current_output_dir()

        self.set_running(False)
        self.refresh_ready_state()
        self.progress.setValue(100)
        self.progress_label.setText("Finished")

        self.log("Tracking done.")
        self.log(f"Video: {self.video_path}")
        self.log(f"CSV: {self.csv_path}")
        self.log(f"Tracks JSON: {self.tracking_json_path}")
        self.log(f"Saved ROIs JSON: {self.saved_rois_path}")

        self.display_result_table(self.result_df)
        self.display_result_plot(self.result_df)
        self.update_output_paths_text()
        self.results_tabs.setCurrentIndex(0)

        QMessageBox.information(
            self,
            "Done",
            "Tracking finished. Outputs are visible in the right Results and export panel.\n\n"
            f"Video:\n{self.video_path}\n\nCSV:\n{self.csv_path}",
        )

        self.thread = None
        self.worker = None

    @Slot(str)
    def on_worker_error(self, error_text):
        self.set_running(False)
        self.progress_label.setText("Error")
        self.log("ERROR during tracking:")
        self.log(error_text)

        QMessageBox.critical(self, "Tracking error", error_text)

        self.thread = None
        self.worker = None

    def display_result_table(self, df):
        self.table.clear()
        self.table.setSortingEnabled(False)

        if df is None or df.empty:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            return

        preview = df.head(1000).copy()
        self.table.setRowCount(len(preview))
        self.table.setColumnCount(len(preview.columns))
        self.table.setHorizontalHeaderLabels([str(c) for c in preview.columns])

        for r in range(len(preview)):
            for c, col in enumerate(preview.columns):
                val = preview.iloc[r, c]
                if isinstance(val, float):
                    text = f"{val:.6g}"
                else:
                    text = str(val)
                self.table.setItem(r, c, QTableWidgetItem(text))

        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.resizeColumnsToContents()
        self.table.setSortingEnabled(True)

    def display_result_plot(self, df):
        self.plot_canvas.ax.clear()

        if df is None or df.empty or "frame" not in df.columns:
            self.draw_empty_plot()
            return

        plot_cols = []
        for c in df.columns:
            if c == "frame" or str(c).lower().startswith("meta_"):
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                plot_cols.append(c)

        if not plot_cols:
            self.plot_canvas.ax.text(0.5, 0.5, "No numeric ROI columns", ha="center", va="center")
            self.plot_canvas.ax.axis("off")
            self.plot_canvas.draw()
            return

        for c in plot_cols:
            self.plot_canvas.ax.plot(df["frame"], df[c], label=str(c), linewidth=1.5)

        self.plot_canvas.ax.set_title("ROI mean intensity over time")
        self.plot_canvas.ax.set_xlabel("Frame")
        self.plot_canvas.ax.set_ylabel("Mean value")
        self.plot_canvas.ax.grid(True, alpha=0.35)
        if len(plot_cols) <= 12:
            self.plot_canvas.ax.legend(loc="best", fontsize=8)
        else:
            self.plot_canvas.ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7)
        self.plot_canvas.figure.tight_layout()
        self.plot_canvas.draw()

    def save_rois_json(self):
        if self.saved_rois is None:
            QMessageBox.information(self, "No ROIs", "No hay ROIs guardados en sesión.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save ROIs JSON",
            os.path.join(DEFAULT_DIR, "saved_rois.json"),
            "JSON files (*.json);;All files (*.*)",
        )
        if not path:
            return

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.saved_rois, f, ensure_ascii=False, indent=2)
            self.log(f"Saved ROIs JSON: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def load_rois_json(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load ROIs JSON",
            DEFAULT_DIR,
            "JSON files (*.json);;All files (*.*)",
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            if "rois" not in payload:
                raise ValueError("Este JSON no tiene clave 'rois'.")

            self.saved_rois = payload
            self.roi_mode_combo.setCurrentText("Use saved ROIs in session")
            self.refresh_ready_state()
            self.log(f"Loaded ROIs JSON: {path}")
            self.log(f"ROIs loaded: {len(payload.get('rois', []))}")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.log(f"ERROR loading ROIs: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RoiTrackingApp()
    window.show()
    sys.exit(app.exec())