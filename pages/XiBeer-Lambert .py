import sys
import os
import json
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime
from math import ceil

import numpy as np
import pandas as pd
from tifffile import imwrite

from PySide6.QtCore import Qt, QObject, QThread, Signal, Slot
from PySide6.QtGui import QColor
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
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QLineEdit,
    QGroupBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QMessageBox,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ============================================================
# Utils: wavelengths, spectra, MBLL, TIFF export
# ============================================================

def load_lambdas_from_json_or_xml(reflectance_path, xml_path):
    """
    Devuelve longitudes de onda en el mismo orden de canales del .npy.

    IMPORTANTE:
    No reordeno las bandas aquí, porque si reordenas lambdas pero no reordenas
    reflectance_stack, rompes la correspondencia canal <-> longitud de onda.
    """

    lambdas = None

    if reflectance_path:
        folder = os.path.dirname(os.path.abspath(reflectance_path))
        stem = os.path.splitext(os.path.basename(reflectance_path))[0]

        candidates = [
            os.path.join(folder, f"{stem}.meta.json"),
            os.path.join(folder, f"{stem}.json"),
        ]

        for json_path in candidates:
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                v = meta.get("virtual_wavelengths_nm")
                if v is not None and len(v) == 16:
                    lambdas = np.array(v, dtype=np.float32)
                    return lambdas, json_path

    if lambdas is None:
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"No encontré XML: {xml_path}")

        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Primero intenta virtual bands de hsi_reflectance
        vbs = root.findall(
            ".//correction_matrix[name='hsi_reflectance']/virtual_bands/virtual_band/wavelength_nm"
        )

        if vbs:
            lambdas = np.array([float(v.text) for v in vbs], dtype=np.float32)
            return lambdas, xml_path

        # Fallback: bandas físicas
        bands = sorted(
            root.findall(".//band"),
            key=lambda b: int(b.get("index", 0))
        )

        if bands:
            lambdas = np.array(
                [float(b.find("peaks/peak/wavelength_nm").text) for b in bands],
                dtype=np.float32
            )
            return lambdas, xml_path

    raise RuntimeError("No pude obtener longitudes de onda desde JSON ni XML.")


def read_spectra_excel(excel_path):
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"No encontré Excel de espectros: {excel_path}")

    df = pd.read_excel(excel_path)

    cols = {c.lower().strip(): c for c in df.columns}

    if "lambda" not in cols:
        raise ValueError("El Excel debe tener una columna llamada 'lambda'.")

    col_lambda = cols["lambda"]
    col_hbo2 = cols.get("hbo2") or cols.get("hb02")
    col_hb = cols.get("hb")

    if col_hbo2 is None:
        raise ValueError("El Excel debe tener una columna 'HbO2' o 'Hb02'.")

    if col_hb is None:
        raise ValueError("El Excel debe tener una columna 'Hb'.")

    return df, col_lambda, col_hbo2, col_hb


def pick_row_nearest(df, col_lambda, wavelength, tol=2):
    m = df[np.isclose(df[col_lambda], wavelength, atol=tol)]

    if not m.empty:
        return m.iloc[0]

    idx = (df[col_lambda] - wavelength).abs().idxmin()
    return df.loc[idx]


def compute_band_diagnostics(lambdas, excel_path, band1, band2):
    if band1 == band2:
        raise ValueError("Selecciona dos bandas diferentes.")

    df_spec, col_lambda, col_hbo2, col_hb = read_spectra_excel(excel_path)

    df_zoom = df_spec[
        (df_spec[col_lambda] >= 450) &
        (df_spec[col_lambda] <= 650)
    ].copy()

    if df_zoom.empty:
        raise ValueError("El Excel no tiene datos entre 450 y 650 nm.")

    lambda1 = float(lambdas[band1])
    lambda2 = float(lambdas[band2])

    row1 = pick_row_nearest(df_zoom, col_lambda, lambda1)
    row2 = pick_row_nearest(df_zoom, col_lambda, lambda2)

    HbO2_l1 = float(row1[col_hbo2])
    Hb_l1 = float(row1[col_hb])
    HbO2_l2 = float(row2[col_hbo2])
    Hb_l2 = float(row2[col_hb])

    E = np.array(
        [
            [HbO2_l1, Hb_l1],
            [HbO2_l2, Hb_l2],
        ],
        dtype=float
    )

    condE = float(np.linalg.cond(E))

    delta1 = HbO2_l1 - Hb_l1
    delta2 = HbO2_l2 - Hb_l2

    scale1 = max(abs(HbO2_l1), abs(Hb_l1), 1e-12)
    scale2 = max(abs(HbO2_l2), abs(Hb_l2), 1e-12)

    rel1 = abs(delta1) / scale1
    rel2 = abs(delta2) / scale2

    signs_opposite = (
        np.sign(delta1) != 0 and
        np.sign(delta2) != 0 and
        np.sign(delta1) != np.sign(delta2)
    )

    score_mean = 0.5 * (rel1 + rel2)
    score_geo = float(np.sqrt(rel1 * rel2))

    return {
        "lambda1": lambda1,
        "lambda2": lambda2,
        "band1": int(band1),
        "band2": int(band2),
        "HbO2_l1": HbO2_l1,
        "Hb_l1": Hb_l1,
        "HbO2_l2": HbO2_l2,
        "Hb_l2": Hb_l2,
        "E": E,
        "condE": condE,
        "delta1": delta1,
        "delta2": delta2,
        "rel1": rel1,
        "rel2": rel2,
        "signs_opposite": signs_opposite,
        "score_mean": score_mean,
        "score_geo": score_geo,
    }


def mbll_2w_delta_pixel(
    R_abs,
    band1,
    band2,
    eps_HbO2_1,
    eps_Hb_1,
    eps_HbO2_2,
    eps_Hb_2,
    baseline_frames=60,
    DPF1=1.0,
    DPF2=1.0,
    L=1.0,
    reflectance_path=None,
    out_root="/home/alonso/Desktop",
    chunk_size=32,
    mask_min_R=1e-4,
    clip_nonneg=True,
    progress_callback=None,
    log_callback=None,
):
    if R_abs is None:
        raise ValueError("R_abs es None.")

    if R_abs.ndim != 4:
        raise ValueError(f"R_abs debe tener shape T,C,H,W. Recibido: {R_abs.shape}")

    T, C, H, W = R_abs.shape

    if band1 < 0 or band1 >= C or band2 < 0 or band2 >= C:
        raise ValueError(f"Bandas fuera de rango. C={C}, band1={band1}, band2={band2}")

    if reflectance_path is not None:
        base_dir = os.path.dirname(os.path.abspath(reflectance_path))
        base_name = os.path.splitext(os.path.basename(reflectance_path))[0]
        run_dir = os.path.join(base_dir, f"{base_name}_MBLL")
    else:
        base_dir = os.path.abspath(out_root)
        run_dir = os.path.join(base_dir, "mbll_run")

    os.makedirs(run_dir, exist_ok=True)

    eps = 1e-12
    n = min(max(1, int(baseline_frames)), T)

    a11 = DPF1 * L * float(eps_HbO2_1)
    a12 = DPF1 * L * float(eps_Hb_1)
    a21 = DPF2 * L * float(eps_HbO2_2)
    a22 = DPF2 * L * float(eps_Hb_2)

    det = a11 * a22 - a12 * a21

    if abs(det) < 1e-12:
        raise ValueError("Matriz E mal condicionada: det≈0. Cambia bandas o DPF.")

    inv11 = a22 / det
    inv12 = -a12 / det
    inv21 = -a21 / det
    inv22 = a11 / det

    if log_callback:
        log_callback(f"MBLL output: {run_dir}")
        log_callback(f"Input shape: T={T}, C={C}, H={H}, W={W}")
        log_callback(f"Baseline frames: {n}")

    R1_base = np.clip(R_abs[:n, band1].astype(np.float32), mask_min_R, 1.0)
    R2_base = np.clip(R_abs[:n, band2].astype(np.float32), mask_min_R, 1.0)

    OD1_0 = -np.log(R1_base + eps).mean(axis=0)
    OD2_0 = -np.log(R2_base + eps).mean(axis=0)

    HbO2_0 = inv11 * OD1_0 + inv12 * OD2_0
    Hb_0 = inv21 * OD1_0 + inv22 * OD2_0

    if clip_nonneg:
        HbO2_0 = np.clip(HbO2_0, 0.0, None)
        Hb_0 = np.clip(Hb_0, 0.0, None)

    tHb_0 = np.clip(HbO2_0 + Hb_0, eps, None)
    good = (tHb_0 > 10 * eps).astype(np.float32)

    dHbO2_path = os.path.join(run_dir, "dHbO2.npy")
    dHb_path = os.path.join(run_dir, "dHb.npy")
    HbO2_path = os.path.join(run_dir, "HbO2.npy")
    Hb_path = os.path.join(run_dir, "Hb.npy")
    StO2_path = os.path.join(run_dir, "StO2.npy")
    StO2mean_path = os.path.join(run_dir, "StO2_mean.npy")

    dHbO2_mm = np.lib.format.open_memmap(
        dHbO2_path, mode="w+", dtype=np.float32, shape=(T, H, W)
    )
    dHb_mm = np.lib.format.open_memmap(
        dHb_path, mode="w+", dtype=np.float32, shape=(T, H, W)
    )
    HbO2_mm = np.lib.format.open_memmap(
        HbO2_path, mode="w+", dtype=np.float32, shape=(T, H, W)
    )
    Hb_mm = np.lib.format.open_memmap(
        Hb_path, mode="w+", dtype=np.float32, shape=(T, H, W)
    )
    StO2_mm = np.lib.format.open_memmap(
        StO2_path, mode="w+", dtype=np.float32, shape=(T, H, W)
    )
    StO2mean = np.lib.format.open_memmap(
        StO2mean_path, mode="w+", dtype=np.float32, shape=(T,)
    )

    chunk_size = max(1, int(chunk_size))
    n_chunks = ceil(T / float(chunk_size))

    for k, t0 in enumerate(range(0, T, chunk_size), start=1):
        t1 = min(t0 + chunk_size, T)

        I1 = np.clip(R_abs[t0:t1, band1].astype(np.float32), mask_min_R, 1.0)
        I2 = np.clip(R_abs[t0:t1, band2].astype(np.float32), mask_min_R, 1.0)

        OD1 = -np.log(I1 + eps)
        OD2 = -np.log(I2 + eps)

        dOD1 = OD1 - OD1_0
        dOD2 = OD2 - OD2_0

        dHbO2_blk = inv11 * dOD1 + inv12 * dOD2
        dHb_blk = inv21 * dOD1 + inv22 * dOD2

        HbO2_blk = HbO2_0 + dHbO2_blk
        Hb_blk = Hb_0 + dHb_blk

        if clip_nonneg:
            HbO2_blk = np.clip(HbO2_blk, 0.0, None)
            Hb_blk = np.clip(Hb_blk, 0.0, None)

        tHb_blk = np.clip(HbO2_blk + Hb_blk, eps, None)
        StO2_blk = np.clip(HbO2_blk / tHb_blk, 0.0, 1.0)

        HbO2_blk *= good
        Hb_blk *= good
        StO2_blk *= good
        dHbO2_blk *= good
        dHb_blk *= good

        dHbO2_mm[t0:t1] = dHbO2_blk
        dHb_mm[t0:t1] = dHb_blk
        HbO2_mm[t0:t1] = HbO2_blk
        Hb_mm[t0:t1] = Hb_blk
        StO2_mm[t0:t1] = StO2_blk
        StO2mean[t0:t1] = StO2_blk.reshape(t1 - t0, -1).mean(axis=1).astype(np.float32)

        dHbO2_mm.flush()
        dHb_mm.flush()
        HbO2_mm.flush()
        Hb_mm.flush()
        StO2_mm.flush()
        StO2mean.flush()

        if progress_callback:
            progress_callback(k / n_chunks, f"MBLL bloques {k}/{n_chunks} frames {t0}:{t1}")

    meta = {
        "reflectance_path": reflectance_path,
        "out_dir": run_dir,
        "shape_input": [int(T), int(C), int(H), int(W)],
        "bands_used": {
            "band1": int(band1),
            "band2": int(band2),
        },
        "epsilons": {
            "eps_HbO2_1": float(eps_HbO2_1),
            "eps_Hb_1": float(eps_Hb_1),
            "eps_HbO2_2": float(eps_HbO2_2),
            "eps_Hb_2": float(eps_Hb_2),
        },
        "DPF": {
            "DPF1": float(DPF1),
            "DPF2": float(DPF2),
        },
        "L": float(L),
        "baseline_frames": int(baseline_frames),
        "chunk_size": int(chunk_size),
        "mask_min_R": float(mask_min_R),
        "clip_nonneg": bool(clip_nonneg),
        "note": "MBLL 2-lambda con ancla basal por pixel; dOD = OD - OD0; x = E^{-1} y.",
    }

    meta_path = os.path.join(run_dir, "mbll_meta.json")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {
        "folder": run_dir,
        "dHbO2_path": dHbO2_path,
        "dHb_path": dHb_path,
        "HbO2_path": HbO2_path,
        "Hb_path": Hb_path,
        "StO2_path": StO2_path,
        "StO2mean_path": StO2mean_path,
        "meta_path": meta_path,
    }


def _auto_vmin_vmax_joint(rgb, p_low=2, p_high=98, ignore_zeros=True):
    a = np.asarray(rgb, dtype=np.float32)
    flat = a.reshape(-1, a.shape[-1])
    flat = flat[np.all(np.isfinite(flat), axis=1)]

    if flat.size == 0:
        return 0.0, 1.0

    vals = flat.reshape(-1)

    if ignore_zeros:
        vals = vals[vals > 0]

    if vals.size == 0:
        return 0.0, 1.0

    vmin = float(np.percentile(vals, p_low))
    vmax = float(np.percentile(vals, p_high))

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return 0.0, 1.0

    return vmin, vmax


def _to_uint16_joint(rgb, vmin=None, vmax=None, gamma=0.8, ignore_zeros=True):
    im = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if vmin is None or vmax is None:
        vmin, vmax = _auto_vmin_vmax_joint(
            im,
            p_low=2,
            p_high=98,
            ignore_zeros=ignore_zeros
        )

    norm = (im - vmin) / (vmax - vmin + 1e-12)
    norm = np.clip(norm, 0.0, 1.0)

    if abs(gamma - 1.0) > 1e-6:
        norm = np.power(norm, 1.0 / gamma)

    return (norm * 65535.0 + 0.5).astype(np.uint16)


def _to_uint16_gray(img, p_low=2, p_high=98, gamma=1.0, ignore_zeros=True):
    im = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    vals = im[np.isfinite(im)]

    if ignore_zeros:
        vals = vals[vals > 0]

    if vals.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.percentile(vals, p_low))
        vmax = float(np.percentile(vals, p_high))

        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin = float(np.min(vals))
            vmax = float(np.max(vals))

        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = 0.0, 1.0

    norm = (im - vmin) / (vmax - vmin + 1e-12)
    norm = np.clip(norm, 0.0, 1.0)

    if abs(gamma - 1.0) > 1e-6:
        norm = np.power(norm, 1.0 / gamma)

    return (norm * 65535.0 + 0.5).astype(np.uint16)


def tiffiles_export(
    reflectance_stack,
    reflectance_path=None,
    out_root="/home/alonso/Desktop",
    progress_callback=None,
    log_callback=None,
):
    if reflectance_stack is None:
        raise ValueError("reflectance_stack es None.")

    if reflectance_stack.ndim != 4:
        raise ValueError(f"Se esperaba shape T,C,H,W. Recibido: {reflectance_stack.shape}")

    T, C, H, W = reflectance_stack.shape

    if C < 3:
        raise ValueError(f"Se requieren al menos 3 canales para RGB. C={C}")

    GRAY_CH = 0
    R_idx, G_idx, B_idx = 15, 10, 0
    GAMMA_GRAY = 0.8
    GAMMA_RGB = 0.8
    IGNORE_ZEROS = True

    if reflectance_path:
        base_dir = os.path.dirname(os.path.abspath(reflectance_path))
    else:
        base_dir = os.path.abspath(out_root)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base_dir, f"TIFFs_{stamp}")
    out_gray = os.path.join(out_dir, "grayscale")
    out_rgb = os.path.join(out_dir, "rgb")

    os.makedirs(out_gray, exist_ok=True)
    os.makedirs(out_rgb, exist_ok=True)

    if log_callback:
        log_callback(f"Exportando TIFFs en: {out_dir}")

    total_steps = T * 2
    done = 0

    for i in range(T):
        img = reflectance_stack[i, GRAY_CH]
        img_u16 = _to_uint16_gray(
            img,
            p_low=2,
            p_high=98,
            gamma=GAMMA_GRAY,
            ignore_zeros=IGNORE_ZEROS,
        )

        imwrite(
            os.path.join(out_gray, f"frame_{i:04d}.tif"),
            img_u16,
            photometric="minisblack",
        )

        done += 1

        if progress_callback:
            progress_callback(done / total_steps, f"TIFF gray {i + 1}/{T}")

    for i in range(T):
        r = reflectance_stack[i, R_idx]
        g = reflectance_stack[i, G_idx]
        b = reflectance_stack[i, B_idx]

        rgb = np.stack([r, g, b], axis=-1)

        rgb_u16 = _to_uint16_joint(
            rgb,
            gamma=GAMMA_RGB,
            ignore_zeros=IGNORE_ZEROS,
        )

        imwrite(
            os.path.join(out_rgb, f"frame_{i:04d}.tif"),
            rgb_u16,
            photometric="rgb",
        )

        done += 1

        if progress_callback:
            progress_callback(done / total_steps, f"TIFF RGB {i + 1}/{T}")

    return out_dir


# ============================================================
# Worker thread
# ============================================================

class CalculationWorker(QObject):
    progress = Signal(int, str)
    log = Signal(str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(
        self,
        reflectance_stack,
        reflectance_path,
        band_params,
        baseline_frames,
        chunk_size,
        DPF1,
        DPF2,
        L,
        mask_min_R,
        clip_nonneg,
        export_tiffs=True,
    ):
        super().__init__()

        self.reflectance_stack = reflectance_stack
        self.reflectance_path = reflectance_path
        self.band_params = band_params
        self.baseline_frames = baseline_frames
        self.chunk_size = chunk_size
        self.DPF1 = DPF1
        self.DPF2 = DPF2
        self.L = L
        self.mask_min_R = mask_min_R
        self.clip_nonneg = clip_nonneg
        self.export_tiffs = export_tiffs

    @Slot()
    def run(self):
        try:
            self.log.emit("Iniciando MBLL...")

            def mbll_progress(frac, msg):
                value = int(frac * 70)
                self.progress.emit(value, msg)

            def tiff_progress(frac, msg):
                value = 70 + int(frac * 30)
                self.progress.emit(value, msg)

            results = mbll_2w_delta_pixel(
                R_abs=self.reflectance_stack,
                band1=self.band_params["band1"],
                band2=self.band_params["band2"],
                eps_HbO2_1=self.band_params["HbO2_l1"],
                eps_Hb_1=self.band_params["Hb_l1"],
                eps_HbO2_2=self.band_params["HbO2_l2"],
                eps_Hb_2=self.band_params["Hb_l2"],
                baseline_frames=self.baseline_frames,
                DPF1=self.DPF1,
                DPF2=self.DPF2,
                L=self.L,
                reflectance_path=self.reflectance_path,
                chunk_size=self.chunk_size,
                mask_min_R=self.mask_min_R,
                clip_nonneg=self.clip_nonneg,
                progress_callback=mbll_progress,
                log_callback=self.log.emit,
            )

            if self.export_tiffs:
                self.log.emit("Iniciando exportación TIFF...")
                tiff_dir = tiffiles_export(
                    reflectance_stack=self.reflectance_stack,
                    reflectance_path=self.reflectance_path,
                    progress_callback=tiff_progress,
                    log_callback=self.log.emit,
                )
                results["tiff_dir"] = tiff_dir

            self.progress.emit(100, "Listo")
            self.finished.emit(results)

        except Exception:
            self.error.emit(traceback.format_exc())


# ============================================================
# Matplotlib canvas
# ============================================================

class ImageCanvas(FigureCanvas):
    def __init__(self):
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        super().__init__(self.figure)


# ============================================================
# Main PySide6 app
# ============================================================

class SpectralAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Spectral analysis of Ximea cameras - PySide6")
        self.resize(1350, 850)

        self.default_dir = "/home/alonso/Desktop/"

        self.reflectance_stack = None
        self.reflectance_path = None
        self.lambdas = None
        self.lambdas_source = None
        self.band_params = None
        self.selected_plot_band = 0

        self.thread = None
        self.worker = None

        self.xml_default = "ximea files/CMV2K-SSM4x4-460_600-15.7.20.6.xml"
        self.excel_default = "ximea files/HbO2_Hb_spectrum_full.xlsx"

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
        main_layout.addWidget(splitter)

        # ---------------- LEFT PANEL ----------------
        left = QWidget()
        left_layout = QVBoxLayout(left)

        file_group = QGroupBox("Files")
        file_layout = QGridLayout(file_group)

        self.reflectance_edit = QLineEdit()
        self.reflectance_edit.setReadOnly(True)

        self.btn_load_reflectance = QPushButton("Load reflectance .npy")
        self.btn_load_reflectance.clicked.connect(self.load_reflectance_file)

        self.xml_edit = QLineEdit(self.xml_default)
        self.btn_xml = QPushButton("XML...")
        self.btn_xml.clicked.connect(self.browse_xml)

        self.excel_edit = QLineEdit(self.excel_default)
        self.btn_excel = QPushButton("Spectra Excel...")
        self.btn_excel.clicked.connect(self.browse_excel)

        self.data_info = QLabel("No reflectance stack loaded.")
        self.data_info.setWordWrap(True)

        file_layout.addWidget(QLabel("Reflectance:"), 0, 0)
        file_layout.addWidget(self.reflectance_edit, 0, 1)
        file_layout.addWidget(self.btn_load_reflectance, 0, 2)

        file_layout.addWidget(QLabel("Ximea XML:"), 1, 0)
        file_layout.addWidget(self.xml_edit, 1, 1)
        file_layout.addWidget(self.btn_xml, 1, 2)

        file_layout.addWidget(QLabel("Hb spectra:"), 2, 0)
        file_layout.addWidget(self.excel_edit, 2, 1)
        file_layout.addWidget(self.btn_excel, 2, 2)

        file_layout.addWidget(self.data_info, 3, 0, 1, 3)

        left_layout.addWidget(file_group)

        # ---------------- VISUALIZATION ----------------
        vis_group = QGroupBox("Reflectance visualization")
        vis_layout = QGridLayout(vis_group)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self.update_visualization)

        self.vis_band_slider = QSlider(Qt.Horizontal)
        self.vis_band_slider.setMinimum(0)
        self.vis_band_slider.setMaximum(15)
        self.vis_band_slider.setEnabled(False)
        self.vis_band_slider.valueChanged.connect(self.update_visualization)

        self.frame_label = QLabel("Frame: 0")
        self.vis_band_label = QLabel("Band: 0")

        vis_layout.addWidget(self.frame_label, 0, 0)
        vis_layout.addWidget(self.frame_slider, 0, 1)

        vis_layout.addWidget(self.vis_band_label, 1, 0)
        vis_layout.addWidget(self.vis_band_slider, 1, 1)

        left_layout.addWidget(vis_group)

        # ---------------- BAND SELECTION ----------------
        band_group = QGroupBox("Band selection for MBLL")
        band_layout = QGridLayout(band_group)

        self.band1_slider = QSlider(Qt.Horizontal)
        self.band1_slider.setMinimum(0)
        self.band1_slider.setMaximum(15)
        self.band1_slider.setValue(13)
        self.band1_slider.valueChanged.connect(self.update_band_selection)

        self.band2_slider = QSlider(Qt.Horizontal)
        self.band2_slider.setMinimum(0)
        self.band2_slider.setMaximum(15)
        self.band2_slider.setValue(11)
        self.band2_slider.valueChanged.connect(self.update_band_selection)

        self.band1_label = QLabel("Band HbO₂: 13")
        self.band2_label = QLabel("Band Hb: 11")

        self.btn_refresh_bands = QPushButton("Reload wavelengths / spectra")
        self.btn_refresh_bands.clicked.connect(self.update_band_selection)

        self.diagnostic_text = QTextEdit()
        self.diagnostic_text.setReadOnly(True)
        self.diagnostic_text.setMaximumHeight(140)

        self.lambda_table = QTableWidget(16, 2)
        self.lambda_table.setHorizontalHeaderLabels(["Channel", "Wavelength nm"])
        self.lambda_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.lambda_table.setMaximumHeight(220)

        band_layout.addWidget(self.band1_label, 0, 0)
        band_layout.addWidget(self.band1_slider, 0, 1)

        band_layout.addWidget(self.band2_label, 1, 0)
        band_layout.addWidget(self.band2_slider, 1, 1)

        band_layout.addWidget(self.btn_refresh_bands, 2, 0, 1, 2)
        band_layout.addWidget(self.lambda_table, 3, 0, 1, 2)
        band_layout.addWidget(self.diagnostic_text, 4, 0, 1, 2)

        left_layout.addWidget(band_group)

        # ---------------- PARAMETERS ----------------
        params_group = QGroupBox("MBLL parameters")
        params_layout = QGridLayout(params_group)

        self.baseline_spin = QSpinBox()
        self.baseline_spin.setMinimum(1)
        self.baseline_spin.setMaximum(100000)
        self.baseline_spin.setValue(60)

        self.chunk_spin = QSpinBox()
        self.chunk_spin.setMinimum(1)
        self.chunk_spin.setMaximum(10000)
        self.chunk_spin.setValue(32)

        self.dpf1_spin = QDoubleSpinBox()
        self.dpf1_spin.setDecimals(4)
        self.dpf1_spin.setMinimum(0.0001)
        self.dpf1_spin.setMaximum(1000)
        self.dpf1_spin.setValue(1.0)

        self.dpf2_spin = QDoubleSpinBox()
        self.dpf2_spin.setDecimals(4)
        self.dpf2_spin.setMinimum(0.0001)
        self.dpf2_spin.setMaximum(1000)
        self.dpf2_spin.setValue(1.0)

        self.L_spin = QDoubleSpinBox()
        self.L_spin.setDecimals(4)
        self.L_spin.setMinimum(0.0001)
        self.L_spin.setMaximum(1000)
        self.L_spin.setValue(1.0)

        self.mask_spin = QDoubleSpinBox()
        self.mask_spin.setDecimals(8)
        self.mask_spin.setMinimum(0.0)
        self.mask_spin.setMaximum(1.0)
        self.mask_spin.setSingleStep(0.0001)
        self.mask_spin.setValue(1e-4)

        self.clip_checkbox = QCheckBox("Clip Hb/HbO₂ non-negative")
        self.clip_checkbox.setChecked(True)

        self.export_tiff_checkbox = QCheckBox("Export TIFFs after MBLL")
        self.export_tiff_checkbox.setChecked(True)

        params_layout.addWidget(QLabel("Baseline frames:"), 0, 0)
        params_layout.addWidget(self.baseline_spin, 0, 1)

        params_layout.addWidget(QLabel("Chunk size:"), 1, 0)
        params_layout.addWidget(self.chunk_spin, 1, 1)

        params_layout.addWidget(QLabel("DPF1:"), 2, 0)
        params_layout.addWidget(self.dpf1_spin, 2, 1)

        params_layout.addWidget(QLabel("DPF2:"), 3, 0)
        params_layout.addWidget(self.dpf2_spin, 3, 1)

        params_layout.addWidget(QLabel("L:"), 4, 0)
        params_layout.addWidget(self.L_spin, 4, 1)

        params_layout.addWidget(QLabel("mask_min_R:"), 5, 0)
        params_layout.addWidget(self.mask_spin, 5, 1)

        params_layout.addWidget(self.clip_checkbox, 6, 0, 1, 2)
        params_layout.addWidget(self.export_tiff_checkbox, 7, 0, 1, 2)

        left_layout.addWidget(params_group)

        # ---------------- RUN ----------------
        self.btn_run = QPushButton("Run MBLL")
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self.run_calculations)

        self.progress = QProgressBar()
        self.progress.setValue(0)

        self.progress_label = QLabel("Idle")

        left_layout.addWidget(self.btn_run)
        left_layout.addWidget(self.progress)
        left_layout.addWidget(self.progress_label)

        left_layout.addStretch()

        splitter.addWidget(left)

        # ---------------- RIGHT PANEL ----------------
        right = QWidget()
        right_layout = QVBoxLayout(right)

        self.canvas = ImageCanvas()
        right_layout.addWidget(self.canvas, stretch=4)

        spectrum_group = QGroupBox("Wavelength / hemoglobin reference")
        spectrum_layout = QVBoxLayout(spectrum_group)

        self.spectrum_canvas = ImageCanvas()
        self.spectrum_canvas.setMinimumHeight(300)

        self.spectrum_info = QLabel(
            "Carga el .npy/XML y el Excel de espectros. El gráfico usa las longitudes de onda reales "
            "del código y las curvas HbO₂/Hb del Excel."
        )
        self.spectrum_info.setWordWrap(True)

        spectrum_layout.addWidget(self.spectrum_canvas, stretch=1)
        spectrum_layout.addWidget(self.spectrum_info)

        right_layout.addWidget(spectrum_group, stretch=3)

        self.spectrum_canvas.mpl_connect("button_press_event", self.on_spectrum_plot_clicked)

        logs_group = QGroupBox("Logs")
        logs_layout = QVBoxLayout(logs_group)

        self.logs = QTextEdit()
        self.logs.setReadOnly(True)

        logs_layout.addWidget(self.logs)
        right_layout.addWidget(logs_group, stretch=2)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(root)

        self.update_empty_canvas()

    def log(self, text):
        stamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{stamp}] {text}")

    def update_empty_canvas(self):
        self.canvas.ax.clear()
        self.canvas.ax.text(
            0.5,
            0.5,
            "Load reflectance .npy",
            ha="center",
            va="center",
            fontsize=14,
        )
        self.canvas.ax.axis("off")
        self.canvas.draw()
        self.update_empty_spectrum_canvas()

    def update_empty_spectrum_canvas(self):
        self.spectrum_canvas.ax.clear()
        self.spectrum_canvas.ax.text(
            0.5,
            0.5,
            "No wavelength / Hb spectrum loaded yet",
            ha="center",
            va="center",
            fontsize=12,
        )
        self.spectrum_canvas.ax.axis("off")
        self.spectrum_canvas.draw()

    def browse_xml(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Ximea XML",
            self.default_dir,
            "XML files (*.xml);;All files (*.*)"
        )

        if path:
            self.xml_edit.setText(path)
            self.update_band_selection()

    def browse_excel(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Hb spectra Excel",
            self.default_dir,
            "Excel files (*.xlsx *.xls);;All files (*.*)"
        )

        if path:
            self.excel_edit.setText(path)
            self.update_band_selection()

    def load_reflectance_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select reflectance .npy file",
            self.default_dir,
            "NumPy files (*.npy);;All files (*.*)"
        )

        if not path:
            return

        try:
            data = np.load(path, allow_pickle=False, mmap_mode="r")

            if data.ndim != 4:
                raise ValueError(f"El .npy debe tener shape T,C,H,W. Recibido: {data.shape}")

            T, C, H, W = data.shape

            if C != 16:
                self.log(f"Advertencia: C={C}. Esperaba 16 bandas.")

            self.reflectance_stack = data
            self.reflectance_path = path
            self.reflectance_edit.setText(path)

            self.data_info.setText(
                f"Loaded reflectance stack\n"
                f"Shape: {data.shape}\n"
                f"dtype: {data.dtype}"
            )

            self.log(f"Reflectance loaded: {path}")
            self.log(f"Shape: {data.shape}, dtype: {data.dtype}")

            self.frame_slider.setEnabled(True)
            self.frame_slider.setMinimum(0)
            self.frame_slider.setMaximum(T - 1)
            self.frame_slider.setValue(0)

            self.vis_band_slider.setEnabled(True)
            self.vis_band_slider.setMinimum(0)
            self.vis_band_slider.setMaximum(C - 1)
            self.vis_band_slider.setValue(0)

            self.band1_slider.setMaximum(C - 1)
            self.band2_slider.setMaximum(C - 1)

            if self.band1_slider.value() >= C:
                self.band1_slider.setValue(C - 1)

            if self.band2_slider.value() >= C:
                self.band2_slider.setValue(max(0, C - 2))

            self.update_visualization()
            self.update_band_selection()

        except Exception as e:
            QMessageBox.critical(self, "Error loading .npy", str(e))
            self.log(f"ERROR loading .npy: {e}")

    def update_visualization(self):
        if self.reflectance_stack is None:
            self.update_empty_canvas()
            return

        try:
            frame_idx = self.frame_slider.value()
            band_idx = self.vis_band_slider.value()

            self.frame_label.setText(f"Frame: {frame_idx}")
            self.vis_band_label.setText(f"Band: {band_idx}")

            image = self.reflectance_stack[frame_idx, band_idx]
            im_clean = np.nan_to_num(
                image,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            vals = im_clean[np.isfinite(im_clean)]

            if vals.size == 0:
                vmin, vmax = 0.0, 1.0
            else:
                vmin = float(np.percentile(vals, 2))
                vmax = float(np.percentile(vals, 98))

                if vmax <= vmin:
                    vmin = float(np.min(vals))
                    vmax = float(np.max(vals))

                if vmax <= vmin:
                    vmax = vmin + 1e-6

            self.canvas.ax.clear()
            self.canvas.ax.imshow(im_clean, cmap="gray", vmin=vmin, vmax=vmax)
            self.canvas.ax.set_title(f"Frame {frame_idx} | Band {band_idx}")
            self.canvas.ax.axis("off")
            self.canvas.draw()
            self.update_spectrum_plot()

        except Exception as e:
            self.log(f"ERROR visualization: {e}")

    def update_lambda_table(self):
        self.lambda_table.clearContents()

        if self.lambdas is None:
            for i in range(16):
                self.lambda_table.setItem(i, 0, QTableWidgetItem(str(i)))
                self.lambda_table.setItem(i, 1, QTableWidgetItem("NA"))
            return

        for i in range(len(self.lambdas)):
            self.lambda_table.setItem(i, 0, QTableWidgetItem(str(i)))
            self.lambda_table.setItem(i, 1, QTableWidgetItem(f"{float(self.lambdas[i]):.2f}"))

    def _get_current_channel_indices(self):
        """Return current channel indices safely clipped to the available wavelength array."""
        n = len(self.lambdas) if self.lambdas is not None else 16

        view_band = self.vis_band_slider.value() if self.vis_band_slider.isEnabled() else self.selected_plot_band
        band1 = self.band1_slider.value()
        band2 = self.band2_slider.value()

        view_band = int(np.clip(view_band, 0, max(0, n - 1)))
        band1 = int(np.clip(band1, 0, max(0, n - 1)))
        band2 = int(np.clip(band2, 0, max(0, n - 1)))

        return view_band, band1, band2

    def on_spectrum_plot_clicked(self, event):
        """
        Click sobre el gráfico: selecciona el canal de cámara cuya longitud de onda
        está más cerca del punto donde hiciste click.
        """
        if self.lambdas is None or event.xdata is None:
            return

        try:
            lambdas = np.asarray(self.lambdas, dtype=float)
            idx = int(np.argmin(np.abs(lambdas - float(event.xdata))))
            self.selected_plot_band = idx

            if self.vis_band_slider.isEnabled():
                self.vis_band_slider.setValue(idx)
            else:
                self.update_spectrum_plot()

        except Exception as e:
            self.log(f"ERROR spectrum click: {e}")

    def update_lambda_table_selection(self):
        """Highlight selected channels in the wavelength table."""
        if self.lambdas is None:
            return

        view_band, band1, band2 = self._get_current_channel_indices()

        for row in range(self.lambda_table.rowCount()):
            for col in range(self.lambda_table.columnCount()):
                item = self.lambda_table.item(row, col)
                if item is None:
                    continue

                if row == band1:
                    item.setBackground(QColor("#ffd6d6"))  # MBLL HbO2 band
                elif row == band2:
                    item.setBackground(QColor("#d6f5d6"))  # MBLL Hb band
                elif row == view_band:
                    item.setBackground(QColor("#fff1b8"))  # displayed reflectance band
                else:
                    item.setBackground(QColor("white"))

    def update_spectrum_plot(self):
        """
        Plot HbO2/Hb spectra and overlay camera channel wavelengths.

        Uses:
        - self.lambdas loaded by load_lambdas_from_json_or_xml()
        - Hb/HbO2 spectra loaded by read_spectra_excel()
        """
        ax = self.spectrum_canvas.ax
        ax.clear()

        if self.lambdas is None or len(self.lambdas) == 0:
            ax.text(0.5, 0.5, "No wavelength data available", ha="center", va="center")
            ax.axis("off")
            self.spectrum_canvas.draw()
            self.spectrum_info.setText(
                "No hay longitudes de onda cargadas todavía. Carga el reflectance .npy/XML."
            )
            return

        lambdas = np.asarray(self.lambdas, dtype=float)
        channels = np.arange(len(lambdas))

        view_band, band1, band2 = self._get_current_channel_indices()
        view_lambda = float(lambdas[view_band])
        lambda1 = float(lambdas[band1])
        lambda2 = float(lambdas[band2])

        excel_path = self.excel_edit.text().strip()

        try:
            df_spec, col_lambda, col_hbo2, col_hb = read_spectra_excel(excel_path)

            # Limit strictly to the spectral range covered by the camera channels.
            # Nothing outside min(channel wavelengths) and max(channel wavelengths) is drawn.
            x_min = float(np.nanmin(lambdas))
            x_max = float(np.nanmax(lambdas))

            df_zoom = df_spec[
                (df_spec[col_lambda] >= x_min) &
                (df_spec[col_lambda] <= x_max)
            ].copy()

            if df_zoom.empty:
                raise ValueError(
                    f"El Excel no contiene datos dentro del rango de canales "
                    f"({x_min:.2f}-{x_max:.2f} nm)."
                )

            x = df_zoom[col_lambda].astype(float).to_numpy()
            y_hbo2 = df_zoom[col_hbo2].astype(float).to_numpy()
            y_hb = df_zoom[col_hb].astype(float).to_numpy()

            ax.plot(x, y_hbo2, linewidth=2.0, label="HbO₂ spectrum", color="red")
            ax.plot(x, y_hb, linewidth=2.0, label="Hb spectrum", color="green")

            y_all = np.concatenate([
                y_hbo2[np.isfinite(y_hbo2)],
                y_hb[np.isfinite(y_hb)],
            ])

            if y_all.size:
                y_min = float(np.nanmin(y_all))
                y_max = float(np.nanmax(y_all))
            else:
                y_min, y_max = 0.0, 1.0

            if y_max <= y_min:
                y_max = y_min + 1.0

            y_range = y_max - y_min
            marker_y = y_min - 0.08 * y_range
            text_y = y_min - 0.16 * y_range

            # Camera channels as spectral positions.
            for ch, lam in zip(channels, lambdas):
                ax.axvline(lam, color="0.75", linewidth=0.8, alpha=0.6, zorder=0)
                ax.scatter([lam], [marker_y], s=28, color="black", zorder=3)
                ax.text(
                    lam,
                    text_y,
                    str(int(ch)),
                    ha="center",
                    va="top",
                    fontsize=8,
                    rotation=90,
                )

            def mark_selected(lam, label, color, ypos_factor):
                ax.axvline(lam, color=color, linestyle="--", linewidth=1.8, alpha=0.95)
                ax.scatter([lam], [marker_y], s=100, color=color, edgecolors="black", linewidths=0.7, zorder=4)
                ax.text(
                    lam,
                    y_max + ypos_factor * y_range,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=color,
                    fontweight="bold",
                )

            mark_selected(
                view_lambda,
                f"View ch {view_band}\n{view_lambda:.2f} nm",
                "orange",
                0.04,
            )
            mark_selected(
                lambda1,
                f"MBLL 1 ch {band1}\n{lambda1:.2f} nm",
                "red",
                0.18,
            )
            mark_selected(
                lambda2,
                f"MBLL 2 ch {band2}\n{lambda2:.2f} nm",
                "green",
                0.32,
            )

            # Values at nearest spectral rows for selected bands.
            row_view = pick_row_nearest(df_spec, col_lambda, view_lambda)
            row1 = pick_row_nearest(df_spec, col_lambda, lambda1)
            row2 = pick_row_nearest(df_spec, col_lambda, lambda2)

            view_hbo2 = float(row_view[col_hbo2])
            view_hb = float(row_view[col_hb])
            hbo2_1 = float(row1[col_hbo2])
            hb_1 = float(row1[col_hb])
            hbo2_2 = float(row2[col_hbo2])
            hb_2 = float(row2[col_hb])

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(marker_y - 0.20 * y_range, y_max + 0.50 * y_range)
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Extinction / absorption coefficient")
            ax.set_title("HbO₂ / Hb spectra limited to camera-channel wavelength range")
            ax.grid(True, linestyle=":", alpha=0.35)
            ax.legend(loc="upper right", fontsize=8)

            delta12 = abs(lambda1 - lambda2)
            delta_v1 = abs(view_lambda - lambda1)
            delta_v2 = abs(view_lambda - lambda2)

            self.spectrum_info.setText(
                f"View channel {view_band}: {view_lambda:.2f} nm "
                f"(HbO₂={view_hbo2:.4g}, Hb={view_hb:.4g}) | "
                f"MBLL 1 channel {band1}: {lambda1:.2f} nm "
                f"(HbO₂={hbo2_1:.4g}, Hb={hb_1:.4g}) | "
                f"MBLL 2 channel {band2}: {lambda2:.2f} nm "
                f"(HbO₂={hbo2_2:.4g}, Hb={hb_2:.4g}) | "
                f"Δ MBLL1-MBLL2={delta12:.2f} nm | "
                f"Δ view-MBLL1={delta_v1:.2f} nm | "
                f"Δ view-MBLL2={delta_v2:.2f} nm"
            )

        except Exception as e:
            # If Excel is missing, still show the channel wavelength map.
            x_min = float(np.nanmin(lambdas))
            x_max = float(np.nanmax(lambdas))

            ax.scatter(lambdas, np.zeros_like(lambdas), s=40, color="black")
            ax.plot(lambdas, np.zeros_like(lambdas), color="0.5", linewidth=1.0)

            for ch, lam in zip(channels, lambdas):
                ax.text(lam, 0.04, str(int(ch)), ha="center", va="bottom", fontsize=8)

            ax.axvline(view_lambda, color="orange", linestyle="--", linewidth=1.8)
            ax.axvline(lambda1, color="red", linestyle="--", linewidth=1.8)
            ax.axvline(lambda2, color="green", linestyle="--", linewidth=1.8)

            ax.scatter([view_lambda], [0], s=100, color="orange", edgecolors="black")
            ax.scatter([lambda1], [0], s=100, color="red", edgecolors="black")
            ax.scatter([lambda2], [0], s=100, color="green", edgecolors="black")

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(-0.15, 0.35)
            ax.set_yticks([])
            ax.set_xlabel("Wavelength (nm)")
            ax.set_title("Camera channel wavelengths only")
            ax.grid(axis="x", linestyle=":", alpha=0.35)

            self.spectrum_info.setText(
                f"Se cargaron las longitudes de onda, pero no pude graficar Hb/HbO₂ desde el Excel: {e}. "
                f"View ch {view_band}: {view_lambda:.2f} nm | "
                f"MBLL1 ch {band1}: {lambda1:.2f} nm | "
                f"MBLL2 ch {band2}: {lambda2:.2f} nm"
            )

        self.spectrum_canvas.figure.tight_layout()
        self.spectrum_canvas.draw()
        self.update_lambda_table_selection()

    def update_band_selection(self):
        try:
            band1 = self.band1_slider.value()
            band2 = self.band2_slider.value()

            self.band1_label.setText(f"Band HbO₂: {band1}")
            self.band2_label.setText(f"Band Hb: {band2}")

            xml_path = self.xml_edit.text().strip()
            excel_path = self.excel_edit.text().strip()

            self.lambdas, self.lambdas_source = load_lambdas_from_json_or_xml(
                self.reflectance_path,
                xml_path,
            )

            if len(self.lambdas) < 16:
                raise ValueError(f"Se esperaban 16 longitudes de onda. Recibido: {len(self.lambdas)}")

            self.update_lambda_table()
            self.update_spectrum_plot()

            params = compute_band_diagnostics(
                lambdas=self.lambdas,
                excel_path=excel_path,
                band1=band1,
                band2=band2,
            )

            self.band_params = params

            E = params["E"]

            lines = []
            lines.append(f"Wavelength source: {self.lambdas_source}")
            lines.append("")
            lines.append(
                f"Band1={band1} → λ1={params['lambda1']:.2f} nm | "
                f"HbO₂={params['HbO2_l1']:.4g}, Hb={params['Hb_l1']:.4g}"
            )
            lines.append(
                f"Band2={band2} → λ2={params['lambda2']:.2f} nm | "
                f"HbO₂={params['HbO2_l2']:.4g}, Hb={params['Hb_l2']:.4g}"
            )
            lines.append("")
            lines.append("E matrix:")
            lines.append(f"[[{E[0,0]:.4g}, {E[0,1]:.4g}],")
            lines.append(f" [{E[1,0]:.4g}, {E[1,1]:.4g}]]")
            lines.append("")
            lines.append(f"cond(E): {params['condE']:.2f}")
            lines.append(f"Δ1 HbO₂-Hb: {params['delta1']:.4g}")
            lines.append(f"Δ2 HbO₂-Hb: {params['delta2']:.4g}")
            lines.append(f"Relative separation 1: {params['rel1']:.3f}")
            lines.append(f"Relative separation 2: {params['rel2']:.3f}")
            lines.append(f"Opposite signs: {'yes' if params['signs_opposite'] else 'no'}")
            lines.append(f"Score mean: {params['score_mean']:.3f}")
            lines.append(f"Score geometric: {params['score_geo']:.3f}")

            if params["condE"] > 1000:
                lines.append("")
                lines.append("WARNING: high numerical instability.")
            elif params["condE"] > 100:
                lines.append("")
                lines.append("Moderate numerical condition.")
            else:
                lines.append("")
                lines.append("Good numerical condition.")

            self.diagnostic_text.setText("\n".join(lines))

            if self.reflectance_stack is not None:
                self.btn_run.setEnabled(True)

        except Exception as e:
            self.band_params = None
            self.btn_run.setEnabled(False)
            self.diagnostic_text.setText(f"Band diagnostics unavailable:\n{str(e)}")
            self.update_lambda_table()
            self.update_spectrum_plot()

    def set_ui_running(self, running):
        self.btn_load_reflectance.setEnabled(not running)
        self.btn_xml.setEnabled(not running)
        self.btn_excel.setEnabled(not running)
        self.btn_refresh_bands.setEnabled(not running)
        self.btn_run.setEnabled(not running)
        self.band1_slider.setEnabled(not running)
        self.band2_slider.setEnabled(not running)

    def run_calculations(self):
        if self.reflectance_stack is None:
            QMessageBox.warning(self, "Missing data", "Load reflectance stack first.")
            return

        self.update_band_selection()

        if self.band_params is None:
            QMessageBox.warning(self, "Missing band parameters", "Band selection is invalid.")
            return

        self.progress.setValue(0)
        self.progress_label.setText("Starting...")
        self.set_ui_running(True)

        self.thread = QThread(self)

        self.worker = CalculationWorker(
            reflectance_stack=self.reflectance_stack,
            reflectance_path=self.reflectance_path,
            band_params=self.band_params,
            baseline_frames=self.baseline_spin.value(),
            chunk_size=self.chunk_spin.value(),
            DPF1=self.dpf1_spin.value(),
            DPF2=self.dpf2_spin.value(),
            L=self.L_spin.value(),
            mask_min_R=self.mask_spin.value(),
            clip_nonneg=self.clip_checkbox.isChecked(),
            export_tiffs=self.export_tiff_checkbox.isChecked(),
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
    def on_worker_finished(self, results):
        self.set_ui_running(False)
        self.progress.setValue(100)
        self.progress_label.setText("Finished")

        self.log("Finished successfully.")
        self.log(f"MBLL folder: {results.get('folder')}")

        if "tiff_dir" in results:
            self.log(f"TIFF folder: {results.get('tiff_dir')}")

        QMessageBox.information(
            self,
            "Done",
            f"MBLL finished.\n\nOutput:\n{results.get('folder')}"
        )

        self.thread = None
        self.worker = None

    @Slot(str)
    def on_worker_error(self, error_text):
        self.set_ui_running(False)
        self.progress_label.setText("Error")
        self.log("ERROR during processing:")
        self.log(error_text)

        QMessageBox.critical(
            self,
            "Processing error",
            error_text
        )

        self.thread = None
        self.worker = None


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SpectralAnalysisApp()
    window.show()
    sys.exit(app.exec())