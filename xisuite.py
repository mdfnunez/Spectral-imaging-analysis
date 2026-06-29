from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from PySide6.QtCore import Signal, Slot, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


# ============================================================
# Dynamic import helpers
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PAGES_DIR = BASE_DIR / "pages"


def _first_existing(candidates: list[Path]) -> Path:
    for p in candidates:
        if p.exists():
            return p
    msg = "No encontré ninguno de estos archivos:\n" + "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(msg)


def import_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"No pude importar {module_name} desde {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


XIDEC_FILE = _first_existing([PAGES_DIR / "Xidec.py"])
BEER_FILE = _first_existing([
    PAGES_DIR / "XiBeer-Lambert .py",   # nombre original subido, con espacio antes de .py
    PAGES_DIR / "XiBeer-Lambert.py",
    PAGES_DIR / "XiBeer_Lambert.py",
])
ANALYSIS_FILE = _first_existing([PAGES_DIR / "XiAnalysis.py"])

xidec_mod = import_module_from_path("pages_xidec", XIDEC_FILE)
beer_mod = import_module_from_path("pages_xibeer_lambert", BEER_FILE)
analysis_mod = import_module_from_path("pages_xianalysis", ANALYSIS_FILE)


# ============================================================
# Shared state
# ============================================================

@dataclass
class PipelineState:
    xidec: Dict[str, Any] = field(default_factory=dict)
    mbll: Dict[str, Any] = field(default_factory=dict)
    analysis: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Page subclasses: same original UI + pipeline methods/signals
# ============================================================

class XidecPage(xidec_mod.XidecMinimal):
    """Original Xidec page + emits its output path when processing finishes."""

    outputReady = Signal(dict)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("1. Xidec")
        self.last_output: Dict[str, Any] = {}

    def export_metadata(self, run_dir, base_name, N, h4, w4, raw_shape, lambda_v, C):
        """
        Called by the original process_data().
        We let the original method write metadata, then we emit the reflectance output.
        """
        super().export_metadata(run_dir, base_name, N, h4, w4, raw_shape, lambda_v, C)

        npy_path = os.path.join(run_dir, f"{base_name}.npy")
        meta_json_path = os.path.join(run_dir, f"{base_name}.meta.json")
        meta_csv_path = os.path.join(run_dir, f"{base_name}.meta.csv")

        result = {
            "run_dir": run_dir,
            "reflectance_path": npy_path,
            "meta_json_path": meta_json_path if os.path.exists(meta_json_path) else None,
            "meta_csv_path": meta_csv_path if os.path.exists(meta_csv_path) else None,
            "input_path": os.path.abspath(self.data_path) if self.data_path else None,
            "shape": [int(N), 16, int(h4), int(w4)],
        }

        self.last_output = result
        self.log(f"Pipeline: reflectance disponible para Beer-Lambert: {npy_path}")
        self.outputReady.emit(result)


class BeerLambertPage(beer_mod.SpectralAnalysisApp):
    """Original Beer-Lambert page + method to load Xidec output by path."""

    outputReady = Signal(dict)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("2. Xi Beer-Lambert")
        self.last_output: Dict[str, Any] = {}

    def load_reflectance_path(self, path: str) -> bool:
        """Load reflectance .npy without opening a file dialog."""
        if not path:
            return False

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

            self.log(f"Pipeline: reflectance cargado desde Xidec: {path}")
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
            return True

        except Exception as e:
            QMessageBox.critical(self, "Error loading reflectance", str(e))
            self.log(f"ERROR loading reflectance from pipeline: {e}")
            return False

    @Slot(dict)
    def on_worker_finished(self, results):
        """Keep original behavior, then emit MBLL result dictionary."""
        self.last_output = dict(results)
        super().on_worker_finished(results)
        self.outputReady.emit(self.last_output)


class AnalysisPage(analysis_mod.RoiTrackingApp):
    """Original Analysis page + methods to load Beer-Lambert outputs by path."""

    outputReady = Signal(dict)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("3. Xi Analysis")
        self.last_output: Dict[str, Any] = {}

    def load_processed_path(self, path: str) -> bool:
        """Load a processed 3D .npy stack without opening a file dialog."""
        if not path:
            return False

        try:
            data = np.load(path, allow_pickle=False, mmap_mode="r")
            if data.ndim != 3:
                raise ValueError(f"El processed stack debe tener shape T,H,W. Recibido: {data.shape}")

            self.processed_stack = data
            self.processed_path = path
            self.processed_edit.setText(path)
            self.log(f"Pipeline: processed stack cargado desde Beer-Lambert: {path}")
            self.log(f"Shape={data.shape}, dtype={data.dtype}")

            self.update_frame_slider_limits()
            self.update_viewer()
            self.refresh_ready_state()
            return True

        except Exception as e:
            QMessageBox.critical(self, "Error loading processed stack", str(e))
            self.log(f"ERROR loading processed stack from pipeline: {e}")
            return False

    def load_tiff_folder_path(self, folder: str) -> bool:
        """Load a TIFF folder without opening a folder dialog."""
        if not folder:
            return False

        folder = self._resolve_tiff_folder(folder)
        files = analysis_mod.list_tiff_files(folder) if folder else []
        if not files:
            self.log(f"Pipeline: no encontré TIFFs en {folder}")
            return False

        self.tiff_folder = folder
        self.tiff_files = files
        self.tiff_edit.setText(folder)
        self.log(f"Pipeline: TIFF folder cargado: {folder}")
        self.log(f"TIFF files found: {len(files)}")

        self.update_frame_slider_limits()
        self.update_viewer()
        self.refresh_ready_state()
        return True

    def load_metadata_path(self, path: Optional[str]) -> bool:
        """Load metadata CSV if available."""
        if not path or not os.path.exists(path):
            return False

        try:
            md = pd.read_csv(path)
            self.metadata = md
            self.metadata_path = path
            self.metadata_edit.setText(path)
            self.log(f"Pipeline: metadata CSV cargado: {path}")
            self.log(f"Metadata shape: {md.shape}")
            return True
        except Exception as e:
            self.log(f"Pipeline: no pude cargar metadata CSV: {e}")
            return False

    @staticmethod
    def _resolve_tiff_folder(folder: str) -> str:
        """
        Beer-Lambert exports a parent folder with grayscale/ and rgb/.
        XiAnalysis expects the folder containing .tif files directly.
        """
        if not folder:
            return folder

        candidates = [
            os.path.join(folder, "grayscale"),
            os.path.join(folder, "rgb"),
            folder,
        ]
        for c in candidates:
            if os.path.isdir(c) and analysis_mod.list_tiff_files(c):
                return c
        return folder

    @Slot(dict)
    def on_worker_finished(self, result):
        """Keep original behavior, then emit final Analysis outputs."""
        self.last_output = dict(result)
        super().on_worker_finished(result)
        self.outputReady.emit(self.last_output)


# ============================================================
# Main unified window
# ============================================================

class XiSuiteWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("XiSuite - Xidec → Beer-Lambert → Analysis")
        self.resize(1500, 930)

        self.state = PipelineState()

        self.xidec_page = XidecPage()
        self.beer_page = BeerLambertPage()
        self.analysis_page = AnalysisPage()

        self.tabs = QTabWidget()
        self.tabs.addTab(self.xidec_page, "1. Xidec")
        self.tabs.addTab(self.beer_page, "2. Beer-Lambert")
        self.tabs.addTab(self.analysis_page, "3. Analysis")

        self._build_ui()
        self._connect_signals()
        self._refresh_status()

    def _build_ui(self):
        root = QWidget()
        layout = QVBoxLayout(root)

        panel = QGroupBox("Pipeline")
        grid = QGridLayout(panel)

        self.lbl_xidec = QLabel("Xidec: sin output")
        self.lbl_xidec.setWordWrap(True)
        self.lbl_beer = QLabel("Beer-Lambert: sin output")
        self.lbl_beer.setWordWrap(True)
        self.lbl_analysis = QLabel("Analysis: sin output")
        self.lbl_analysis.setWordWrap(True)

        self.btn_send_xidec = QPushButton("Usar output de Xidec en Beer-Lambert")
        self.btn_send_xidec.clicked.connect(self.send_xidec_to_beer)

        self.mbll_output_combo = QComboBox()
        self.mbll_output_combo.addItem("StO₂ / StO2.npy", "StO2_path")
        self.mbll_output_combo.addItem("HbO₂ / HbO2.npy", "HbO2_path")
        self.mbll_output_combo.addItem("Hb / Hb.npy", "Hb_path")
        self.mbll_output_combo.addItem("ΔHbO₂ / dHbO2.npy", "dHbO2_path")
        self.mbll_output_combo.addItem("ΔHb / dHb.npy", "dHb_path")
        self.mbll_output_combo.currentIndexChanged.connect(self._on_mbll_combo_changed)

        self.btn_send_mbll = QPushButton("Usar output MBLL en Analysis")
        self.btn_send_mbll.clicked.connect(self.send_mbll_to_analysis)

        self.btn_open_current_folder = QPushButton("Abrir última carpeta de salida")
        self.btn_open_current_folder.clicked.connect(self.open_latest_output_folder)

        grid.addWidget(QLabel("1"), 0, 0, alignment=Qt.AlignTop)
        grid.addWidget(self.lbl_xidec, 0, 1)
        grid.addWidget(self.btn_send_xidec, 0, 2)

        grid.addWidget(QLabel("2"), 1, 0, alignment=Qt.AlignTop)
        grid.addWidget(self.lbl_beer, 1, 1)
        grid.addWidget(self.mbll_output_combo, 1, 2)
        grid.addWidget(self.btn_send_mbll, 1, 3)

        grid.addWidget(QLabel("3"), 2, 0, alignment=Qt.AlignTop)
        grid.addWidget(self.lbl_analysis, 2, 1, 1, 3)
        grid.addWidget(self.btn_open_current_folder, 0, 3)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)

        layout.addWidget(panel)
        layout.addWidget(line)
        layout.addWidget(self.tabs, stretch=1)
        self.setCentralWidget(root)

    def _connect_signals(self):
        self.xidec_page.outputReady.connect(self.on_xidec_ready)
        self.beer_page.outputReady.connect(self.on_mbll_ready)
        self.analysis_page.outputReady.connect(self.on_analysis_ready)

    def _short_path(self, path: Optional[str], max_len: int = 105) -> str:
        if not path:
            return "—"
        text = str(path)
        if len(text) <= max_len:
            return text
        return "…" + text[-max_len:]

    def _refresh_status(self):
        x = self.state.xidec
        b = self.state.mbll
        a = self.state.analysis

        self.lbl_xidec.setText(
            "Xidec output: " + self._short_path(x.get("reflectance_path"))
        )

        self.lbl_beer.setText(
            "Beer-Lambert output folder: " + self._short_path(b.get("folder"))
        )

        csv_path = a.get("csv_path") or self.analysis_page.csv_path
        self.lbl_analysis.setText(
            "Analysis output CSV: " + self._short_path(csv_path)
        )

        self.btn_send_xidec.setEnabled(bool(x.get("reflectance_path")))
        self.btn_send_mbll.setEnabled(bool(b))
        self.btn_open_current_folder.setEnabled(self._latest_output_folder() is not None)

    @Slot(dict)
    def on_xidec_ready(self, result: Dict[str, Any]):
        self.state.xidec = dict(result)
        self._refresh_status()
        self.send_xidec_to_beer(auto=True)

    @Slot(dict)
    def on_mbll_ready(self, result: Dict[str, Any]):
        self.state.mbll = dict(result)
        self._refresh_status()
        self.send_mbll_to_analysis(auto=True)

    @Slot(dict)
    def on_analysis_ready(self, result: Dict[str, Any]):
        self.state.analysis = dict(result)
        self._refresh_status()

    def send_xidec_to_beer(self, auto: bool = False):
        path = self.state.xidec.get("reflectance_path")
        if not path or not os.path.exists(path):
            if not auto:
                QMessageBox.warning(self, "Sin output de Xidec", "Todavía no hay reflectance .npy válido.")
            return

        ok = self.beer_page.load_reflectance_path(path)
        if ok:
            self.tabs.setCurrentWidget(self.beer_page)
        self._refresh_status()

    def _on_mbll_combo_changed(self):
        if self.state.mbll:
            self.send_mbll_to_analysis(auto=True)

    def send_mbll_to_analysis(self, auto: bool = False):
        results = self.state.mbll
        if not results:
            if not auto:
                QMessageBox.warning(self, "Sin output MBLL", "Todavía no hay resultados de Beer-Lambert.")
            return

        key = self.mbll_output_combo.currentData()
        processed_path = results.get(key)
        if not processed_path or not os.path.exists(processed_path):
            if not auto:
                QMessageBox.warning(
                    self,
                    "Output no disponible",
                    f"No encontré el archivo seleccionado para Analysis:\n{processed_path}",
                )
            return

        ok_processed = self.analysis_page.load_processed_path(processed_path)

        # TIFF folder is optional for viewing only, but required for ROI tracking in your current Analysis page.
        tiff_ok = False
        tiff_parent = results.get("tiff_dir")
        if tiff_parent:
            tiff_ok = self.analysis_page.load_tiff_folder_path(tiff_parent)

        # Metadata from Xidec is useful if available.
        self.analysis_page.load_metadata_path(self.state.xidec.get("meta_csv_path"))

        if ok_processed:
            self.tabs.setCurrentWidget(self.analysis_page)
            if not tiff_ok:
                self.analysis_page.log(
                    "Pipeline: no se cargó TIFF folder automáticamente. "
                    "Activa 'Export TIFFs after MBLL' o carga manualmente una carpeta TIFF."
                )

        self._refresh_status()

    def _latest_output_folder(self) -> Optional[str]:
        candidates = [
            self.analysis_page.current_output_dir(),
            self.state.mbll.get("folder"),
            self.state.mbll.get("tiff_dir"),
            self.state.xidec.get("run_dir"),
        ]
        for c in candidates:
            if c and os.path.isdir(c):
                return c
        return None

    def open_latest_output_folder(self):
        folder = self._latest_output_folder()
        if not folder:
            QMessageBox.information(self, "Sin carpeta", "Todavía no hay carpeta de salida.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(folder))


# ============================================================
# Entrypoint
# ============================================================

def main():
    app = QApplication(sys.argv)
    window = XiSuiteWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()