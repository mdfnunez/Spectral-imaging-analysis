import sys
import os
import json
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np
import pandas as pd
import blosc2
from numpy.lib.format import open_memmap

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QPushButton, QLabel, QFileDialog, QProgressBar, QTextEdit)
from PySide6.QtCore import Qt

class XidecMinimal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Xidec - Minimal Decompressor")
        self.resize(500, 350)
        
        # Rutas por defecto (Ajustar según necesidad)
        self.xml_path = "ximea files/CMV2K-SSM4x4-460_600-15.7.20.6.xml"
        self.default_dir = "/home/alonso/Desktop/"

        self.b2nd_path = None
        self.b2nd_loaded = None
        self.white_median = None
        self.dark_median = None

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.btn_select = QPushButton("1. Seleccionar archivo .b2nd")
        self.btn_select.clicked.connect(self.select_file)
        layout.addWidget(self.btn_select)

        self.lbl_file = QLabel("Archivo: Ninguno")
        self.lbl_file.setWordWrap(True)
        layout.addWidget(self.lbl_file)

        self.lbl_refs = QLabel("Referencias (White/Dark): Pendiente")
        layout.addWidget(self.lbl_refs)

        self.btn_process = QPushButton("2. Demosaic & Calibrar")
        self.btn_process.setEnabled(False)
        self.btn_process.clicked.connect(self.process_data)
        layout.addWidget(self.btn_process)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.logs = QTextEdit()
        self.logs.setReadOnly(True)
        layout.addWidget(self.logs)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def log(self, text):
        self.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {text}")
        QApplication.processEvents() # Actualiza la UI para no congelar la ventana

    def load_correction_matrix(self):
        if not os.path.exists(self.xml_path):
            raise FileNotFoundError(f"XML no encontrado en: {self.xml_path}")
            
        root = ET.parse(self.xml_path).getroot()
        target = next((cm for cm in root.findall(".//correction_matrix") 
                       if cm.findtext("name") == "hsi_reflectance" and cm.findtext("type") == "reflectance"), None)
        if target is None:
            raise RuntimeError("No se encontró 'hsi_reflectance' en el XML.")

        C_rows, lambda_v = [], []
        for vb in target.findall(".//virtual_bands/virtual_band"):
            lambda_v.append(float(vb.findtext("wavelength_nm")))
            vals = [float(x) for x in vb.find("coefficients").get("values").split()]
            C_rows.append(vals)

        return np.asarray(C_rows, dtype=np.float32), np.asarray(lambda_v, dtype=np.float32)

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select .b2nd file", self.default_dir, "Blosc2 files (*.b2nd)")
        if not file_path:
            return

        self.b2nd_path = file_path
        self.b2nd_loaded = blosc2.open(file_path, mode="r")
        b2nd_folder = os.path.dirname(file_path)
        
        self.lbl_file.setText(f"Archivo: {os.path.basename(file_path)}")
        self.log(f"Cargado: {os.path.basename(file_path)}")

        # Procesar White
        white_path = os.path.join(b2nd_folder, "white.b2nd")
        if os.path.exists(white_path):
            stack = blosc2.open(white_path, mode="r")
            med = np.median(stack, axis=0)
            self.white_median = np.stack([med[i::4, j::4] for i in range(4) for j in range(4)], axis=0).astype(np.float32)
            self.log("White reference cargada.")
        else:
            self.white_median = None
            self.log("ERROR: white.b2nd no encontrado.")

        # Procesar Dark
        dark_path = os.path.join(b2nd_folder, "dark.b2nd")
        if os.path.exists(dark_path):
            stack = blosc2.open(dark_path, mode="r")
            med = np.median(stack, axis=0)
            self.dark_median = np.stack([med[i::4, j::4] for i in range(4) for j in range(4)], axis=0).astype(np.float32)
            self.log("Dark reference cargada.")
        else:
            self.dark_median = None
            self.log("ERROR: dark.b2nd no encontrado.")

        # Habilitar botón de procesar si todo es correcto
        if self.white_median is not None and self.dark_median is not None:
            self.lbl_refs.setText("Referencias (White/Dark): Encontradas y calculadas")
            self.btn_process.setEnabled(True)
        else:
            self.lbl_refs.setText("Referencias (White/Dark): FALTANTES")
            self.btn_process.setEnabled(False)

    def process_data(self):
        self.btn_process.setEnabled(False)
        self.btn_select.setEnabled(False)
        
        try:
            C, lambda_v = self.load_correction_matrix()
            N, H, W = len(self.b2nd_loaded), *self.b2nd_loaded[0].shape
            h4, w4 = H // 4, W // 4
            
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            base_name = f"reflectance_{os.path.basename(self.b2nd_path)}_{ts}"
            run_dir = os.path.join(os.path.dirname(self.b2nd_path), base_name)
            os.makedirs(run_dir, exist_ok=True)

            npy_path = os.path.join(run_dir, f"{base_name}.npy")
            all_refl = open_memmap(npy_path, mode="w+", dtype=np.float32, shape=(N, 16, h4, w4))

            self.log("Iniciando Demosaic y Corrección...")
            self.progress.setMaximum(N)

            for i in range(N):
                raw_frame = self.b2nd_loaded[i]
                bands_16 = np.stack([raw_frame[r::4, c::4] for r in range(4) for c in range(4)], axis=0).astype(np.float32)
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    refl = (bands_16 - self.dark_median) / (self.white_median - self.dark_median + 1e-6)
                
                corr = np.tensordot(C, refl, axes=(1, 0))
                all_refl[i] = np.where(np.isfinite(corr), corr, 0.0).astype(np.float32)
                
                # Actualizar progreso
                self.progress.setValue(i + 1)
                if i % 10 == 0:  # Evita sobrecargar la UI
                    QApplication.processEvents()

            all_refl.flush()
            self.log(f"Éxito. Archivo guardado en: {npy_path}")
            self.export_metadata(run_dir, base_name, N, h4, w4, H, W, lambda_v, C)

        except Exception as e:
            self.log(f"ERROR: {str(e)}")
            
        finally:
            self.btn_process.setEnabled(True)
            self.btn_select.setEnabled(True)

    def export_metadata(self, run_dir, base_name, N, h4, w4, H, W, lambda_v, C):
        meta_json_path = os.path.join(run_dir, f"{base_name}.meta.json")
        static_meta = {
            "file_data": f"{base_name}.npy",
            "shape_data": [int(N), 16, int(h4), int(w4)],
            "xml_file": os.path.abspath(self.xml_path),
            "virtual_wavelengths_nm": [float(x) for x in lambda_v.tolist()],
            "capture_shape_raw": [int(H), int(W)],
            "correction_matrix_C": [[float(v) for v in row] for row in C.tolist()]
        }
        with open(meta_json_path, "w", encoding="utf-8") as f:
            json.dump(static_meta, f, ensure_ascii=False, indent=2)
        
        # DataFrame Meta
        if hasattr(self.b2nd_loaded, "vlmeta") and self.b2nd_loaded.vlmeta:
            meta = self.b2nd_loaded.vlmeta
            
            def get_meta(key):
                val = meta.get(key.encode() if isinstance(key, str) else key, [])
                if isinstance(val, (bytes, bytearray)): return [val.decode(errors='ignore')]
                try: return val.tolist() if isinstance(val, np.ndarray) else list(val)
                except: return [val]

            ts_list, exp_list, chip_list = get_meta("time_stamp"), get_meta("exposure_us"), get_meta("temperature_chip")
            n = min(len(ts_list), len(exp_list), len(chip_list)) if all([ts_list, exp_list, chip_list]) else 0
            
            if n > 0:
                df = pd.DataFrame({"Timestamp": ts_list[:n], "Exposure_us": exp_list[:n], "Chip_temperature": chip_list[:n]})
                df.to_csv(os.path.join(run_dir, f"{base_name}.meta.csv"), index=False)
                
        self.log("Metadatos (JSON/CSV) guardados correctamente.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = XidecMinimal()
    window.show()
    sys.exit(app.exec())
