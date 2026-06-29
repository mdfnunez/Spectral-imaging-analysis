import sys
import os
import json
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np
import pandas as pd
import blosc2
import tifffile

from numpy.lib.format import open_memmap

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QTextEdit
)


VALID_EXTENSIONS = (".b2nd", ".tif", ".tiff")


class MultiSpectralStack:
    """
    Lector común para:
    - .b2nd
    - .tif / .tiff multipágina
    - TIFF stack
    - TIFF con bandas ya separadas como 16,H,W o H,W,16
    """

    def __init__(self, path):
        self.path = path
        self.ext = os.path.splitext(path)[1].lower()
        self.kind = None
        self.obj = None
        self.tif = None
        self.array = None
        self.axes = ""
        self.mode = None

        if self.ext == ".b2nd":
            self.kind = "b2nd"
            self.obj = blosc2.open(path, mode="r")
            self.shape = tuple(self.obj.shape)

            if len(self.shape) == 2:
                self.n_frames = 1
                self.frame_shape = self.shape
            else:
                self.n_frames = self.shape[0]
                self.frame_shape = self.shape[1:]

        elif self.ext in (".tif", ".tiff"):
            self.kind = "tiff"
            self.tif = tifffile.TiffFile(path)
            series = self.tif.series[0]
            self.axes = getattr(series, "axes", "")
            self.shape = tuple(series.shape)

            # Caso 1: TIFF multipágina con páginas 2D
            if len(self.tif.pages) > 1 and all(
                len(p.shape) == 2 for p in self.tif.pages[:min(5, len(self.tif.pages))]
            ):
                # Si tiene exactamente 16 páginas, asumimos que son 16 bandas de una sola imagen.
                # Si tienes un video raw con exactamente 16 frames, cambia esto a "pages2d_frames".
                if len(self.tif.pages) == 16:
                    self.mode = "pages2d_16bands_single_frame"
                    self.n_frames = 1
                    self.frame_shape = (16, *self.tif.pages[0].shape)
                else:
                    self.mode = "pages2d_frames"
                    self.n_frames = len(self.tif.pages)
                    self.frame_shape = tuple(self.tif.pages[0].shape)

            # Caso 2: TIFF stack en una serie
            else:
                self.mode = "array"
                self.array = series.asarray()
                self.shape = tuple(self.array.shape)

                if self.array.ndim == 2:
                    self.n_frames = 1
                    self.frame_shape = self.array.shape

                elif self.array.ndim == 3:
                    # Ejemplos:
                    # T,H,W
                    # 16,H,W
                    # H,W,16

                    if self.array.shape[0] == 16:
                        self.n_frames = 1
                        self.frame_shape = self.array.shape

                    elif self.array.shape[-1] == 16:
                        self.n_frames = 1
                        self.frame_shape = self.array.shape

                    else:
                        self.n_frames = self.array.shape[0]
                        self.frame_shape = self.array.shape[1:]

                elif self.array.ndim == 4:
                    # Ejemplos:
                    # T,16,H,W
                    # T,H,W,16
                    self.n_frames = self.array.shape[0]
                    self.frame_shape = self.array.shape[1:]

                else:
                    raise ValueError(f"TIFF con forma no soportada: {self.array.shape}")

        else:
            raise ValueError(f"Formato no soportado: {self.ext}")

    def __len__(self):
        return self.n_frames

    def __getitem__(self, i):
        if self.kind == "b2nd":
            if len(self.shape) == 2:
                if i != 0:
                    raise IndexError("Este .b2nd solo tiene un frame.")
                return np.asarray(self.obj[:])
            return np.asarray(self.obj[i])

        if self.kind == "tiff":
            if self.mode == "pages2d_frames":
                return self.tif.pages[i].asarray()

            if self.mode == "pages2d_16bands_single_frame":
                if i != 0:
                    raise IndexError("Este TIFF tiene una sola imagen multiespectral de 16 bandas.")
                return np.stack([p.asarray() for p in self.tif.pages], axis=0)

            if self.mode == "array":
                if self.array.ndim == 2:
                    if i != 0:
                        raise IndexError("Este TIFF solo tiene un frame.")
                    return self.array

                if self.array.ndim == 3:
                    if self.n_frames == 1:
                        return self.array
                    return self.array[i]

                if self.array.ndim == 4:
                    return self.array[i]

        raise RuntimeError("Estado inválido en MultiSpectralStack.")


def frame_to_16_bands(frame):
    """
    Convierte un frame a forma:

    (16, h, w)

    Acepta:
    - raw mosaic xiSpec 4x4: H,W
    - imagen ya separada como 16,H,W
    - imagen ya separada como H,W,16
    """

    frame = np.asarray(frame)

    # Caso 1: raw mosaic 4x4
    if frame.ndim == 2:
        H, W = frame.shape
        h4, w4 = H // 4, W // 4

        if h4 == 0 or w4 == 0:
            raise ValueError(f"Frame demasiado pequeño para demosaic 4x4: {frame.shape}")

        # Recortar bordes si no son múltiplos de 4
        frame = frame[:h4 * 4, :w4 * 4]

        bands_16 = np.stack(
            [frame[r::4, c::4] for r in range(4) for c in range(4)],
            axis=0
        )

        return bands_16.astype(np.float32)

    # Caso 2: ya viene como 16,H,W
    if frame.ndim == 3 and frame.shape[0] == 16:
        return frame.astype(np.float32)

    # Caso 3: ya viene como H,W,16
    if frame.ndim == 3 and frame.shape[-1] == 16:
        return np.moveaxis(frame, -1, 0).astype(np.float32)

    raise ValueError(
        f"No puedo interpretar este frame como 16 bandas. Shape recibido: {frame.shape}. "
        "Espero H,W o 16,H,W o H,W,16."
    )


def median_reference(path):
    """
    Carga white/dark desde .b2nd, .tif o .tiff
    y calcula mediana en forma:

    (16, h, w)
    """

    stack = MultiSpectralStack(path)
    n = len(stack)

    if n == 0:
        raise ValueError(f"La referencia está vacía: {path}")

    first = frame_to_16_bands(stack[0])
    arr = np.empty((n, *first.shape), dtype=np.float32)
    arr[0] = first

    for i in range(1, n):
        arr[i] = frame_to_16_bands(stack[i])

    return np.median(arr, axis=0).astype(np.float32)


def find_reference_file(folder, name):
    """
    Busca referencias en la misma carpeta.

    Acepta:
    white.b2nd
    white.tif
    white.tiff
    White.tif
    WHITE.TIFF
    dark.b2nd
    dark.tif
    dark.tiff

    También acepta nombres como:
    white_reference.tif
    dark_ref.tiff
    """

    exact_candidates = []
    partial_candidates = []

    for filename in os.listdir(folder):
        stem, ext = os.path.splitext(filename)

        if ext.lower() not in VALID_EXTENSIONS:
            continue

        stem_lower = stem.lower()
        name_lower = name.lower()

        full_path = os.path.join(folder, filename)

        if stem_lower == name_lower:
            exact_candidates.append(full_path)

        elif name_lower in stem_lower:
            partial_candidates.append(full_path)

    if exact_candidates:
        return exact_candidates[0]

    if partial_candidates:
        return partial_candidates[0]

    return None


class XidecMinimal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Xidec - Minimal Decompressor")
        self.resize(600, 420)

        # Ajusta esta ruta si tu XML está en otra carpeta
        self.xml_path = "ximea files/CMV2K-SSM4x4-460_600-15.7.20.6.xml"

        # Carpeta inicial
        self.default_dir = "/home/alonso/Desktop/"

        self.data_path = None
        self.data_loaded = None
        self.white_median = None
        self.dark_median = None

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.btn_select = QPushButton("1. Seleccionar archivo .b2nd / .tif / .tiff")
        self.btn_select.clicked.connect(self.select_file)
        layout.addWidget(self.btn_select)

        self.lbl_file = QLabel("Archivo: Ninguno")
        self.lbl_file.setWordWrap(True)
        layout.addWidget(self.lbl_file)

        self.lbl_refs = QLabel("Referencias White/Dark: Pendiente")
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
        QApplication.processEvents()

    def load_correction_matrix(self):
        if not os.path.exists(self.xml_path):
            raise FileNotFoundError(f"XML no encontrado en: {self.xml_path}")

        root = ET.parse(self.xml_path).getroot()

        target = next(
            (
                cm for cm in root.findall(".//correction_matrix")
                if cm.findtext("name") == "hsi_reflectance"
                and cm.findtext("type") == "reflectance"
            ),
            None
        )

        if target is None:
            raise RuntimeError("No se encontró 'hsi_reflectance' en el XML.")

        C_rows = []
        lambda_v = []

        for vb in target.findall(".//virtual_bands/virtual_band"):
            lambda_v.append(float(vb.findtext("wavelength_nm")))
            vals = [float(x) for x in vb.find("coefficients").get("values").split()]
            C_rows.append(vals)

        C = np.asarray(C_rows, dtype=np.float32)
        lambda_v = np.asarray(lambda_v, dtype=np.float32)

        if C.shape != (16, 16):
            self.log(f"Advertencia: la matriz C tiene forma {C.shape}, se esperaba (16, 16).")

        return C, lambda_v

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar archivo multiespectral",
            self.default_dir,
            "Archivos multiespectrales (*.b2nd *.tif *.tiff);;Blosc2 (*.b2nd);;TIFF (*.tif *.tiff);;Todos (*.*)"
        )

        if not file_path:
            return

        try:
            self.btn_process.setEnabled(False)
            self.progress.setValue(0)
            self.white_median = None
            self.dark_median = None

            self.data_path = file_path
            self.data_loaded = MultiSpectralStack(file_path)

            folder = os.path.dirname(file_path)

            self.lbl_file.setText(f"Archivo: {os.path.basename(file_path)}")

            self.log(f"Cargado: {os.path.basename(file_path)}")
            self.log(f"Formato detectado: {self.data_loaded.kind}")
            self.log(f"Modo interno: {self.data_loaded.mode}")
            self.log(f"Frames detectados: {len(self.data_loaded)}")
            self.log(f"Forma de frame detectada: {self.data_loaded.frame_shape}")

            compatible_files = [
                f for f in os.listdir(folder)
                if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS
            ]

            self.log("Archivos compatibles en la carpeta:")
            self.log(str(compatible_files))

            # Buscar white
            white_path = find_reference_file(folder, "white")

            if white_path:
                self.log(f"White encontrado: {os.path.basename(white_path)}")
                self.white_median = median_reference(white_path)
                self.log(f"White calculado con forma: {self.white_median.shape}")
            else:
                self.log("ERROR: no encontré white en .b2nd/.tif/.tiff")

            # Buscar dark
            dark_path = find_reference_file(folder, "dark")

            if dark_path:
                self.log(f"Dark encontrado: {os.path.basename(dark_path)}")
                self.dark_median = median_reference(dark_path)
                self.log(f"Dark calculado con forma: {self.dark_median.shape}")
            else:
                self.log("ERROR: no encontré dark en .b2nd/.tif/.tiff")

            if self.white_median is not None and self.dark_median is not None:
                if self.white_median.shape != self.dark_median.shape:
                    raise ValueError(
                        f"White y Dark tienen formas diferentes: "
                        f"{self.white_median.shape} vs {self.dark_median.shape}"
                    )

                self.lbl_refs.setText("Referencias White/Dark: Encontradas y calculadas")
                self.btn_process.setEnabled(True)

            else:
                self.lbl_refs.setText("Referencias White/Dark: FALTANTES")
                self.btn_process.setEnabled(False)

        except Exception as e:
            self.log(f"ERROR al cargar archivo: {str(e)}")
            self.lbl_refs.setText("Referencias White/Dark: ERROR")
            self.btn_process.setEnabled(False)

    def process_data(self):
        self.btn_process.setEnabled(False)
        self.btn_select.setEnabled(False)

        try:
            C, lambda_v = self.load_correction_matrix()

            N = len(self.data_loaded)

            first_frame = self.data_loaded[0]
            first_bands = frame_to_16_bands(first_frame)

            if first_bands.shape != self.white_median.shape:
                raise ValueError(
                    "El archivo de datos y las referencias no tienen la misma forma después del demosaic.\n"
                    f"Datos: {first_bands.shape}\n"
                    f"White: {self.white_median.shape}\n"
                    f"Dark: {self.dark_median.shape}"
                )

            _, h4, w4 = first_bands.shape
            raw_shape = list(np.asarray(first_frame).shape)

            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            input_name = os.path.splitext(os.path.basename(self.data_path))[0]

            base_name = f"reflectance_{input_name}_{ts}"
            run_dir = os.path.join(os.path.dirname(self.data_path), base_name)
            os.makedirs(run_dir, exist_ok=True)

            npy_path = os.path.join(run_dir, f"{base_name}.npy")

            all_refl = open_memmap(
                npy_path,
                mode="w+",
                dtype=np.float32,
                shape=(N, 16, h4, w4)
            )

            self.log("Iniciando demosaic, dark/white correction y matriz XML...")
            self.progress.setMaximum(N)
            self.progress.setValue(0)

            denominator = self.white_median - self.dark_median
            denominator = denominator + 1e-6

            for i in range(N):
                raw_frame = self.data_loaded[i]
                bands_16 = frame_to_16_bands(raw_frame)

                with np.errstate(divide="ignore", invalid="ignore"):
                    refl = (bands_16 - self.dark_median) / denominator

                corr = np.tensordot(C, refl, axes=(1, 0))
                corr = np.where(np.isfinite(corr), corr, 0.0).astype(np.float32)

                all_refl[i] = corr

                self.progress.setValue(i + 1)

                if i % 10 == 0:
                    QApplication.processEvents()

            all_refl.flush()

            self.log(f"Éxito. Archivo guardado en:")
            self.log(npy_path)

            self.export_metadata(
                run_dir=run_dir,
                base_name=base_name,
                N=N,
                h4=h4,
                w4=w4,
                raw_shape=raw_shape,
                lambda_v=lambda_v,
                C=C
            )

        except Exception as e:
            self.log(f"ERROR: {str(e)}")

        finally:
            self.btn_process.setEnabled(True)
            self.btn_select.setEnabled(True)

    def export_metadata(self, run_dir, base_name, N, h4, w4, raw_shape, lambda_v, C):
        meta_json_path = os.path.join(run_dir, f"{base_name}.meta.json")

        static_meta = {
            "file_data": f"{base_name}.npy",
            "input_file": os.path.abspath(self.data_path),
            "input_format": self.data_loaded.kind,
            "input_extension": os.path.splitext(self.data_path)[1].lower(),
            "shape_data": [int(N), 16, int(h4), int(w4)],
            "xml_file": os.path.abspath(self.xml_path),
            "virtual_wavelengths_nm": [float(x) for x in lambda_v.tolist()],
            "capture_shape_raw_or_input_frame": raw_shape,
            "correction_matrix_C": [[float(v) for v in row] for row in C.tolist()]
        }

        if self.data_loaded.kind == "tiff":
            static_meta["tiff_axes"] = self.data_loaded.axes
            static_meta["tiff_series_shape"] = list(self.data_loaded.shape)
            static_meta["tiff_mode"] = self.data_loaded.mode

        with open(meta_json_path, "w", encoding="utf-8") as f:
            json.dump(static_meta, f, ensure_ascii=False, indent=2)

        self.log(f"Metadata JSON guardada en: {meta_json_path}")

        # Metadata adicional solo para .b2nd
        if self.data_loaded.kind == "b2nd":
            try:
                if hasattr(self.data_loaded.obj, "vlmeta") and self.data_loaded.obj.vlmeta:
                    meta = self.data_loaded.obj.vlmeta

                    def get_meta(key):
                        val = meta.get(key.encode() if isinstance(key, str) else key, [])
                        if isinstance(val, (bytes, bytearray)):
                            return [val.decode(errors="ignore")]
                        try:
                            return val.tolist() if isinstance(val, np.ndarray) else list(val)
                        except Exception:
                            return [val]

                    ts_list = get_meta("time_stamp")
                    exp_list = get_meta("exposure_us")
                    chip_list = get_meta("temperature_chip")

                    if all([ts_list, exp_list, chip_list]):
                        n = min(len(ts_list), len(exp_list), len(chip_list))

                        if n > 0:
                            df = pd.DataFrame({
                                "Timestamp": ts_list[:n],
                                "Exposure_us": exp_list[:n],
                                "Chip_temperature": chip_list[:n]
                            })

                            csv_path = os.path.join(run_dir, f"{base_name}.meta.csv")
                            df.to_csv(csv_path, index=False)
                            self.log(f"Metadata CSV guardada en: {csv_path}")

            except Exception as e:
                self.log(f"Advertencia: no pude exportar metadata .b2nd: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = XidecMinimal()
    window.show()
    sys.exit(app.exec())