import easygui
import numpy as np
import blosc2
import tifffile
from datetime import datetime
import streamlit as st
import os, re, unicodedata
from typing import Optional, Dict
import pandas as pd


def demosaic(image: np.ndarray, mosaic_size: int = 4) -> np.ndarray:
    """Convierte mosaico 2D → cubo (rows, cols, m² bandas)."""
    rows, cols = image.shape[0] // mosaic_size, image.shape[1] // mosaic_size
    bands = mosaic_size * mosaic_size
    out = np.zeros((rows, cols, bands), dtype=image.dtype)
    k = 0
    for r in range(mosaic_size):
        for c in range(mosaic_size):
            out[:, :, k] = image[r::mosaic_size, c::mosaic_size]
            k += 1
    return out
def _slugify(txt: str) -> str:
    txt = unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode()
    txt = re.sub(r"[^0-9A-Za-z]+", "_", txt).strip("_")
    return txt[:60]
def _parse_log(path: str) -> Dict[str, str]:
    m: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                ts, msg = ln.split(" ", 1)
                m[ts.strip()] = msg.strip()
    return m
def save_images_to_tiff(
    b2nd_path: str,
    output_folder: str,
    grayscale: bool = False,
    mosaic_size: int = 4,
    band_idx: int = 0,
    log_path: Optional[str] = None,
    log_list: Optional[list] = None,
) -> None:
    """
    Guarda un único TIFF contiguo con N canales.

    - Cada pixel lleva todos los canales (shape -> (rows, cols, bands)).
    - Admite uint8/uint12/uint16/float32  y activa BigTIFF si hace falta.
    """
    def log(msg):
        print(msg)
        if log_list is not None:
            log_list.append(msg)

    log_map = _parse_log(log_path) if log_path and os.path.isfile(log_path) else {}

    data = blosc2.open(b2nd_path, mode="r")
    os.makedirs(output_folder, exist_ok=True)
    log(f"Total de cuadros: {data.shape[0]}")

    for i in range(data.shape[0]):
        img = data[i][...]
        try:
            ts = data.schunk.vlmeta["time_stamp"][i]
        except (AttributeError, KeyError, IndexError):
            ts = "N-A"

        log_msg = log_map.get(ts, "")
        core = f"{ts}_{_slugify(log_msg)}" if log_msg else ts

        # ── reorganizar bandas ─────────────────────────────
        if grayscale:
            if img.ndim == 3:
                img = img[:, :, band_idx]
            img = img[..., np.newaxis]
        else:
            if img.ndim == 2:
                img = demosaic(img, mosaic_size)
            # Asegura que img sea (rows, cols, bands)
            if img.ndim == 3 and img.shape[0] < img.shape[-1]:
                img = np.transpose(img, (1, 2, 0))

        log(f"[DEBUG] Antes de normalizar: shape={img.shape}, dtype={img.dtype}, min={img.min()}, max={img.max()}")

        # ── normalizar a 16 bits ──────────────────────────
        max_value = np.max(img)
        if max_value > 0:
            img = (img.astype(np.float32) / max_value * 65535).astype(np.uint16)
        else:
            img = img.astype(np.uint16)

        # ── reorganizar a (canal, fila, columna) ──────────
        if img.ndim == 3:
            img = np.transpose(img, (2, 0, 1))
        elif img.ndim == 2:
            img = img[np.newaxis, :, :]

        log(f"[DEBUG] Guardando {core}_frame_{i}.tif: shape={img.shape}, dtype={img.dtype}, min={img.min()}, max={img.max()}")

        out = os.path.join(output_folder, f"{core}_frame_{i}.tif")
        tifffile.imwrite(
            out,
            img,
            photometric="minisblack"
        )
        log(f"Guardado: {out}")
def main():
    st.sidebar.subheader('Decompression of .b2nd files')
    with st.sidebar.expander("Upload .b2nd files"):
        option = st.selectbox("Proveer el archivo .b2nd", ["Subir archivo", "Ruta manual"])
        if option == "Subir archivo":
            uploaded_file = st.file_uploader("Sube tu archivo .b2nd", type=["b2nd"])
            b2nd_path = None

        else:
            button_b2nd = st.button("select folder for .b2nd")
            if button_b2nd:
                b2nd_path = easygui.fileopenbox(
                    "Selecciona el archivo .b2nd",
                    default="*.b2nd",
                    filetypes=["*.b2nd"]
                )
                uploaded_file = None
            if b2nd_path and not os.path.isfile(b2nd_path):
                st.error("¡La ruta especificada no es válida!")
        st.caption('Output folder')
        # 2) Botón para seleccionar carpeta con Tkinter (mostrando discos externos)
        if st.button("Output folder"):
            selected_folder = easygui.diropenbox("Select output folder",default="/home/alonso/Desktop/")
            if selected_folder:
                st.session_state["selected_folder"] = selected_folder
                st.success(f"Carpeta seleccionada: {selected_folder}")
            else:
                st.warning("No se seleccionó ninguna carpeta (cerraste el diálogo).")


        subfolder_name = st.text_input("Name of subfolder", "")

        # 5) Tipo de cámara
        camera_type = st.selectbox("Tipo de cámara", ["Multiespectral", "Grayscale"])
        grayscale = (camera_type == "Grayscale")

        # 6) Botón final para descomprimir
        if st.button("Descomprimir y Guardar Imágenes"):
            # Verificar archivo
            if uploaded_file is not None or (b2nd_path and os.path.isfile(b2nd_path)):
                # Si se subió archivo, guardarlo temporalmente
                if uploaded_file is not None:
                    with open("uploaded_temp.b2nd", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    b2nd_path = "uploaded_temp.b2nd"

                # Resolver la carpeta de salida
                if "selected_folder" in st.session_state and st.session_state["selected_folder"]:
                    output_folder = st.session_state["selected_folder"]
                elif manual_output.strip():
                    output_folder = manual_output.strip()
                else:
                    # Carpeta por defecto: data + fecha/hora
                    output_folder = os.path.join(
                        "data",
                        datetime.now().strftime("%Y%m%d_%H%M%S")
                    )

                # Si hay subcarpeta, la anexamos
                if subfolder_name.strip():
                    output_folder = os.path.join(output_folder, subfolder_name.strip())

                os.makedirs(output_folder, exist_ok=True)

                with st.spinner("Descomprimiendo..."):
                    # Pasa el log decomp_log para logging en tiempo real
                    st.session_state["decomp_log"].clear()
                    save_images_to_tiff(b2nd_path, output_folder, grayscale=grayscale, log_list=st.session_state["decomp_log"])

                st.success(f"¡Imágenes guardadas en {output_folder}!")
            else:
                st.error("Por favor, especifica o sube un archivo .b2nd válido.")
col1, col2 = st.columns(2)
with col1:
    main()
    # Log para Decompression en un DataFrame
    if "decomp_log" not in st.session_state:
        st.session_state["decomp_log"] = []
    st.markdown("**Decompression Log:**")
    decomp_log_df = pd.DataFrame({"Log": st.session_state["decomp_log"]})
    st.dataframe(decomp_log_df, height=200, hide_index=True)

# ─── UTILIDADES DE REFLECTANCIA ───────────────────────────────────────────────

def load_image_stack_safe(folder, expected_channels=16):
    imgs = []
    for fname in sorted(os.listdir(folder)):
        if fname.startswith("._"):
            continue
        if not fname.lower().endswith((".tif", ".tiff")):
            continue
        path = os.path.join(folder, fname)
        try:
            with tifffile.TiffFile(path) as tif:
                if tif.series and tif.series[0].shape[0] == expected_channels:
                    arr = tif.series[0].asarray()
                else:
                    arr = tif.asarray(key=0)
            if arr.ndim == 3 and arr.shape[0] == expected_channels:
                arr = np.moveaxis(arr, 0, -1)
            if not (arr.ndim == 3 and arr.shape[-1] == expected_channels):
                continue
            imgs.append((fname, arr.astype(np.float32)))
        except Exception as e:
            continue
    return imgs

def calcular_y_guardar_reflectancia(dark_f, white_f, data_f):
    # iniciar log limpio
    st.session_state["reflect_log"] = []
    def log(msg, error=False):
        p = "❌ ERROR:" if error else "🔄"
        st.session_state["reflect_log"].append(f"{p} {msg}")

    log("1️⃣ Iniciando cálculo de reflectancia")

    # Carga stacks y chequeo
    try:
        white_imgs = [img for _,img in load_image_stack_safe(white_f)]
        log(f"WHITE: {len(white_imgs)} imágenes cargadas")
        dark_imgs  = [img for _,img in load_image_stack_safe(dark_f)]
        log(f"DARK : {len(dark_imgs)} imágenes cargadas")
        data_list  = load_image_stack_safe(data_f)
        log(f"DATA : {len(data_list)} imágenes encontradas")
    except Exception as e:
        log(str(e), error=True)
        return None

    if not white_imgs or not dark_imgs or not data_list:
        log("Faltan imágenes en alguna carpeta", error=True)
        return None

    # medians
    wm = np.median(np.stack(white_imgs,0), axis=0)
    dm = np.median(np.stack(dark_imgs, 0), axis=0)
    log(f"White mean {wm.shape}, Dark mean {dm.shape}")

    # suavizado
    sigma = max(wm.shape[:2])/50
    ws = gaussian_filter(wm, sigma=(sigma,sigma,0))
    log(f"White smooth {ws.shape}")

    # reflectancias
    refls = {}
    for fn, img in data_list:
        try:
            denom = ws - dm
            eps = 1e-6
            mask = denom <= eps
            denom[mask] = eps
            r = (img - dm)/denom
            r[mask] = np.nan
            r = np.clip(r,0,1).astype(np.float32)
            refls[os.path.splitext(fn)[0]] = r
            log(f"{fn}: reflect {r.shape}")
        except Exception as e:
            log(f"Error {fn}: {e}", error=True)

    if not refls:
        log("No se calculó ninguna reflectancia", error=True)
        return None

    out = os.path.join(data_f, "reflectance")
    os.makedirs(out, exist_ok=True)
    npz_path = os.path.join(out, "reflectance.npz")
    try:
        np.savez_compressed(npz_path, **refls)
        log(f"✅ Guardado {npz_path}")
    except Exception as e:
        log(f"Error al guardar NPZ: {e}", error=True)
        return None

    return npz_path

def write_log(msg, error=False):
    prefix = "❌ ERROR:" if error else "🔄"
    if "reflect_log" not in st.session_state:
        st.session_state["reflect_log"] = []
    st.session_state["reflect_log"].append(f"{prefix} {msg}")
    df = pd.DataFrame({"Log": st.session_state["reflect_log"]})
    log_area.dataframe(df, height=200, hide_index=True)

def main_reflectance():
    with col2:
        st.subheader("Reflectance Calculation")
        # asegurar estado
        for key in ("dark_folder","white_folder","data_folder"):
            if key not in st.session_state:
                st.session_state[key] = ""

        # selección de carpetas
        if st.button("Select DARK folder"):
            d = easygui.diropenbox("Selecciona DARK", default="/home/alonso/Desktop/")
            if d: st.session_state.dark_folder = d
        if st.button("Select WHITE folder"):
            w = easygui.diropenbox("Selecciona WHITE", default="/home/alonso/Desktop/")
            if w: st.session_state.white_folder = w
        if st.button("Select DATA folder"):
            D = easygui.diropenbox("Selecciona DATA", default="/home/alonso/Desktop/")
            if D: st.session_state.data_folder = D

        # área de log
        global log_area
        log_area = st.empty()
        if "reflect_log" not in st.session_state:
            st.session_state["reflect_log"] = []
        df = pd.DataFrame({"Log": st.session_state["reflect_log"]})
        log_area.dataframe(df, height=200, hide_index=True)

        # ejecutar cálculo
        if st.button("Start Reflectance"):
            d = st.session_state.dark_folder
            w = st.session_state.white_folder
            D = st.session_state.data_folder
            st.session_state["reflect_log"] = []
            if not (d and w and D):
                write_log("Faltan carpetas DARK, WHITE o DATA", error=True)
                st.error("❌ Selecciona DARK, WHITE y DATA primero")
                return

            write_log("Cargando imágenes DARK...")
            dark_imgs = [img for _, img in load_image_stack_safe(d)]
            write_log(f"Imágenes DARK cargadas: {len(dark_imgs)}")
            write_log("Cargando imágenes WHITE...")
            white_imgs = [img for _, img in load_image_stack_safe(w)]
            write_log(f"Imágenes WHITE cargadas: {len(white_imgs)}")
            write_log("Cargando imágenes DATA...")
            data_list = load_image_stack_safe(D)
            write_log(f"Imágenes DATA cargadas: {len(data_list)}")

            if not dark_imgs or not white_imgs or not data_list:
                write_log("Faltan imágenes en alguna carpeta", error=True)
                st.error("Faltan imágenes en alguna carpeta")
                return

            try:
                white_mean = np.median(np.stack(white_imgs, 0), axis=0)
                dark_mean = np.median(np.stack(dark_imgs, 0), axis=0)
                write_log(f"White mean: {white_mean.shape}, Dark mean: {dark_mean.shape}")

                sigma = max(white_mean.shape[:2]) / 50
                from scipy.ndimage import gaussian_filter
                white_smooth = gaussian_filter(white_mean, sigma=(sigma, sigma, 0))
                write_log(f"White smooth: {white_smooth.shape}")

                refls = {}
                for fname, img in data_list:
                    try:
                        denom = white_smooth - dark_mean
                        eps = 1e-6
                        mask = denom <= eps
                        denom[mask] = eps
                        r = (img - dark_mean) / denom
                        r[mask] = np.nan
                        r = np.clip(r, 0, 1).astype(np.float32)
                        refls[os.path.splitext(fname)[0]] = r
                        write_log(f"{fname}: reflectancia {r.shape}")
                    except Exception as e:
                        write_log(f"Error {fname}: {e}", error=True)

                if not refls:
                    write_log("No se calculó ninguna reflectancia", error=True)
                    st.error("No se calculó ninguna reflectancia")
                    return

                out = os.path.join(D, "reflectance")
                os.makedirs(out, exist_ok=True)
                npz_path = os.path.join(out, "reflectance.npz")
                np.savez_compressed(npz_path, **refls)
                write_log(f"✅ Guardado {npz_path}")
                st.success(f"¡Reflectancia lista! NPZ en:\n`{npz_path}`")
            except Exception as e:
                write_log(f"Error en el cálculo: {e}", error=True)
                st.error(f"Error en el cálculo: {e}")

# ─── EJECUCIÓN ────────────────────────────────────────────────────────────────
main_reflectance()