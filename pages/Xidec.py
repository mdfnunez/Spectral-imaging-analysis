import streamlit as st
import blosc2
import easygui
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from datetime import datetime
from tifffile import imwrite
from numpy.lib.format import open_memmap  
import os, json

#Header
st.set_page_config('Xidec',layout="wide")
st.title('Xidec')
st.caption('Software for decompression of .b2nd files from the Xilens program')
st.caption('Output: reflectance + corrección espectral (virtual bands) según XML; orden ascendente por λ (460→594 nm).')

# Paths (Change paths to local PC if needed)
global default
default="/home/alonso/Desktop/" #change the default path accordingly
global xml_path #path to xml file
xml_path="ximea files/CMV2K-SSM4x4-460_600-15.7.20.6.xml"
#####


def load_correction_matrix(xml_path):
    """
    Lee del XML la matriz de corrección espectral para REFLECTANCIA (hsi_reflectance).
    Devuelve:
      C: np.ndarray (16,16)  -> filas = 'virtual_bands', columnas = canales crudos (orden índice de patrón 0..15)
      lambda_v: np.ndarray(16,) -> longitudes de onda de las bandas virtuales corregidas (nm)
    """
    root = ET.parse(xml_path).getroot()
    # Busca la matriz correcta por nombre y tipo
    target = None
    for cm in root.findall(".//correction_matrix"):
        name = cm.findtext("name", default="")
        ctype = cm.findtext("type", default="")
        if name == "hsi_reflectance" and ctype == "reflectance":
            target = cm
            break
    if target is None:
        raise RuntimeError("No se encontró 'hsi_reflectance' (type='reflectance') en el XML.")

    C_rows, lambda_v = [], []
    for vb in target.findall(".//virtual_bands/virtual_band"):
        wl = float(vb.findtext("wavelength_nm"))
        lambda_v.append(wl)
        coeff = vb.find("coefficients")
        vals = [float(x) for x in coeff.get("values").split()]
        if len(vals) != 16:
            raise ValueError("Cada virtual_band debe tener 16 coeficientes.")
        C_rows.append(vals)

    C = np.asarray(C_rows, dtype=np.float32)       # (16,16)
    lambda_v = np.asarray(lambda_v, dtype=np.float32)  # (16,)
    return C, lambda_v


def select_file():
    b2nd_file = easygui.fileopenbox('Select .b2nd file', default=default)
    if b2nd_file is None:
        st.session_state["b2nd_loaded"] = None
        return

    # Cargar medición
    b2nd_loaded = blosc2.open(b2nd_file, mode="r")
    loaded_se=st.session_state["b2nd_loaded"] = b2nd_loaded
    st.sidebar.caption(f"Selected file {loaded_se}")
    st.session_state["b2nd_filename"]=os.path.basename(b2nd_file)
    b2nd_folder = os.path.dirname(b2nd_file)

    # --- WHITE ---
    white_path = os.path.join(b2nd_folder, "white.b2nd")
    white_se=st.session_state['white_path'] = white_path
    if os.path.exists(white_path):
        st.sidebar.caption(white_se)
        white_stack = blosc2.open(white_path, mode="r")
        median_white = np.median(white_stack, axis=0)  # (H,W)
        white_per_channel = np.stack(
            [median_white[i::4, j::4] for i in range(4) for j in range(4)],
            axis=0
        ).astype(np.float32)  # (16, H/4, W/4)
        st.session_state["white_median"] = white_per_channel
    else:
        st.warning("white.b2nd no encontrado en la carpeta.")
        st.session_state["white_median"] = None

    # --- DARK ---
    dark_path = os.path.join(b2nd_folder, "dark.b2nd")
    dark_se=st.session_state["dark_path"] = dark_path
    if os.path.exists(dark_path):
        st.sidebar.caption(dark_se)
        dark_stack = blosc2.open(dark_path, mode="r")
        median_dark = np.median(dark_stack, axis=0)  # (H,W)
        dark_per_channel = np.stack(
            [median_dark[i::4, j::4] for i in range(4) for j in range(4)],
            axis=0
        ).astype(np.float32)  # (16, H/4, W/4)
        st.session_state["dark_median"] = dark_per_channel
    else:
        st.warning("dark.b2nd no encontrado.")
        st.session_state["dark_median"] = None



def demosaic_and_save(b2nd, dark_img, white_img, xml_path, out_dir,filename):
    prefix=f"reflectance_{filename}"
    # 1) Matriz de corrección (reflectancia) y λ virtuales
    C, lambda_v = load_correction_matrix(xml_path)  # (16,16), (16,)

    # 2) Salida (carpeta con mismo nombre base)
    N, H, W = len(b2nd), *b2nd[0].shape
    h4, w4 = H // 4, W // 4
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_name = f"{prefix}_{ts}"
    run_dir = os.path.join(out_dir, base_name)
    os.makedirs(run_dir, exist_ok=True)  # <-- crea la carpeta

    npy_path        = os.path.join(run_dir, f"{base_name}.npy")
    meta_json_path  = os.path.join(run_dir, f"{base_name}.meta.json")
    meta_csv_path   = os.path.join(run_dir, f"{base_name}.meta.csv")

    all_refl = open_memmap(npy_path, mode="w+", dtype=np.float32, shape=(N, 16, h4, w4))

    # 3) Validación de shapes
    if dark_img.shape != (16, h4, w4) or white_img.shape != (16, h4, w4):
        raise ValueError(f"dark/white shape mismatch. Esperado (16,{h4},{w4}), "
                         f"dark={dark_img.shape}, white={white_img.shape}")

    bar = st.progress(0.0)

    for i in range(N):
        raw_frame = b2nd[i]  # (H,W)

        # Demosaic 4x4 → (16,h4,w4) en orden de índice de patrón 0..15
        bands_16 = np.stack([raw_frame[r::4, c::4] for r in range(4) for c in range(4)], axis=0).astype(np.float32)

        # Reflectancia por píxel
        with np.errstate(divide='ignore', invalid='ignore'):
            refl = (bands_16 - dark_img) / (white_img - dark_img + 1e-6)

        # Corrección espectral (virtual bands)
        corr = np.tensordot(C, refl, axes=(1, 0))  # (16,h4,w4)

        # Limpieza y guardado
        corr = np.where(np.isfinite(corr), corr, 0.0).astype(np.float32)
        all_refl[i] = corr
        bar.progress((i + 1) / N)

    all_refl.flush()
    st.success(f"Reflectancia corregida guardada en: {npy_path}")
    st.caption("Bandas virtuales (nm): " + ", ".join(f"{x:.1f}" for x in lambda_v))

    # 4) Metadatos (CSV + JSON) en la misma carpeta
    df_meta = build_meta_dataframe(b2nd)  # puede ser DataFrame vacío si no hay vlmeta
    try:
        if len(df_meta) > 0:
            df_meta.to_csv(meta_csv_path, index=False)
    except Exception as e:
        st.warning(f"No pude guardar CSV de metadatos: {e}")

    static_meta = {
        "file_data": os.path.basename(npy_path),
        "shape_data": [int(N), 16, int(h4), int(w4)],
        "xml_file": os.path.abspath(xml_path),
        "virtual_wavelengths_nm": [float(x) for x in lambda_v.tolist()],
        "correction_matrix_rows": 16,
        "correction_matrix_cols": 16,
        "dark_path": st.session_state.get("dark_path", None),
        "white_path": st.session_state.get("white_path", None),
        "capture_shape_raw": [int(H), int(W)],
        "note": "Reflectance=(I-dark)/(white-dark); spectral correction: I_corr = C @ I."
    }
    # opcional: guardar C completo para reproducibilidad
    static_meta["correction_matrix_C"] = [[float(v) for v in row] for row in C.tolist()]

    try:
        with open(meta_json_path, "w", encoding="utf-8") as f:
            json.dump(static_meta, f, ensure_ascii=False, indent=2)
        st.info(f"Metadatos guardados:\n- {meta_json_path}\n- {meta_csv_path}")
    except Exception as e:
        st.warning(f"No pude guardar JSON de metadatos: {e}")

    # (Opcional) devuelve paths por si quieres enlazar en la UI
    return {"folder": run_dir, "npy": npy_path, "meta_json": meta_json_path, "meta_csv": meta_csv_path}


def build_meta_dataframe(b2nd):
    """
    Extrae metadatos de b2nd.vlmeta a un DataFrame estándar (si existen).
    Columns: Timestamp, Exposure_us, Chip_temperature
    """
    # Por si vlmeta no existe o está vacío:
    if not hasattr(b2nd, "vlmeta") or b2nd.vlmeta is None:
        return pd.DataFrame([])

    meta = b2nd.vlmeta
    def get_meta(key, default=None):
        # keys vienen como bytes, convertimos a str
        bkey = key if isinstance(key, bytes) else key.encode()
        val = meta.get(bkey, default)
        # blosc2 puede devolver arrays tipo bytes; normalizamos a list nativa
        if val is None:
            return []
        if isinstance(val, (bytes, bytearray)):
            try:
                # por si son timestamps serializados, NO los decodamos ciegamente
                return [val.decode()]  # best-effort; si falla, cae al except
            except Exception:
                return [str(val)]
        # numpy → python list
        try:
            import numpy as np
            if isinstance(val, np.ndarray):
                return val.tolist()
        except Exception:
            pass
        # iterable → list
        try:
            return list(val)
        except TypeError:
            return [val]

    ts_list   = get_meta("time_stamp", [])
    exp_list  = get_meta("exposure_us", [])
    chip_list = get_meta("temperature_chip", [])

    # igualamos longitudes al mínimo común
    n = min(len(ts_list), len(exp_list), len(chip_list)) if all([ts_list, exp_list, chip_list]) else 0
    if n == 0:
        return pd.DataFrame([])

    return pd.DataFrame({
        "Timestamp": ts_list[:n],
        "Exposure_us": exp_list[:n],
        "Chip_temperature": chip_list[:n],
    })


    ### Metadata en

def show_mosaic_frame(b2nd_loaded):
    first_frame=b2nd_loaded[0]
    st.image(first_frame, clamp=True,caption="1st frame with mosaic")

def mosaic_pattern(b2nd_loaded):
    frame = b2nd_loaded[0]
    block = frame[:8, :8]  # extract the pattern 8x8 to show more than one complete tile

    fig, ax = plt.subplots(figsize=(3, 3))
    im = ax.imshow(block, cmap='gray', interpolation='nearest')  # evita suavizado
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.grid(True, color='white', linewidth=0.5)

    # Etiquetar cada celda con el canal correspondiente
    for i in range(4):
        for j in range(4):
            canal = i * 4 + j
            ax.text(j, i, f"{canal}", color='red', ha='center', va='center', fontsize=10)

    st.pyplot(fig)
    st.caption('Mosaic pattern 4x4, 16 indexes shown in red, the pattern is repeated every 4x4 (rows x columns)')
#Calculations

def front_end():
    # Inicializa claves
    for key, default_val in [
        ("b2nd_selected", False),
        ("demosaic_done", False),
        ("b2nd_loaded", None),
        ("white_median", None),
        ("dark_median", None),
        ("logs", []),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default_val

    col1, col2, col3 = st.columns([1,1,0.5])
    st.sidebar.subheader("Selection area")

    # Selección .b2nd + white/dark
    if st.sidebar.button("Select .b2nd file to decompress"):
        select_file()
        b2nd_loaded = st.session_state.get("b2nd_loaded")
        white_med   = st.session_state.get("white_median")
        dark_med    = st.session_state.get("dark_median")


        if b2nd_loaded is not None and white_med is not None and dark_med is not None:

            st.session_state.b2nd_selected = True
            info_txt = getattr(b2nd_loaded, "info", None)
            st.session_state.logs.append(f"[{datetime.now():%H:%M:%S}] Loaded: {info_txt if info_txt else 'SChunk'}")
            st.session_state.logs.append("-"*40)
        else:
            st.sidebar.error("Missing: .b2nd and/or white/dark references")

    if st.session_state.b2nd_selected:
        b2nd = st.session_state.b2nd_loaded
        filename=st.session_state.get("b2nd_filename")

        with col1:
            show_mosaic_frame(b2nd)
            st.session_state.logs.append("First frame shown")
        with col2:
            mosaic_pattern(b2nd)

            st.session_state.logs.append("Mosaic pattern shown")

        # Calibration and export
        with col1:
            if st.button("Demosaic & Calibrate"):
                b2nd  = st.session_state.get("b2nd_loaded")
                white = st.session_state.get("white_median")
                dark  = st.session_state.get("dark_median")
                if b2nd is None or white is None or dark is None:
                    st.error("Faltan datos: carga .b2nd + white + dark.")
                else:
                    try:
                        demosaic_and_save(b2nd, dark, white,xml_path,default,filename)
                    except Exception as e:
                        st.exception(e)

        with col3:
            st.subheader("Logs")
            st.markdown("\n".join(st.session_state.logs[-200:]))



# ─── Ejecutar ──────────────────────────────────────────────
front_end()



