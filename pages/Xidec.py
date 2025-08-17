import streamlit as st
import blosc2
import easygui
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from datetime import datetime
from tifffile import imwrite
from numpy.lib.format import open_memmap  # 👈 clave para .npy sin RAM
import os, json

#Header
st.set_page_config('Xidec',layout="wide")
st.title('Xidec')
st.caption('Software for decompression of .b2nd files from the Xilens program')

# Paths
global default
default="/home/alonso/Desktop/"
global xml_path
xml_path="ximea files/CMV2K-SSM4x4-460_600-15.7.20.6.xml"

#Functions
st.caption('Logs')
logs=st.empty()

logs.write("inicial")
def select_file():
    b2nd_file=easygui.fileopenbox('Select .b2nd file',default=default)
    if b2nd_file is not None:
        # carga .b2nd
        b2nd_loaded = blosc2.open(b2nd_file, mode="r")
        st.session_state["b2nd_loaded"] = b2nd_loaded
        st.sidebar.success(f"Selected file {b2nd_file}")

        # white reference
        white_path = easygui.fileopenbox('Select white .b2nd', default=default)
        white_stack = blosc2.open(white_path, mode="r")

        # Median from the white stack and gives a single frame H,W with the median for each pixel
        median_mosaic_white = np.median(white_stack, axis=0)  # shape (H, W)

        # 2) Extraer por canal TODO el mapa espacial (sin colapsar):
        white_per_channel = np.stack(
            [median_mosaic_white[i::4, j::4]
            for i in range(4)
            for j in range(4)],
            axis=0
        ).astype(np.float32)  # shape (16, H/4, W/4)

        st.session_state["white_median"] = white_per_channel


     # dark reference
        dark_path = easygui.fileopenbox('Select dark .b2nd', default=default)
        dark_stack = blosc2.open(dark_path, mode="r")

        # 1) Mediana temporal: un solo “mosaic frame”
        median_mosaic_dark = np.median(dark_stack, axis=0)  # shape (H, W)

        # 2) Extraer TODO el mapa espacial de cada canal (sin colapsar):
        dark_per_channel = np.stack(
            [median_mosaic_dark[i::4, j::4]
            for i in range(4)
            for j in range(4)],
            axis=0
        ).astype(np.float32)  # shape (16, H/4, W/4)

        st.session_state["dark_median"] = dark_per_channel

       

    else:
        st.session_state["b2nd_loaded"] = None

    # ahora devolvemos todo lo que hemos calculado
    return (
        st.session_state.get("b2nd_loaded"),
        st.session_state.get("white_median"),
        st.session_state.get("dark_median"),
    )

def show_mosaic_frame(b2nd_loaded):
    first_frame=b2nd_loaded[0]
    st.image(first_frame, clamp=True,caption="1st frame with mosaic")

def mosaic_pattern(b2nd_loaded):
    frame = b2nd_loaded[0]
    block = frame[:8, :8]  # Extrae el patrón 4×4 original

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

def demosaic_and_save(b2nd, xml_path, dark_vec, white_vec,
                      white_dark_order="index"):  # "index" o "lambda"

    os.makedirs(os.path.join(default, "tiffs"), exist_ok=True)
    tiff_dir = os.path.join(default, "tiffs")

    # --- 1) XML: responsividad y λ en ORDEN DE ÍNDICE ---
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bands = sorted(root.findall(".//band"), key=lambda b: int(b.get("index", 0)))
    logs.write("Hola")
    resp_scalar_idx = np.array([
        np.mean([float(v) for v in b.find("response").attrib["values"].split()])
        for b in bands
    ], dtype=np.float32)

    wavelengths_idx = np.array(
        [float(b.find("peaks/peak/wavelength_nm").text) for b in bands],
        dtype=np.float32
    )
    # Permutación índice->λ y su inversa λ->índice
    sort_idx = np.argsort(wavelengths_idx)              # idx -> lambda-order
    inv_idx = np.empty_like(sort_idx)                   # lambda-order -> idx
    inv_idx[sort_idx] = np.arange(len(sort_idx))

    wls_sorted = wavelengths_idx[sort_idx]

    resp = resp_scalar_idx[:, None, None]  # broadcast (16,1,1)

    # --- 3) Metadata y archivos de salida ---
    N, H, W = b2nd.shape
    meta = b2nd.schunk.vlmeta
    # Asegurar strings
    raw_ts = meta[b'time_stamp']
    timestamps = [
        t.decode() if isinstance(t, (bytes, bytearray)) else str(t)
        for t in raw_ts
    ]
    keys = [b'time_stamp', b'exposure_us', b'temperature_chip']

    first_ts = timestamps[0].replace("-", "").replace("_", "")
    npy_path   = os.path.join(default, f"{first_ts}_reflectance.npy")
    ts_npy_path= os.path.join(default, f"{first_ts}_timestamps.npy")

    # Memmap destino (N, 16, H/4, W/4)
    all_refl = open_memmap(npy_path, mode="w+", dtype=np.float32,
                           shape=(N, 16, H // 4, W // 4))
    ts_array = np.empty(N, dtype=f"<U{max(len(t) for t in timestamps)}")

    # --- 4) Procesar frame por frame (calibrar en ORDEN DE ÍNDICE) ---
    for idx in range(N):
        frame = b2nd[idx]

        # Separar canales en ORDEN DE ÍNDICE
        chans = [frame[i::4, j::4] for i in range(4) for j in range(4)]
        chan = np.stack(chans, axis=0).astype(np.float32)

        with np.errstate(divide='ignore', invalid='ignore'):
            norm = (chan - dark_vec) / (white_vec - dark_vec + 1e-6)
        refl_idx = norm / resp  # aún en ORDEN DE ÍNDICE

        # Reordenar por λ SOLO para guardar/salidas
        refl_sorted = refl_idx[sort_idx]

        # Guardar en memmap
        all_refl[idx] = refl_sorted
        ts_array[idx] = timestamps[idx]

        # (Opcional) Guardar TIFF de 16 canales (orden por λ)
        r = refl_sorted.copy()
        r -= r.min()
        r /= (r.max() + 1e-6)
        r_u16 = (r * 65535).astype(np.uint16)

        ts_str = timestamps[idx].replace("-", "").replace("_", "")
        tiff_path = os.path.join(tiff_dir, f"{ts_str}.tiff")
        description = json.dumps({k.decode(): meta[k][idx] for k in keys})
        imwrite(tiff_path, r_u16, dtype=np.uint16,
                photometric="minisblack", metadata=None, description=description)

    # --- 5) Guardar timestamps y sidecar con info de orden ---
    np.save(ts_npy_path, ts_array)
    # Muy útil: guardar sidecar JSON con sort_idx y wls
    sidecar = {
        "order": "lambda",                     # el .npy está por λ
        "sort_idx": sort_idx.tolist(),         # índice->λ
        "wavelengths_sorted_nm": [float(x) for x in wls_sorted]
    }
    with open(os.path.join(default, f"{first_ts}_reflectance.meta.json"), "w") as f:
        json.dump(sidecar, f, indent=2)

    return npy_path, ts_npy_path, tiff_dir


def front_end():
    # ─── Inicializar session_state ──────────────────────────
    for key, default in [
        ("b2nd_selected", False),
        ("demosaic_done",  False),
        ("b2nd_file",      None),
        ("b2nd_loaded",    None),
        ("white_median",   None),
        ("dark_median",    None),
        ("channels",       None),
        ("metadata",       None),
        ("logs",           []),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ─── Layout ──────────────────────────────────────────────
    col1, col2, col3 = st.columns([1,1,0.5])
    st.sidebar.subheader("Selection area")

    # ─── Botón para seleccionar .b2nd ────────────────────────
    if st.sidebar.button("Select .b2nd file to decompress"):
        # Ejecuta tu función de selección y cálculo de white/dark
        b2nd_loaded, white_med, dark_med = select_file()
        if b2nd_loaded is not None:
            st.session_state.b2nd_selected = True
            st.session_state.b2nd_loaded   = b2nd_loaded
            st.session_state.white_median  = white_med
            st.session_state.dark_median   = dark_med
            st.session_state.demosaic_done = False

            # Log
            st.session_state.logs.append(f"[{datetime.now():%H:%M:%S}] Loaded: {b2nd_loaded.info}")
            st.session_state.logs.append("-"*40)
        else:
            st.sidebar.error("No .b2nd selected")

    # ─── Si ya se cargó .b2nd, muestro mosaico y tabla ─────
    if st.session_state.b2nd_selected:
        b2nd = st.session_state.b2nd_loaded
        with col1:
            show_mosaic_frame(b2nd)
            st.session_state.logs.append("First frame shown")
        with col2:
            mosaic_pattern(b2nd)
            st.session_state.logs.append("Mosaic pattern shown")

          

        # ─── Botón de demosaic + calibración ──────────────────
        with col1:
            if st.button("Demosaic & Calibrate"):
                # Tu función que hace demosaic, calibración y guardado de TIFF + NPZ
                demosaic_and_save(
                    b2nd,
                    xml_path="ximea files/CMV2K-SSM4x4-460_600-15.7.20.6.xml",
                    dark_vec=st.session_state.dark_median,
                    white_vec=st.session_state.white_median,
                                                  )
        with col3:
            st.subheader("Logs")
            st.markdown("**Physical wavelengths obtained from xml file**")






# ─── Ejecutar ──────────────────────────────────────────────
front_end()



