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
default="/home/alonso/Desktop/" #change the default path accordingly
global xml_path
xml_path="ximea files/CMV2K-SSM4x4-460_600-15.7.20.6.xml"


def select_file():
    b2nd_file = easygui.fileopenbox('Select .b2nd file', default=default)
    if b2nd_file is not None:
        # Carga .b2nd principal
        b2nd_loaded = blosc2.open(b2nd_file, mode="r")
        st.session_state["b2nd_loaded"] = b2nd_loaded
        st.sidebar.success(f"Selected file {b2nd_file}")

        # White reference
        white_path = easygui.fileopenbox('Select white .b2nd', default=default)
        if white_path:
            white_stack = blosc2.open(white_path, mode="r")
            median_mosaic_white = np.median(white_stack, axis=0)  # (H,W)
            white_per_channel = np.stack(
                [median_mosaic_white[i::4, j::4] for i in range(4) for j in range(4)],
                axis=0
            ).astype(np.float32)  # (16, H/4, W/4)
            st.session_state["white_median"] = white_per_channel

        # Dark reference
        dark_path = easygui.fileopenbox('Select dark .b2nd', default=default)
        if dark_path:
            dark_stack = blosc2.open(dark_path, mode="r")
            median_mosaic_dark = np.median(dark_stack, axis=0)  # (H,W)
            dark_per_channel = np.stack(
                [median_mosaic_dark[i::4, j::4] for i in range(4) for j in range(4)],
                axis=0
            ).astype(np.float32)  # (16, H/4, W/4)
            st.session_state["dark_median"] = dark_per_channel
    else:
        st.session_state["b2nd_loaded"] = None

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
def demosaic_and_save(b2nd, dark_vec, white_vec):
    import os, json
    import numpy as np
    import xml.etree.ElementTree as ET
    from tifffile import imwrite
    from numpy.lib.format import open_memmap

    # Carpetas
    base_tiff_dir = os.path.join(default, "tiffs")
    orig16_dir    = os.path.join(base_tiff_dir, "orig16")       # float32 "fiel"
    prev16_dir    = os.path.join(base_tiff_dir, "preview16")    # uint16 Z-stack
    rgb_dir       = os.path.join(base_tiff_dir, "rgb")          # uint8 RGB
    os.makedirs(orig16_dir, exist_ok=True)
    os.makedirs(prev16_dir, exist_ok=True)
    os.makedirs(rgb_dir, exist_ok=True)

    # 1) XML
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bands = sorted(root.findall(".//band"), key=lambda b: int(b.get("index", 0)))
    resp_scalar_idx = np.array([
        np.mean([float(v) for v in bands[k].find("response").attrib["values"].split()])
        for k in range(len(bands))
    ], dtype=np.float32)
    wavelengths_idx = np.array(
        [float(bands[k].find("peaks/peak/wavelength_nm").text) for k in range(len(bands))],
        dtype=np.float32
    )
    sort_idx  = np.argsort(wavelengths_idx)
    wls_sorted= wavelengths_idx[sort_idx]
    resp      = resp_scalar_idx[:, None, None]  # (16,1,1)

    # 2) Meta
    N = len(b2nd)
    H, W = b2nd[0].shape
    meta = b2nd.vlmeta
    raw_ts = meta.get(b'time_stamp', [f"frame_{i:06d}" for i in range(N)])
    timestamps = [t.decode() if isinstance(t,(bytes,bytearray)) else str(t) for t in raw_ts]
    first_ts = timestamps[0].replace("-", "").replace("_", "")

    npy_path    = os.path.join(default, f"{first_ts}_reflectance.npy")
    ts_npy_path = os.path.join(default, f"{first_ts}_timestamps.npy")
    all_refl = open_memmap(npy_path, mode="w+", dtype=np.float32, shape=(N,16,H//4,W//4))
    ts_array = np.empty(N, dtype=f"<U{max(len(t) for t in timestamps)}")

    # 2.5) Límites fijos para previews (p2–p98) con muestreo
    sample = min(N, 64)
    acc = []
    for i in range(sample):
        f = b2nd[i]
        chans = [f[a::4, b::4] for a in range(4) for b in range(4)]
        chan = np.stack(chans, axis=0).astype(np.float32)
        with np.errstate(divide='ignore', invalid='ignore'):
            norm = (chan - dark_vec) / (white_vec - dark_vec + 1e-6)
        refl_idx = norm / (resp + 1e-12)
        acc.append(refl_idx[sort_idx])
    stack_small = np.stack(acc, axis=0) if acc else np.zeros((1,16,H//4,W//4), np.float32)
    stack_small = np.where(np.isfinite(stack_small), stack_small, np.nan)
    per_band_lo = np.nanpercentile(stack_small, 2,  axis=(0,2,3)).astype(np.float32)  # (16,)
    per_band_hi = np.nanpercentile(stack_small, 98, axis=(0,2,3)).astype(np.float32) # (16,)
    per_band_hi = np.maximum(per_band_hi, per_band_lo + 1e-6)

    # Índices RGB (R=650, G=550, B=460 nm aprox.)
    idx_R = int(np.argmin(np.abs(wls_sorted - 650.0)))
    idx_G = int(np.argmin(np.abs(wls_sorted - 550.0)))
    idx_B = int(np.argmin(np.abs(wls_sorted - 460.0)))

    # 3) Loop
    meta_keys = [b'time_stamp', b'exposure_us', b'temperature_chip']
    for idx in range(N):
        frame = b2nd[idx]
        chans = [frame[i::4, j::4] for i in range(4) for j in range(4)]
        chan = np.stack(chans, axis=0).astype(np.float32)

        with np.errstate(divide='ignore', invalid='ignore'):
            norm = (chan - dark_vec) / (white_vec - dark_vec + 1e-6)
        refl_idx = norm / (resp + 1e-12)

        # Orden λ y limpieza
        refl_sorted = refl_idx[sort_idx].astype(np.float32, copy=False)
        refl_sorted = np.where(np.isfinite(refl_sorted), refl_sorted, 0.0)

        # Guarda maestro .npy (fiel)
        all_refl[idx] = refl_sorted
        ts_array[idx] = timestamps[idx]

        # Paths + desc
        ts_str = timestamps[idx].replace("-", "").replace("_", "")
        path_orig = os.path.join(orig16_dir, f"{ts_str}_orig16.tiff")
        path_prev = os.path.join(prev16_dir, f"{ts_str}_preview16.tiff")
        path_rgb  = os.path.join(rgb_dir,   f"{ts_str}_rgb.tiff")
        desc = {}
        for k in meta_keys:
            if k in meta:
                v = meta[k][idx]
                desc[k.decode()] = v.decode() if isinstance(v,(bytes,bytearray)) else str(v)

        # A) ORIG16: float32 sin tocar (para Fiji/napari)
        imwrite(
            path_orig,
            np.ascontiguousarray(refl_sorted),   # (Z=16,Y,X)
            dtype=np.float32,
            imagej=True,
            metadata={"axes":"ZYX"},
            photometric="minisblack",
            description=json.dumps(desc),
            bigtiff=True
        )

        # B) PREVIEW16: uint16 con límites fijos por banda (compatibilidad universal)
        r = (refl_sorted - per_band_lo[:,None,None]) / (per_band_hi[:,None,None] - per_band_lo[:,None,None])
        r = np.clip(r, 0.0, 1.0)
        r_u16 = (r * 65535).astype(np.uint16)
        imwrite(
            path_prev,
            r_u16,                           # (Z=16,Y,X)
            dtype=np.uint16,
            imagej=True,
            metadata={"axes":"ZYX"},
            photometric="minisblack",
            description=json.dumps(desc)
        )

        # C) RGB: uint8 usando los mismos límites de esas 3 bandas
        def _scale_band(img, lo, hi):
            return np.clip((img - lo) / (hi - lo), 0.0, 1.0)

        R = _scale_band(refl_sorted[idx_R], per_band_lo[idx_R], per_band_hi[idx_R])
        G = _scale_band(refl_sorted[idx_G], per_band_lo[idx_G], per_band_hi[idx_G])
        B = _scale_band(refl_sorted[idx_B], per_band_lo[idx_B], per_band_hi[idx_B])
        rgb8 = (np.stack([R,G,B], axis=-1) * 255.0).astype(np.uint8)
        imwrite(path_rgb, rgb8, photometric="rgb")

    # 4) Guardar timestamps + sidecar
    np.save(ts_npy_path, ts_array)
    sidecar = {
        "order":"lambda",
        "sort_idx":[int(x) for x in sort_idx.tolist()],
        "wavelengths_sorted_nm":[float(x) for x in wls_sorted],
        "preview_percentiles":{"low_p2":per_band_lo.tolist(),"high_p98":per_band_hi.tolist()},
        "rgb_indices":{"R":int(idx_R),"G":int(idx_G),"B":int(idx_B)},
        "note":"Mediciones desde *_reflectance.npy. Previews: preview16 uint16(ZYX) y rgb uint8 con límites fijos."
    }
    with open(os.path.join(default, f"{first_ts}_reflectance.meta.json"), "w") as f:
        json.dump(sidecar, f, indent=2)


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

    # Vista previa mosaico/patrón
    if st.session_state.b2nd_selected:
        b2nd = st.session_state.b2nd_loaded
        with col1:
            show_mosaic_frame(b2nd)
            st.session_state.logs.append("First frame shown")
        with col2:
            mosaic_pattern(b2nd)
            st.session_state.logs.append("Mosaic pattern shown")

        # Calibrar y exportar
        with col1:
            if st.button("Demosaic & Calibrate"):
                b2nd  = st.session_state.get("b2nd_loaded")
                white = st.session_state.get("white_median")
                dark  = st.session_state.get("dark_median")
                if b2nd is None or white is None or dark is None:
                    st.error("Faltan datos: carga .b2nd + white + dark.")
                else:
                    try:
                        demosaic_and_save(b2nd, dark_vec=dark, white_vec=white)
                        st.success("Calibración terminada. Archivos escritos en /tiffs y .npy/.json en la carpeta por defecto.")
                    except Exception as e:
                        st.exception(e)

        with col3:
            st.subheader("Logs")
            st.markdown("\n".join(st.session_state.logs[-200:]))



# ─── Ejecutar ──────────────────────────────────────────────
front_end()



