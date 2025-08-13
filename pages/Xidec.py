import streamlit as st
import blosc2
import easygui
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import os
from tifffile import imwrite
from datetime import datetime
import json


st.set_page_config('Xidec',layout="wide")
st.title('Xidec')
st.caption('Decompression of .b2nd files from the Xilens program')

default="/home/alonso/Desktop/"

def select_file():
    b2nd_file=easygui.fileopenbox('Select .b2nd file',default="/home/alonso/Desktop/")
    if b2nd_file is not None:
        # carga .b2nd
        b2nd_loaded = blosc2.open(b2nd_file, mode="r")
        st.session_state["b2nd_loaded"] = b2nd_loaded
        st.sidebar.success(f"Selected file {b2nd_file}")

        # white reference
        white_path = easygui.fileopenbox('Select white .b2nd', default=default)
        white_stack = blosc2.open(white_path, mode="r")

        # 1) Mediana temporal: un solo mosaic frame
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

@st.cache_data
def load_wavelengths():
    xml_path="ximea files/CMV2K-SSM4x4-460_600-15.7.20.6.xml"
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bands = sorted(root.findall(".//band"), key=lambda b: int(b.get("index",0)))
    wls = [float(b.find("peaks/peak/wavelength_nm").text) for b in bands]
   
    return np.array(wls, dtype=np.float32)


def load_responsivity_scalar():
    """
    Lee del XML las curvas de responsividad y las promedia para obtener
    un único scalar por banda.
    Devuelve array shape (n_canales,).
    """
    xml_path="ximea files/CMV2K-SSM4x4-460_600-15.7.20.6.xml"

    tree = ET.parse(xml_path)
    root = tree.getroot()
    bands = sorted(root.findall(".//band"), key=lambda b: int(b.get("index",0)))
    resp_curves = []
    for b in bands:
        vals = b.find("response").attrib.get("values","").split()
        curve = np.array([float(v) for v in vals], dtype=np.float32)
        resp_curves.append(curve)
    # responsividad media por banda:
    return np.array([c.mean() for c in resp_curves], dtype=np.float32)

def false_color_image(channels: np.ndarray,
                      rgb_indices: tuple[int,int,int] = (2, 7, 11)
                     ) -> np.ndarray:
   
    r = channels[rgb_indices[0]]
    g = channels[rgb_indices[1]]
    b = channels[rgb_indices[2]]
    
    def norm(x):
        mn, mx = np.nanmin(x), np.nanmax(x)
        return (x - mn) / (mx - mn + 1e-6)
    
    r_n = norm(r)
    g_n = norm(g)
    b_n = norm(b)
    
    # Stack and convert to uint8
    img = np.dstack((r_n, g_n, b_n))
    img8 = (img * 255).astype(np.uint8)
    return img8

def False_RGB(channels: np.ndarray):
    with st.expander('RGB image'):
        if channels is not None:
            C, H, W = channels.shape
            st.sidebar.subheader("False color")
            # Creamos selectboxes para R, G, B
            r = st.sidebar.selectbox("Red",  list(range(C)), index=2)
            g = st.sidebar.selectbox("Green", list(range(C)), index=7)
            b = st.sidebar.selectbox("Blue",  list(range(C)), index=11)
            
            if st.sidebar.button("Show false color image"):
                img_fc = false_color_image(channels, (r, g, b))
                st.image(img_fc)

wls=load_wavelengths()
xml_path="ximea files/CMV2K-SSM4x4-460_600-15.7.20.6.xml"
responsivity=load_responsivity_scalar()


def demosaic_and_save(b2nd, xml_path, dark_vec, white_vec):
    import os, json
    import numpy as np
    import pandas as pd
    import xml.etree.ElementTree as ET
    import streamlit as st
    from tifffile import imwrite

    from numpy.lib.format import open_memmap  # 👈 clave para .npy sin RAM

    default = "/home/alonso/Desktop/"
    tiff_dir = os.path.join(default, "tiffs")
    os.makedirs(tiff_dir, exist_ok=True)

    # --- 1) Cargar responsividad y longitudes de onda desde XML ---
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bands = sorted(root.findall(".//band"), key=lambda b: int(b.get("index", 0)))
    resp_scalar = np.array([
        np.mean([float(v) for v in b.find("response").attrib["values"].split()])
        for b in bands
    ], dtype=np.float32)

    wavelengths = np.array(
        [float(b.find("peaks/peak/wavelength_nm").text) for b in bands],
        dtype=np.float32
    )
    sort_idx = np.argsort(wavelengths)
    wls_sorted = wavelengths[sort_idx]

    if np.all(np.diff(wls_sorted) >= 0):
        st.success("✅ Canales reordenados correctamente por longitud de onda.")
    else:
        st.error("⚠️ ¡Ocurrió un problema! Las longitudes de onda no están ordenadas.")

    df_check = pd.DataFrame({
        "Nueva posición k": np.arange(len(sort_idx)),
        "Canal original m": sort_idx,
        "λ (nm) ordenada": wls_sorted.round(1)
    })
    st.subheader("Comprobación de reordenamiento espectral")
    st.dataframe(df_check, hide_index=True)

    # --- 2) Preparar datos base ---
    N, H, W = b2nd.shape
    meta = b2nd.schunk.vlmeta
    timestamps = meta[b'time_stamp']
    keys = [b'time_stamp', b'exposure_us', b'temperature_chip']

    meta_list = [{k.decode(): meta[k][i] for k in keys} for i in range(N)]
    st.session_state["meta_list"] = meta_list

    # --- 3) Evitar RAM: open_memmap hacia .npy directamente ---
    first_ts = timestamps[0].replace("-", "").replace("_", "")
    npy_path = os.path.join(default, f"{first_ts}_reflectance.npy")
    ts_npy_path = os.path.join(default, f"{first_ts}_timestamps.npy")

    # crea archivo .npy mapeable, de tamaño final (N, 16, H//4, W//4)
    all_refl = open_memmap(
        npy_path, mode="w+", dtype=np.float32, shape=(N, 16, H // 4, W // 4)
    )

    # timestamps en array de unicode (tamaño suficiente para tus strings)
    ts_array = np.empty(N, dtype=f"<U{max(len(t) for t in timestamps)}")

    # Convertir white/dark a float32 y asegurar forma correcta
    dark_vec = dark_vec.astype(np.float32)
    white_vec = white_vec.astype(np.float32)
    resp = resp_scalar[:, None, None]

    # --- 4) Procesar frame por frame ---
    progress = st.progress(0.0)

    for idx in range(N):
        frame = b2nd[idx]

        # Separar canales del mosaico 4x4
        chans = [frame[i::4, j::4] for i in range(4) for j in range(4)]
        chan = np.stack(chans, axis=0).astype(np.float32)

        # --- Calibración ---
        with np.errstate(divide='ignore', invalid='ignore'):
            norm = (chan - dark_vec) / (white_vec - dark_vec + 1e-6)
        refl = norm / resp

        # --- Reordenar canales ---
        refl_sorted = refl[sort_idx]

        # Guardar en memmap .npy (sin RAM total)
        all_refl[idx] = refl_sorted
        ts_array[idx] = timestamps[idx]

        # --- Guardar TIFF individual (opcional, para inspección) ---
        refl_norm = refl_sorted - refl_sorted.min()
        refl_norm /= (refl_norm.max() + 1e-6)
        refl_to_save = (refl_norm * 65535).astype(np.uint16)

        ts_str = timestamps[idx].replace("-", "").replace("_", "")
        tiff_path = os.path.join(tiff_dir, f"{ts_str}.tiff")
        description = json.dumps({k.decode(): meta[k][idx] for k in keys})

        imwrite(
            tiff_path,
            refl_to_save,
            dtype=np.uint16,
            photometric="minisblack",
            metadata=None,
            description=description
        )

        progress.progress((idx + 1) / N)

    # --- 5) Guardar timestamps en .npy aparte (también mapeable después) ---
    np.save(ts_npy_path, ts_array)

    # --- 6) Log final ---
    st.session_state.logs.append("✅ Archivos guardados correctamente:")
    st.session_state.logs.append(f"📂 TIFF multicanal en: {tiff_dir}")
    st.session_state.logs.append(f"📦 Archivo .npy (reflectance): {npy_path}")
    st.session_state.logs.append(f"🕒 Timestamps .npy: {ts_npy_path}")
    st.session_state.logs.append("📊 Orden espectral aplicado:")
    st.session_state.logs.append(f"Orden: {sort_idx.tolist()}")
    st.session_state.logs.append(f"Wavelengths ordenados (nm): {wls_sorted.round(1).tolist()}")
    st.session_state.logs.append("-" * 40)

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
                    white_vec=st.session_state.white_median                )
        with col3:
            st.subheader("Logs")
            st.markdown("**Physical wavelengths obtained from xml file**")

            # Crea el DataFrame de un plumazo
            df_phys = pd.DataFrame({
                "Canal": np.arange(len(wls)),
                "Wavelength (nm)": wls
            })

            # Muéstralo con Streamlit
            st.dataframe(df_phys, hide_index=True)

                
  



# ─── Ejecutar ──────────────────────────────────────────────
front_end()



