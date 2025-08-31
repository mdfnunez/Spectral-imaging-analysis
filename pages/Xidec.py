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

# Paths (Change paths to local PC if needed)
global default
default="/home/alonso/Desktop/" #change the default path accordingly
global xml_path
xml_path="ximea files/CMV2K-SSM4x4-460_600-15.7.20.6.xml"
#####

def select_file():
    b2nd_file = easygui.fileopenbox('Select .b2nd file', default=default)
    if b2nd_file is not None:

        # Open .b2nd in lazy mode
        b2nd_loaded = blosc2.open(b2nd_file, mode="r")
        st.session_state["b2nd_loaded"] = b2nd_loaded
        st.sidebar.caption(f"Selected file {b2nd_loaded}")

        # White reference
        b2nd_folder = os.path.dirname(b2nd_file)
        white_path=os.path.join(b2nd_folder,"white.b2nd")
        st.session_state['white_path']=white_path
        if white_path:
            st.sidebar.caption(st.session_state["white_path"])


            ###Open white file with lazy mode
            white_stack = blosc2.open(white_path, mode="r")

            #Get the median by calculating with 100 frames into single one with median per pixel
            median_mosaic_white = np.median(white_stack, axis=0)  # (H,W)
            
            ### Separate channels 
            
            white_per_channel = np.stack(
                [median_mosaic_white[i::4, j::4] for i in range(4) for j in range(4)],
                axis=0
            ).astype(np.float32)  # (16, H/4, W/4)
            st.session_state["white_median"] = white_per_channel

        # Dark reference
        dark_path = os.path.join(b2nd_folder,"dark.b2nd")
        st.session_state["dark_path"]=dark_path
        if dark_path:
            st.sidebar.caption(dark_path)
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

def meta_data(b2nd):
        #  Metadata
    N = len(b2nd)
    H, W = b2nd[0].shape
    meta = b2nd.vlmeta
    #st.write(meta.keys())
    ###["exposure_us","acq_nframe","color_filter_array","time_stamp","temperature_chip","temperature_house","temperature_house_back_side","temperature_sensor_board"]
    exposure_meta=meta[b"exposure_us"]
    time_stamp=meta[b"time_stamp"]
    temp_chip=meta[b"temperature_chip"]
    df_meta=pd.DataFrame({"Timestamp":time_stamp,"Exposure":exposure_meta,"Chip temperature":temp_chip})
    st.session_state["df_meta"]=df_meta
    st.dataframe(st.session_state.df_meta)
    csv = st.session_state.df_meta.to_csv(index=False).encode("utf-8")
    date=datetime.now()
    st.download_button(
        label="📥 Download metadata (CSV)",
        data=csv,
        file_name=f"metadata_{date}.csv",
        mime="text/csv"
    )

    ### Metadata ends
#Calculations

def demosaic_and_save(b2nd, dark_vec, white_vec):
    def calibration_data():
        #XML
        tree = ET.parse(xml_path)
        root = tree.getroot()
        #Find bands and sort by index
        bands = root.findall(".//band")
        bands = sorted(bands, key=lambda b: int(b.get("index", 0)))

        #Get de average response per band
        resp_means = []
        for band in bands:
            values_str = band.find("response").attrib["values"]
            # Convertir a lista de floats
            values = [float(v) for v in values_str.split()]
            # Get average values
            mean_value = np.mean(values)
            resp_means.append(mean_value)
        #Change to float
        resp_scalar_idx = np.array(resp_means, dtype=np.float32)

        #Show dataframe with scalar average values
        st.session_state["resp_escalar_idx"]=resp_scalar_idx

        #Get wavelengths from each band
        wavelengths = []
        for band in bands:
            wl_text = band.find("peaks/peak/wavelength_nm").text
            wl_value = float(wl_text)
            wavelengths.append(wl_value)


        wavelengths_idx = np.array(wavelengths, dtype=np.float32)
        st.session_state["wavelengths_idx"]=wavelengths_idx
        
        ###Dataframe showing physical bands and sensor response
        dat_resp_wave=pd.DataFrame({"Physical band":range(16),"Average sensor response":st.session_state["resp_escalar_idx"],"Wavelenghts":st.session_state["wavelengths_idx"]})
        st.session_state["dat_resp_wave"]=dat_resp_wave
        st.dataframe(st.session_state["dat_resp_wave"],hide_index=True)
        st.caption('The multispectral camera has 16 physical bands, each defined by an optical filter and sensor sensitivity curve. The wavelength corresponds to the central peak of each filter, while the response scalar represents the average responsivity of the sensor in that band. Raw pixel values are normalized by this response to equalize sensitivity across bands and allow accurate spectral comparison.')

        return resp_scalar_idx,wavelengths_idx,

    resp_scalar_idx, wavelengths_idx = calibration_data()
    resp = resp_scalar_idx[:, None, None].astype(np.float32)  # (16,1,1)

    #Reorganize bands, will be used after reflectance calculations
    sort_idx = np.argsort(wavelengths_idx)     

    
    # Datos de ejemplo
    N, H, W = len(b2nd), *b2nd[0].shape

    # Crear timestamp (ej: 2025-08-31_14-32-10)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Construir path con timestamp
    out_path = os.path.join(default, f"reflectance_{timestamp}.npy")

    # Crear memmap
    all_refl = open_memmap(out_path, mode="w+", dtype=np.float32, shape=(N, 16, H//4, W//4))

    #Calculations and progress bar
    bar = st.progress(0.0)

    for i in range(N):
        raw_frame = b2nd[i]

        # Demosaic 4x4, 16 bands
        bands_16 = []
        for r in range(4):
            for c in range(4):
                bands_16.append(raw_frame[r::4, c::4])
        bands_16 = np.stack(bands_16, axis=0).astype(np.float32)   # (16, H/4, W/4)

        # Radiometric normalization (dark_vec/white_vec shape=(16,1,1))
        with np.errstate(divide='ignore', invalid='ignore'):
            norm = (bands_16 - dark_vec) / (white_vec - dark_vec + 1e-6)

        # Correcteed reflectance with sensor resposivity
        refl_phys = norm / (resp + 1e-12)
        
        #Reorganize bands in wavelenghts order
        to_save = refl_phys if sort_idx is None else refl_phys[sort_idx]

        # Limpieza numérica y escritura directa a disco (memmap)
        to_save = np.where(np.isfinite(to_save), to_save, 0.0).astype(np.float32)
        all_refl[i] = to_save

        # Actualiza progreso
        bar.progress((i + 1) / N)

    # Sincroniza a disco (opcional pero recomendado)
    all_refl.flush()

    st.success(f"Reflectance calculated and saved in: {default}")



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

    # Preview mosaic pattern
    if st.session_state.b2nd_selected:
        b2nd = st.session_state.b2nd_loaded
        with col1:
            show_mosaic_frame(b2nd)
            st.session_state.logs.append("First frame shown")
        with col2:
            mosaic_pattern(b2nd)
            st.subheader("Metadata")
            meta_data(b2nd)

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
                        demosaic_and_save(b2nd, dark_vec=dark, white_vec=white)
                    except Exception as e:
                        st.exception(e)

        with col3:
            st.subheader("Logs")
            st.markdown("\n".join(st.session_state.logs[-200:]))



# ─── Ejecutar ──────────────────────────────────────────────
front_end()



