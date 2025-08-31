import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import easygui
import os, glob
from tifffile import imread
import tifffile as tiff
import tkinter as tk
from tkinter import simpledialog, Button
from PIL import Image, ImageTk
import cv2
from bisect import bisect_right

global default
default="/home/alonso/Desktop/"
st.set_page_config(layout="wide")

def header_agsantos():
    iol1, iol2 = st.columns([4, 1])
    with iol1:
        st.title("Spectral analysis of Ximea cameras")
        st.caption('AG-Santos Neurovascular research Laboratory')
    with iol2:
        st.image('images/agsantos.png', width=130)
    st.markdown("_______________________")
header_agsantos()


def folder_path_acquisition():
    # ———————————————————————————————
    # 1) Reflectance NPY (memmap)
    # ———————————————————————————————
    if st.sidebar.button('Add proccesed stack .npy file', key="processed stack"):
        path = easygui.fileopenbox(
            msg="Select a .npy file with processed stacks",
            default="/home/alonso/Desktop/",
            filetypes=["*.npy"]
        )
        if path:
            data = np.load(path, allow_pickle=False, mmap_mode="r")
            processed_stack = data  # memmap-like

            if processed_stack.dtype == np.float64:
                st.warning("Processed en float64. Convertiré por bloques a float32 si procesas todo.")

            st.session_state['processed_stack'] = processed_stack

            if "logs" not in st.session_state:
                st.session_state.logs = []
            st.session_state.logs.append(
                f"✅ Proccesed stack (memmap) shape: {processed_stack.shape}, dtype: {processed_stack.dtype}"
            )

    tiff_files=st.sidebar.button('Add tiff files')
    if tiff_files:
        tiff_dir_path=easygui.diropenbox("Add folder path for tiff files",default=default)
        st.session_state["tiff_path"]=tiff_dir_path
    metadata = st.sidebar.file_uploader("📂 Upload CSV with metadata", type=".csv")
    # Si el usuario subió algo, lo guardamos en session_state
    if metadata is not None:
        st.session_state["metadata"] = metadata
    # Retornamos lo que se subió (sirve en este rerun)
    return (
        st.session_state.get('processed_stack', None),
        st.session_state.get("metadata",None),
        st.session_state.get("tiff_path")
        
    )
#load variables
try:
    processed_stack,metadata,tiff_path=folder_path_acquisition()
except:
    st.info('Load both .npz and metadata (.csv)')
# Ends load variables

def show_timestamps_panel(timestamps):
    if timestamps is None:
        st.info("No se han cargado timestamps todavía.")
        return
    metadat_df=pd.read_csv(metadata)
    with st.expander('Metadata dataframe'):
        st.dataframe(metadat_df)



def _norm01(a):
    a = np.asarray(a, np.float32)
    a[~np.isfinite(a)] = np.nan
    vmin, vmax = np.nanpercentile(a, [2, 98])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.nanmin(a)), float(np.nanmax(a))
    out = (a - vmin) / (vmax - vmin + 1e-12)
    return np.clip(out, 0, 1), vmin, vmax

def _load_tiff_list(tiff_path):
    # ordena por nombre; asume un TIFF por frame
    files = sorted(glob.glob(os.path.join(tiff_path, "*.tif")))
    if not files:
        raise FileNotFoundError(f"No encontré TIFFs en: {tiff_path}")
    return files

def visualizer(processed_stack, tiff_path):
    # UI
    col1, col2 = st.columns([0.55, 1.45])
    with col1:
        T, H, W = processed_stack.shape
        t = st.slider("Frame (t)", 0, T-1, 0, 1)
        alpha = st.slider("Transparencia máscara (α)", 0.0, 1.0, 0.6, 0.05)
        cmap = st.selectbox("Colormap", ["coolwarm", "seismic", "bwr", "magma", "viridis"])
        use_thresh = st.checkbox("Usar umbral en máscara", value=False)
        if use_thresh:
            thr_mode = st.radio("Modo umbral", ["Percentil (%)", "Valor absoluto"], horizontal=True)
            if thr_mode == "Percentil (%)":
                thr_p = st.slider("Percentil", 0, 100, 90, 1)
            else:
                thr_val = st.number_input("Valor (absoluto)", value=0.0, format="%.6f")
        edge_only = st.checkbox("Mostrar solo máscara (ocultar fondo fuera de ROI)", value=False)

    # Cargar fondo TIFF correspondiente
    files = _load_tiff_list(tiff_path)
    if t >= len(files):
        st.error(f"No hay TIFF para t={t}. Encontrados: {len(files)}")
        return
    bg = imread(files[t])  # (H,W) uint16/uint8/float...
    # Ajustar tamaños si difieren
    if bg.shape != (H, W):
        st.warning(f"Tamaño distinto: TIFF {bg.shape} vs stack {(H,W)}. Intentaré recortar/coincidir.")
        H0, W0 = bg.shape
        Hm, Wm = min(H0, H), min(W0, W)
        bg = bg[:Hm, :Wm]
        proc = processed_stack[t, :Hm, :Wm].astype(np.float32)
    else:
        proc = processed_stack[t].astype(np.float32)

    # Normalizaciones
    bg01, _, _ = _norm01(bg)
    proc01, vmin, vmax = _norm01(proc)

    # Umbral opcional para mostrar máscara parcial
    if use_thresh:
        if thr_mode == "Percentil (%)":
            thr_val = np.nanpercentile(proc, thr_p)
        # si es absoluto ya lo tenemos en thr_val del sidebar
        mask = proc >= thr_val
        # donde no hay máscara, ponemos NaN para que no se pinte
        mask_alpha = alpha * mask.astype(np.float32)
        # si quieres ocultar fondo fuera de ROI
        if edge_only:
            bg01 = bg01 * mask.astype(np.float32)
    else:
        mask_alpha = alpha

    # Plot
    fig, ax = plt.subplots()
    ax.imshow(bg01, cmap="gray", interpolation="nearest")
    im = ax.imshow(proc, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", alpha=mask_alpha)
    ax.axis("off")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.6)
    cbar.set_label("Δ (unidades relativas)", rotation=270, labelpad=10, fontsize=7)
    cbar.ax.tick_params(labelsize=7)

    with col2:
        st.pyplot(fig)
        fname = getattr(processed_stack, "filename", "array en memoria")
        st.caption(f"t={t} | rango=({vmin:.4g}–{vmax:.4g}) | fondo: {os.path.basename(files[t])} | stack: {fname}")


col1,col2,col3=st.columns(3)

visualizer(processed_stack,tiff_path)

with col3:
    st.empty()

