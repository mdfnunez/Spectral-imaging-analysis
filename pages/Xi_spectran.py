import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import easygui
import os
import skimage.exposure
import time
import tifffile as tiff
import tkinter as tk
from tkinter import simpledialog, Button
from PIL import Image, ImageTk
import cv2
from bisect import bisect_right
import xml.etree.ElementTree as ET

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
    # 1) Reflectance NPZ
    # ———————————————————————————————
    if st.sidebar.button('Add reflectance .npz file'):
        path = easygui.fileopenbox(
            msg="Select a .npz file with reflectance stacks",
            default="/home/alonso/Desktop/",
            filetypes=["*.npz"]
        )
        if path:
            data = np.load(path, allow_pickle=True)
            reflectance_stack = data["reflectance"]

            st.session_state['reflectance_stack'] = reflectance_stack
            st.session_state['reflectance_npz_path'] = path
            st.session_state["path"] = path

            # Inicializa logs si no existe
            if "logs" not in st.session_state:
                st.session_state.logs = []
            st.session_state.logs.append(f"✅ Reflectance cargada con shape: {reflectance_stack.shape}")

    # Mostrar la ruta aunque no se presione el botón (persistente)
    if "path" in st.session_state:
        st.sidebar.caption(st.session_state['path'])


# ———————————————————————————————
# 2) Original TIFF folder
# ———————————————————————————————
    if st.sidebar.button("Add folder with .tiff files"):
        folder = easygui.diropenbox(
            msg='Select folder with original .tiff images',
            default="/home/alonso/Desktop"
        )
        if folder:
            imgs = []
            for fn in sorted(os.listdir(folder)):
                if fn.lower().endswith((".tif", ".tiff")) and not fn.startswith("."):
                    try:
                        imgs.append(tiff.imread(os.path.join(folder, fn)))
                    except Exception as e:
                        st.warning(f"Skipping {fn}: {e}")
            if imgs:
                stack = np.stack(imgs, axis=0)
                st.session_state['original_stack'] = stack
                st.session_state['original_folder_path'] = folder
                st.session_state.logs.append(f"✅ Original TIFF stack: {stack.shape}")
            else:
                st.error("⚠️ No valid TIFFs found.")
    if "original_folder_path" in st.session_state:
        st.sidebar.caption(st.session_state["original_folder_path"])

    # ———————————————————————————————
    # 3) Processed NPZ
    # ———————————————————————————————

    if st.sidebar.button("Add processed .npz file"):
        p = easygui.fileopenbox(
            msg="Select a .npz file with processed indices",
            default="/home/alonso/Desktop/",
            filetypes=["*.npz"]
        )
    
        if p:
            data = np.load(p, allow_pickle=True)
            
            if "logs" not in st.session_state:
                st.session_state.logs = []

            st.session_state.logs.append(f"✅ Processed data loaded with shape: {data['arr_0'].shape}")
            st.session_state["p"] = p
            st.session_state['processed_data'] = data

    # Mostramos la ruta si ya está guardada en session_state
    if "p" in st.session_state:
        st.sidebar.caption(st.session_state['p'])

        # ———————————————————————————————
        # 4) Devuelve los tres stacks
        # ———————————————————————————————
        return (
            st.session_state.get('reflectance_stack', None),
            st.session_state.get('original_stack', None),
            st.session_state.get('processed_stack', None),
        )

folder_path_acquisition()

def coefficients_show():
    # Cargar archivo Excel
    coefficients = pd.read_excel("ximea files/HbO2_Hb_spectrum_full.xlsx")

    # Mostrar tabla dentro de un expander
    with st.expander('Molar extinction coefficients', expanded=False):
        st.dataframe(coefficients,hide_index=True)

    # Crear y mostrar gráfico dentro de un expander
    with st.expander('Molar extinction graph', expanded=False):
        # Filtrar rango deseado en lambda
        filtered = coefficients[(coefficients["lambda"] >= 450) & (coefficients["lambda"] <= 650)]

        fig, ax = plt.subplots()
        ax.plot(filtered["lambda"], filtered["Hb02"], label="HbO2", color="red")
        ax.plot(filtered["lambda"], filtered["Hb"], label="Hb", color="blue")
        ax.set_xlabel("Longitud de onda (nm)")
        ax.set_ylabel("Extinción molar")
        ax.set_title("Espectros entre 400–700 nm")
        ax.legend()
        st.pyplot(fig)






 #GUI   


def band_selection():
    # --- 1) Cargar longitudes de onda desde XML ---
    tree = ET.parse("ximea files/CMV2K-SSM4x4-460_600-15.7.20.6.xml")
    root = tree.getroot()
    bands = sorted(root.findall(".//band"), key=lambda b: int(b.get("index", 0)))
    wavelengths = np.array([float(b.find("peaks/peak/wavelength_nm").text) for b in bands], dtype=np.float32)

    with st.expander('Wavelengths according to physical index'):
        st.dataframe(wavelengths)

    # --- 2) Leer espectro molar desde Excel ---
    df_spec = pd.read_excel("ximea files/HbO2_Hb_spectrum_full.xlsx")

    # --- 3) Band selection interface ---
    colb1, colb2 = st.columns(2)
    with colb1:
        band1 = st.number_input("Select band for HbO₂", min_value=0, max_value=15, value=2, step=1, key="band1_spec")
    with colb2:
        band2 = st.number_input("Select band for Hb", min_value=0, max_value=15, value=5, step=1, key="band2_spec")

    λ1 = int(wavelengths[band1])
    st.info(λ1)
    λ2 = int(wavelengths[band2])


    # --- 5) Filtrar entre 400 y 700 nm ---
    df_zoom = df_spec[(df_spec["lambda"] >= 450) & (df_spec["lambda"] <= 650)]

    # --- 6) Graficar espectros y bandas seleccionadas ---
    with st.expander('Molar extinction graph', expanded=False):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df_zoom["lambda"], df_zoom["Hb02"], label="ε HbO₂", color="crimson")
        ax.plot(df_zoom["lambda"], df_zoom["Hb"], label="ε Hb", color="royalblue")
        ax.fill_between(df_zoom["lambda"], df_zoom["Hb02"], df_zoom["Hb"], color='gray', alpha=0.2)

        ax.axvline(λ1, color="crimson", linestyle="--", lw=2, label=f"Band HbO₂ ~ {λ1} nm")
        ax.axvline(λ2, color="royalblue", linestyle="--", lw=2, label=f"Band Hb ~ {λ2} nm")

        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Coefficient ε (normalized)")
        ax.set_title("Molar extinction spectrum")
        ax.grid(True)
        ax.legend()
        st.caption('Coefficients and selected bands')
        st.pyplot(fig)

    # Buscar filas cercanas a λ1 y λ2
    match1 = df_spec[np.isclose(df_spec["lambda"], λ1)]
    match2 = df_spec[np.isclose(df_spec["lambda"], λ2)]

    # Si todo bien, accede
    row1 = match1.iloc[0]
    row2 = match2.iloc[0]
    st.write("📌 Coeficientes seleccionados:")
    st.write(row1)
    st.write(row2)

    Hb02_λ1 = row1['Hb02']
    Hb_λ1   = row1['Hb']
    Hb02_λ2 = row2['Hb02']
    Hb_λ2   = row2['Hb']

    # --- 8) Matriz de extinción y determinante ---
    E = np.array([
        [Hb02_λ1, Hb_λ1],
        [Hb02_λ2, Hb_λ2]
    ])
    det = np.linalg.det(E)

    if abs(det) < 0.01:
        st.warning(f"⚠️ Determinante muy bajo ({det:.4f}). Riesgo de inestabilidad numérica.")
    elif abs(det) < 0.05:
        st.info(f"ℹ️ Determinante moderado ({det:.4f}). Aceptable, pero con precaución.")
    else:
        st.success(f"✅ Determinante adecuado: {det:.4f}")

    st.caption("🔍 El determinante indica si las bandas permiten diferenciar bien entre HbO₂ y Hb. Cuanto mayor, mejor separación espectral.")

    return λ1, λ2, Hb02_λ1, Hb_λ1, Hb02_λ2, Hb_λ2, E, band1, band2

col1,col2,col3=st.columns([1,1,0.5])
with col1:
    band_selection()
with col2:
    coefficients_show()
with col3:
    #log column
    st.subheader('Logs')
    if "logs" not in st.session_state:
        st.session_state.logs=[]
    st.write(st.session_state.get("logs"))


#Acqisiton of data files sidebar
reflectance_stack,original_stack,processed_stack=folder_path_acquisition()

