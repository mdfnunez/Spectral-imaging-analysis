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
    if st.sidebar.button('Add reflectance .npz file',key=9815):
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
            st.session_state.logs.append(f"Stack information: {data.files}")
            timestamps=data["timestamps"]
            st.session_state["timestamps"]=timestamps

    # Mostrar la ruta aunque no se presione el botón (persistente)
    if "path" in st.session_state:
        st.sidebar.caption(st.session_state['path'])


# ———————————————————————————————
# 2) Original TIFF folder
# ———————————————————————————————
    if st.sidebar.button("Add folder with .tiff files",key=3456):
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

        return (
            st.session_state.get('reflectance_stack', None),
            st.session_state.get('original_stack', None),
            st.session_state.get("timestamps",None)
            
        )

def coefficients_show():
    # Cargar archivo Excel
    coefficients = pd.read_excel("ximea files/HbO2_Hb_spectrum_full.xlsx")
    # Mostrar tabla dentro de un expander
    with st.expander('Molar extinction coefficients', expanded=False):
        st.dataframe(coefficients,hide_index=True)


#load variables
try:
    reflectance_stack,original_stack,timestamps=folder_path_acquisition()
except:
    st.info('Load both .npz and tiff files')

# Ends load variables



#Band selection 
def band_selection():
    # --- 1) Load wavelengts from XML ---
    tree = ET.parse("ximea files/CMV2K-SSM4x4-460_600-15.7.20.6.xml")
    root = tree.getroot()
    bands = sorted(root.findall(".//band"), key=lambda b: int(b.get("index", 0)))
    wavelengths = np.array([float(b.find("peaks/peak/wavelength_nm").text) for b in bands], dtype=np.float32)
    
    #Arranged from lower to maximum wavelengths
    wavelengths=np.sort(wavelengths)

    #load absorption coefficients HbO2 and Hb
    df_spec = pd.read_excel("ximea files/HbO2_Hb_spectrum_full.xlsx")

    # --- 3) Band selection interface ---
    with col2:
        band1 = st.slider("Select band for HbO₂", min_value=0, max_value=15, value=2, step=1, key="band1_spec")
    with col2:
        band2 = st.slider("Select band for Hb", min_value=0, max_value=15, value=5, step=1, key="band2_spec")

    #Selection of wavelength in wavelengths with the sliders
    λ1 = int(wavelengths[band1])
    λ2 = int(wavelengths[band2])


    # --- 5) Select wavelengths withing 400 and 700 nm ---
    df_zoom = df_spec[(df_spec["lambda"] >= 450) & (df_spec["lambda"] <= 650)]

    # --- 6) Graph with molar coefficients and selection of bands ---
    with col2:
        with st.expander('Molar extinction graph and selected bands', expanded=True):
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(df_zoom["lambda"], df_zoom["Hb02"], label="ε HbO₂", color="crimson")
            ax.plot(df_zoom["lambda"], df_zoom["Hb"], label="ε Hb", color="royalblue")
            ax.fill_between(df_zoom["lambda"], df_zoom["Hb02"], df_zoom["Hb"], color='gray', alpha=0.4)

            ax.axvline(λ1, color="crimson", linestyle="--", lw=2, label=f"Band HbO₂ ~ {λ1} nm")
            ax.axvline(λ2, color="royalblue", linestyle="--", lw=2, label=f"Band Hb ~ {λ2} nm")

            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Coefficient ε (normalized)")
            ax.set_title("Molar extinction spectrum")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)


    # Buscar coincidencia cercana a λ1
    match1 = df_spec[np.isclose(df_spec["lambda"], λ1, atol=2)]

    # Buscar coincidencia cercana a λ2
    match2 = df_spec[np.isclose(df_spec["lambda"], λ2, atol=2)]
   
    # Si todo bien, accede
    row1 = match1.iloc[0]
    row2 = match2.iloc[0]

    #acquire absorbance coefficients
    Hb02_λ1 = row1['Hb02']
    Hb_λ1   = row1['Hb']
    Hb02_λ2 = row2['Hb02']
    Hb_λ2   = row2['Hb']
    data=pd.DataFrame([{"HbO2-1":Hb02_λ1,"HbO2-2":Hb02_λ2, "Hb1": Hb_λ1, "Hb2":Hb_λ2}])
    st.dataframe(data,hide_index=True)
   
    E = np.array([
        [Hb02_λ1, Hb_λ1],
        [Hb02_λ2, Hb_λ2]
    ])
    print("Matriz E:\n", E)
    print("Determinante:", np.linalg.det(E))

    # --- 8) Matriz de extinción y determinante ---
    E = np.array([
        [Hb02_λ1, Hb_λ1],
        [Hb02_λ2, Hb_λ2]
    ])
    det = np.linalg.det(E)
    condition_number = np.linalg.cond(E)


    if abs(det) < 0.01:
        st.warning(f"⚠️ Determinante muy bajo ({det:.4f}). Riesgo de inestabilidad numérica.")
    elif abs(det) < 0.05:
        st.info(f"ℹ️ Determinante moderado ({det:.4f}). Aceptable, pero con precaución.")
    else:
        st.success(f"✅ Determinant adequate: {det:.4f}")
    
    if condition_number > 1000:
        st.warning(f"⚠️ Alta inestabilidad numérica. Número de condición: {condition_number:.2f}")
    elif condition_number > 100:
        st.info(f"ℹ️ Condición moderada. Número de condición: {condition_number:.2f}")
    else:
        st.success(f"✅ Buena condición numérica: {condition_number:.2f}")



    return λ1, λ2, Hb02_λ1, Hb_λ1, Hb02_λ2, Hb_λ2, E, band1, band2



#Extended beer lambert calculations
def beer_lambert_calculations(λ1, λ2, Hb02_λ1, Hb_λ1, Hb02_λ2, Hb_λ2, E, band1, band2,reflectance_stack,original_stack,timestamps):
    #Calculations based on como se compara este metodo con un beer lambert simple que no condidera pathleng ni g ni DPF from Neil T. Clancy
    #Tissue hemoglobin index (THI)
    #StO2 index

    #AShow metadata (timestamps)
    data_sto2=pd.DataFrame(timestamps)
    st.session_state.logs.append(f" Timestamps:{data_sto2}")


    #Normalize to 0-1
    reflectance_norm = reflectance_stack / np.max(reflectance_stack)
    reflectance_norm = np.clip(reflectance_norm, 1e-6, 1.0)
    st.write(reflectance_norm[0,0,:,:])

    # Negative natural logaritm to obtain absorbance
    absorbance_stack=-np.log(reflectance_norm)
    st.write(absorbance_stack[0,0,:,:])

    with col1:
        with st.expander('Modified Beer-Lambert calculations',expanded=True):
            st.empty()
    with col2:
            st.empty()








col1,col2,col3=st.columns([1,1,0.5])
with col1:
    λ1, λ2, Hb02_λ1, Hb_λ1, Hb02_λ2, Hb_λ2, E, band1, band2=band_selection()
    try:
        beer_lambert_calculations(λ1, λ2, Hb02_λ1, Hb_λ1, Hb02_λ2, Hb_λ2, E, band1, band2,reflectance_stack,original_stack,timestamps)
    except:
        st.warning('Upload .npz and .tiff files')
with col2:
    st.empty()
with col3:
    coefficients_show()
    #log column
    st.subheader('Logs')
    if "logs" not in st.session_state:
        st.session_state.logs=[]
    st.write(st.session_state.get("logs"))



