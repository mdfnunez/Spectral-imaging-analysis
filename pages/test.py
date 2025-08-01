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

def band_selection():
    coefficients=pd.read_excel("ximea files/HbO2_Hb_spectrum_full.xlsx")
    #show coefficients table
    with st.expander('Molar extinction coefficients',expanded=False):
        st.dataframe(coefficients)








 #GUI   
col1,col2,col3=st.columns([1,1,0.5])
with col1:
    st.empty()
with col2:
    band_selection()
with col3:
    #log column
    st.subheader('Logs')
    if "logs" not in st.session_state:
        st.session_state.logs=[]
    st.write(st.session_state.get("logs"))


#Acqisiton of data files sidebar
reflectance_stack,original_stack,processed_stack=folder_path_acquisition()

