# app.py
import streamlit as st
import os
import subprocess
import numpy as np
from PIL import Image
import tifffile
st.subheader("Spectral Image Processing App")

with st.expander("Select images", expanded=False):

    # Select folder of spectral images
    col1,col2,col3= st.columns(3)
    with col1:
        if st.button("Select folder of spectral images"):
            folder_path = subprocess.check_output([
                "python3", "-c",
                "import tkinter as tk; from tkinter import filedialog; root = tk.Tk(); root.withdraw(); print(filedialog.askdirectory(initialdir='/home/alonso/Desktop'))"
            ]).decode("utf-8").strip()
            st.session_state['folder_path'] = folder_path
    with col2:
        # Select black reference image
        if st.button("Select black reference image"):
            black_ref_path = subprocess.check_output([
                "python3", "-c",
                "import tkinter as tk; from tkinter import filedialog; root = tk.Tk(); root.withdraw(); print(filedialog.askopenfilename(filetypes=[('Image Files', '*.tif *.tiff *.png *.jpg')]))"
            ]).decode("utf-8").strip()
            st.session_state['black_ref_path'] = black_ref_path
    with col3:
        # Select white reference image
        if st.button("Select white reference image"):
            white_ref_path = subprocess.check_output([
                "python3", "-c",
                "import tkinter as tk; from tkinter import filedialog; root = tk.Tk(); root.withdraw(); print(filedialog.askopenfilename(filetypes=[('Image Files', '*.tif *.tiff *.png *.jpg')]))"
            ]).decode("utf-8").strip()
            st.session_state['white_ref_path'] = white_ref_path

    folder_path = st.session_state.get('folder_path', '')
    black_ref_path = st.session_state.get('black_ref_path', '')
    white_ref_path = st.session_state.get('white_ref_path', '')

    if folder_path and os.path.isdir(folder_path):
        files = os.listdir(folder_path)
        image_files = [f for f in files if f.endswith(('.tif', '.tiff', '.png', '.jpg'))]
        with col1:
            st.caption(f"Folder: {folder_path}")
            st.success("Found image files")

        with col2:
             st.caption(f"Black reference image: {black_ref_path}")
        with col3:
             st.caption(f"White refernce image: {white_ref_path}")
    elif folder_path:
        with col1:
             st.warning("Introduce una ruta válida.")

    if black_ref_path and os.path.isfile(black_ref_path):
        with col2:
             st.success(f"Black reference image selected: {os.path.basename(black_ref_path)}")
    elif black_ref_path:
        with col2:
             st.warning("Black reference image path is not valid.")

    if white_ref_path and os.path.isfile(white_ref_path):
        with col3:
             st.success(f"White reference image selected: {os.path.basename(white_ref_path)}")
    elif white_ref_path:
        with col3:
            st.warning("White reference image path is not valid.")
sol1,sol2 = st.columns(2)
with sol1:
    with st.expander('Monospectral image processing', expanded=False):
        st.write("Monospectral image processing options will appear here.")
with sol2:
    with st.expander('Multispectral image', expanded=False):

        if folder_path and os.path.isdir(folder_path):
            image_files = [
                f for f in os.listdir(folder_path)
                if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg')) and not f.startswith('._')
            ]
            if image_files:
                img_path = os.path.join(folder_path, image_files[0])
                try:
                    # 1) Leer la imagen
                    if img_path.lower().endswith(('.tif', '.tiff')):
                        img_array = tifffile.imread(img_path)
                    else:
                        img = Image.open(img_path)
                        img_array = np.array(img)

                    # 2) Detectar dónde está la dimensión de canales
                    shape = img_array.shape
                    st.write(f"Original shape: {shape}")
                    if img_array.ndim == 3 and shape[0] == 16:
                        chan_axis = 0
                    elif img_array.ndim == 3 and shape[-1] == 16:
                        chan_axis = -1
                    else:
                        st.error(f"No veo 16 canales en {os.path.basename(img_path)}. ¡Tienes {shape}!")
                        st.stop()

                    # 3) Slider para escoger canal
                    num_ch = shape[chan_axis]
                    ch = st.slider("Selecciona canal (0–15)", 0, num_ch - 1, 0)

                    # 4) Extraerlo
                    if chan_axis == 0:
                        channel_img = img_array[ch, :, :]
                    else:
                        channel_img = img_array[:, :, ch]

                    # 1) Mírate min/max o percentiles para entender tu canal
                    vmin, vmax = np.percentile(channel_img, (1, 99))
                    st.write(f"Percentil 1%={vmin:.2f}, 99%={vmax:.2f}")

                    # 2) Normaliza entre 0 y 1 usando esos percentiles
                    channel_norm = (channel_img.astype(np.float32) - vmin) / (vmax - vmin)
                    channel_norm = np.clip(channel_norm, 0, 1)

                    # 3) Muestra la imagen normalizada
                    st.image(channel_norm, caption=f"Canal {ch} normalizado", clamp=True)
                    
                except Exception as e:
                    st.error(f"No se pudo abrir la imagen: {os.path.basename(img_path)}. Error: {e}")
            else:
                st.info("No se encontraron imágenes en la carpeta seleccionada.")
    with st.expander("Multispectral image processing options", expanded=False):
        st.write("Multispectral image processing options will appear here.")
    with st.expander("Calibración y Reflectancia", expanded=False):
        """
        Expander para calibrar imágenes multiespectrales y calcular reflectancia.
        Utiliza imágenes dark y white de referencia para calibrar todas las imágenes de muestra en folder_path.
        Guarda los resultados en un subfolder Reflectance_Images.
        """

        # Parámetros de usuario para visualización (no afectan el cálculo ni el guardado)
        st.markdown("**Opciones de visualización (no afectan el guardado):**")
        p_low = st.slider("Percentil bajo para visualización", 0.0, 10.0, 1.0, step=0.5)
        p_high = st.slider("Percentil alto para visualización", 90.0, 100.0, 99.0, step=0.5)
        epsilon = 1e-6  # Para evitar divisiones por cero

        # 1. Definir carpeta de salida
        output_dir = os.path.join(folder_path, "Reflectance_Images")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            st.info(f"Carpeta de salida creada: {output_dir}")
        else:
            st.info(f"Carpeta de salida: {output_dir}")

        # 2. Cargar y apilar imágenes dark y white, unificando a (C,H,W)
        def load_and_stack(file_list):
            """Carga imágenes TIFF y las convierte a (C,H,W)."""
            stack = []
            for f in file_list:
                arr = tifffile.imread(f)
                if arr.ndim == 3:
                    # Detectar si (C,H,W) o (H,W,C)
                    if arr.shape[0] == 16:
                        arr_chw = arr
                    elif arr.shape[-1] == 16:
                        arr_chw = np.moveaxis(arr, -1, 0)
                    else:
                        st.warning(f"Archivo {os.path.basename(f)} no tiene 16 canales.")
                        continue
                    stack.append(arr_chw)
                else:
                    st.warning(f"Archivo {os.path.basename(f)} no es 3D.")
            return np.stack(stack, axis=0)  # shape (N,C,H,W)

        stack_dark = load_and_stack(dark_files)
        stack_white = load_and_stack(white_files)

        # 3. Calcular promedios dark y white
        dark_mean = np.mean(stack_dark, axis=0)    # (C,H,W)
        white_mean = np.mean(stack_white, axis=0)  # (C,H,W)

        # 4. Procesar imágenes de muestra
        sample_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(('.tif', '.tiff')) and not f.startswith('._')
        ]
        n_samples = len(sample_files)
        progress_bar = st.progress(0, text="Procesando imágenes de reflectancia...")

        reflectance_shapes = []
        for idx, sample_path in enumerate(sample_files):
            try:
                # a) Leer y convertir a (C,H,W)
                arr = tifffile.imread(sample_path)
                if arr.ndim == 3:
                    if arr.shape[0] == 16:
                        sample_chw = arr
                    elif arr.shape[-1] == 16:
                        sample_chw = np.moveaxis(arr, -1, 0)
                    else:
                        st.warning(f"{os.path.basename(sample_path)} no tiene 16 canales.")
                        continue
                else:
                    st.warning(f"{os.path.basename(sample_path)} no es 3D.")
                    continue

                # b) Calcular reflectancia
                denom = (white_mean - dark_mean).clip(min=epsilon)
                rho = (sample_chw - dark_mean) / denom  # (C,H,W)
                reflectance_shapes.append(rho.shape)

                # c) Visualización opcional
                vmin, vmax = np.percentile(rho, [p_low, p_high])
                rho_vis = np.clip((rho - vmin) / (vmax - vmin), 0, 1)
                if idx == 0:
                    st.image(rho_vis[0], caption=f"Canal 0 reflectancia (visualizado)", clamp=True)

                # d) Guardar reflectancia como TIFF (sin clipeo)
                out_name = f"Reflectance_{os.path.splitext(os.path.basename(sample_path))[0]}.tif"
                out_path = os.path.join(output_dir, out_name)
                tifffile.imwrite(out_path, rho.astype(np.float32))

            except Exception as e:
                st.error(f"Error procesando {os.path.basename(sample_path)}: {e}")

            progress_bar.progress((idx + 1) / n_samples, text=f"Procesando {idx+1}/{n_samples}")
        # 5. Resumen final
        st.success("Procesamiento completado.")
        st.write(f"**Imágenes procesadas:** {n_samples}")
        if reflectance_shapes:
            st.write(f"**Forma de reflectancia:** {reflectance_shapes[0]} (C,H,W)")
        st.write(f"**Carpeta de salida:** `{output_dir}`")
        