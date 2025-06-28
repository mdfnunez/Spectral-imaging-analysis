import streamlit as st
import os
import subprocess
import numpy as np
from PIL import Image
import tifffile
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import shutil
import streamlit as st
import os
import subprocess
import numpy as np
from PIL import Image
import tifffile
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift
import shutil
import tempfile
import imageio.v2 as imageio  # FIX: usar imageio.v2 para compatibilidad con video


tab1,tab2=st.tabs(['Image processing','Perfusion maps'])
with tab1:
    def load_image_stack(folder_path, expected_channels=16):
        image_files = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.tif', '.tiff'))
        ])

        images = []
        for fname in image_files:
            img_path = os.path.join(folder_path, fname)
            try:
                img_array = tifffile.imread(img_path)
                if img_array.ndim == 3 and img_array.shape[0] == expected_channels:
                    img_array = np.moveaxis(img_array, 0, -1)
                elif img_array.ndim == 3 and img_array.shape[-1] == expected_channels:
                    pass
                else:
                    continue
                images.append((fname, img_array.astype(np.float32)))
            except Exception as e:
                st.warning(f"No se pudo cargar {fname}: {e}")
        return images

    st.subheader("Spectral Image Processing App")

    with st.expander("Select images", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Select folder of spectral images"):
                folder_path = subprocess.check_output([
                    "python3", "-c",
                    "import tkinter as tk; from tkinter import filedialog; root = tk.Tk(); root.withdraw(); print(filedialog.askdirectory(initialdir='/home/alonso/Desktop'))"
                ]).decode("utf-8").strip()
                st.session_state['folder_path'] = folder_path

        with col2:
            if st.button("Select folder with DARK images"):
                dark_ref_folder = subprocess.check_output([
                    "python3", "-c",
                    "import tkinter as tk; from tkinter import filedialog; root = tk.Tk(); root.withdraw(); print(filedialog.askdirectory(initialdir='/home/alonso/Desktop'))"
                ]).decode("utf-8").strip()
                st.session_state['dark_ref_folder'] = dark_ref_folder
                st.success("Dark reference folder selected correctly.")

        with col3:
            if st.button("Select folder with WHITE images"):
                white_ref_folder = subprocess.check_output([
                    "python3", "-c",
                    "import tkinter as tk; from tkinter import filedialog; root = tk.Tk(); root.withdraw(); print(filedialog.askdirectory(initialdir='/home/alonso/Desktop'))"
                ]).decode("utf-8").strip()
                st.session_state['white_ref_folder'] = white_ref_folder
                st.success("White reference folder selected correctly.")

        folder_path = st.session_state.get('folder_path', '')
        dark_ref_folder = st.session_state.get('dark_ref_folder', '')
        white_ref_folder = st.session_state.get('white_ref_folder', '')

        if folder_path and os.path.isdir(folder_path):
            files = os.listdir(folder_path)
            image_files = [f for f in files if f.endswith(('.tif', '.tiff', '.png', '.jpg'))]
            with col1:
                st.caption(f"Folder: {folder_path}")
                st.success("Found image files")

        if dark_ref_folder:
            with col2:
                st.caption(f"Dark reference folder: {dark_ref_folder}")

        if white_ref_folder:
            with col3:
                st.caption(f"White reference folder: {white_ref_folder}")

    with st.expander("🔍 White and Dark Analysis", expanded=False):
        run_analysis = st.button("Iniciar análisis de white/dark")

        if run_analysis:
            if os.path.isdir(white_ref_folder) and os.path.isdir(dark_ref_folder):
                white_stack = [img for _, img in load_image_stack(white_ref_folder)]
                dark_stack = [img for _, img in load_image_stack(dark_ref_folder)]

                if white_stack and dark_stack:
                    white_stack = np.stack(white_stack, axis=0)
                    dark_stack = np.stack(dark_stack, axis=0)

                    st.success(f"{white_stack.shape[0]} white y {dark_stack.shape[0]} dark images cargadas.")

                    white_clipped = np.clip(
                        white_stack,
                        np.percentile(white_stack, 5, axis=(0, 1, 2), keepdims=True),
                        np.percentile(white_stack, 95, axis=(0, 1, 2), keepdims=True)
                    )

                    dark_clipped = np.clip(
                        dark_stack,
                        np.percentile(dark_stack, 5, axis=(0, 1, 2), keepdims=True),
                        np.percentile(dark_stack, 95, axis=(0, 1, 2), keepdims=True)
                    )

                    white_mean = np.mean(white_clipped, axis=0)
                    dark_mean = np.mean(dark_clipped, axis=0)

                    ch = st.slider("Selecciona canal a visualizar (0–15)", 0, 15, 0)

                    st.write("Visualización de canal seleccionado:")
                    col_w, col_d = st.columns(2)

                    with col_w:
                        fig1, ax1 = plt.subplots()
                        im1 = ax1.imshow(white_mean[:, :, ch], cmap='gray')
                        ax1.set_title(f"White Mean - Canal {ch}")
                        plt.colorbar(im1, ax=ax1)
                        st.pyplot(fig1)

                    with col_d:
                        fig2, ax2 = plt.subplots()
                        im2 = ax2.imshow(dark_mean[:, :, ch], cmap='gray')
                        ax2.set_title(f"Dark Mean - Canal {ch}")
                        plt.colorbar(im2, ax=ax2)
                        st.pyplot(fig2)

                    np.save("white_mean.npy", white_mean)
                    np.save("dark_mean.npy", dark_mean)
                    st.success("Promedios guardados como 'white_mean.npy' y 'dark_mean.npy'")
                else:
                    st.error("Error al cargar las imágenes de referencia.")
            else:
                st.warning("Selecciona carpetas válidas para referencias white y dark.")

    with st.expander("🧪 Procesar reflectancia de múltiples carpetas"):
        st.markdown("Selecciona las carpetas que contienen las imágenes por condición. Se generará la reflectancia y se guardará en subcarpetas.")
        if st.button("Seleccionar carpetas para convertir a reflectancia"):
            folders = subprocess.check_output([
                "python3", "-c",
                "import tkinter as tk; from tkinter import filedialog; root = tk.Tk(); root.withdraw(); import pathlib; paths = filedialog.askdirectory(mustexist=True, initialdir='/home/alonso/Desktop'); print(paths)"
            ]).decode("utf-8").strip().split(":::")
            st.session_state['reflectance_folders'] = folders

        folders = st.session_state.get('reflectance_folders', [])
        if folders:
            white = np.load("white_mean.npy")
            dark = np.load("dark_mean.npy")
            white_smooth = gaussian_filter(white, sigma=(10, 10, 0))

            for folder in folders:
                st.write(f"Procesando carpeta: {folder}")
                images = load_image_stack(folder)
                output_dir = os.path.join(folder, "reflectance")
                os.makedirs(output_dir, exist_ok=True)

                for fname, img_array in images:
                    try:
                        reflectance = (img_array - dark) / (white_smooth - dark)
                        reflectance = np.clip(reflectance, 0, 1)
                        out_path = os.path.join(output_dir, fname.replace(".tif", "_refl.npy").replace(".tiff", "_refl.npy"))
                        np.save(out_path, reflectance)
                    except Exception as e:
                        st.warning(f"Error procesando {fname}: {e}")

                st.success(f"Carpeta '{folder}' procesada y guardada en '{output_dir}'")
    with st.expander("📊 Visualizar mapas de reflectancia (.npy)"):
        uploaded_file = st.file_uploader("Selecciona un archivo .npy", type=["npy"])

        if uploaded_file is not None:
            reflectance = np.load(uploaded_file)

            if reflectance.ndim == 3 and reflectance.shape[-1] == 16:
                ch = st.slider("Selecciona canal (0–15)", 0, 15, 0)

                vmin, vmax = np.percentile(reflectance[:, :, ch], (1, 99))
                norm_img = np.clip((reflectance[:, :, ch] - vmin) / (vmax - vmin), 0, 1)

                fig, ax = plt.subplots()
                im = ax.imshow(norm_img, cmap='gray')
                ax.set_title(f"Reflectancia - Canal {ch}")
                plt.colorbar(im, ax=ax)
                st.pyplot(fig)
            else:
                st.error("El archivo no parece tener 16 canales en la última dimensión.")
with tab2:
    import streamlit as st
    import os
    import subprocess
    import numpy as np
    from PIL import Image
    import tifffile
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    from skimage.registration import phase_cross_correlation
    from scipy.ndimage import fourier_shift
    import shutil
    import tempfile
    import imageio.v2 as imageio  # FIX: usar imageio.v2 para compatibilidad con video


    def load_image_stack(folder_path, expected_channels=16):
        image_files = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.tif', '.tiff'))
        ])

        images = []
        for fname in image_files:
            img_path = os.path.join(folder_path, fname)
            try:
                img_array = tifffile.imread(img_path)
                if img_array.ndim == 3 and img_array.shape[0] == expected_channels:
                    img_array = np.moveaxis(img_array, 0, -1)
                elif img_array.ndim == 3 and img_array.shape[-1] == expected_channels:
                    pass
                else:
                    continue
                images.append((fname, img_array.astype(np.float32)))
            except Exception as e:
                st.warning(f"No se pudo cargar {fname}: {e}")
        return images


    def stabilize_stack(stack):
        ref = stack[0]
        stabilized = [ref]
        for img in stack[1:]:
            shift, error, diffphase = phase_cross_correlation(ref, img)
            shifted = np.fft.ifftn(fourier_shift(np.fft.fftn(img), shift)).real
            stabilized.append(shifted)
        return np.array(stabilized)


    def normalize_image(img, mode="percentile"):
        if mode == "percentile":
            vmin, vmax = np.percentile(img, (5, 95))
        elif mode == "std":
            mean = np.mean(img)
            std = np.std(img)
            vmin = mean - 1.5 * std
            vmax = mean + 1.5 * std
        elif mode == "log":
            img = np.log1p(img)
            vmin, vmax = np.percentile(img, (5, 95))
        else:
            vmin, vmax = 0, 1
        norm_img = np.clip((img - vmin) / (vmax - vmin + 1e-6), 0, 1)
        return norm_img


    st.subheader("Spectral Image Processing App")

    # --- EXPANDER: Visualizar mapas de oxigenación ---
    with st.expander("🪸 Generar mapa de oxigenación (HbO2/Hb)"):
        perf_folder = ""
        if st.button("Seleccionar carpeta con archivos .npy de reflectancia"):
            perf_folder = subprocess.check_output([
                "python3", "-c",
                "import tkinter as tk; from tkinter import filedialog; root = tk.Tk(); root.withdraw(); print(filedialog.askdirectory(initialdir='/home/alonso/Desktop'))"
            ]).decode("utf-8").strip()
            st.session_state['perf_folder'] = perf_folder

        perf_folder = st.session_state.get('perf_folder', '')
        if perf_folder and os.path.isdir(perf_folder):
            npy_files = sorted([f for f in os.listdir(perf_folder) if f.endswith(".npy")])
            if npy_files:
                st.markdown("Elige los canales asociados a **HbO2** y **Hb** según la sensibilidad espectral de tu cámara:")
                ch_hbo2 = st.number_input("Canal para HbO2", min_value=0, max_value=15, value=5)
                ch_hb = st.number_input("Canal para Hb", min_value=0, max_value=15, value=10)

                stack_hbo2 = []
                stack_hb = []
                file_names = []

                for file in npy_files:
                    try:
                        data = np.load(os.path.join(perf_folder, file))
                        if data.ndim == 3 and data.shape[2] > max(ch_hbo2, ch_hb):
                            refl_hbo2 = data[:, :, ch_hbo2]
                            refl_hb = data[:, :, ch_hb]
                            stack_hbo2.append(1 - refl_hbo2)
                            stack_hb.append(1 - refl_hb)
                            file_names.append(file)
                    except Exception as e:
                        st.warning(f"No se pudo leer {file}: {e}")

                if len(stack_hbo2) > 0 and len(stack_hb) > 0:
                    stack_hbo2 = np.stack(stack_hbo2, axis=0)
                    stack_hb = np.stack(stack_hb, axis=0)

                    if st.checkbox("Aplicar estabilización de movimiento", value=True):
                        with st.spinner("Estabilizando imágenes..."):
                            stack_hbo2 = stabilize_stack(stack_hbo2)
                            stack_hb = stabilize_stack(stack_hb)

                    mean_hbo2 = np.mean(stack_hbo2, axis=0)
                    mean_hb = np.mean(stack_hb, axis=0)

                    epsilon = 1e-6
                    ox_map = mean_hbo2 / (mean_hbo2 + mean_hb + epsilon)

                    fig, ax = plt.subplots()
                    im = ax.imshow(ox_map, cmap='coolwarm', vmin=0, vmax=1)
                    ax.set_title("Mapa de oxigenación estimada HbO2 / (HbO2 + Hb)")
                    plt.colorbar(im, ax=ax)
                    st.pyplot(fig)

                    ox_time_series = stack_hbo2 / (stack_hbo2 + stack_hb + epsilon)
                    st.session_state['ox_time_series'] = ox_time_series
                    st.session_state['file_names'] = file_names

                    export_mode = st.selectbox("Modo de normalización para exportar", ["lineal", "percentile", "std", "log"])

                    if st.button("Exportar imágenes como PNG"):
                        export_folder = os.path.join(perf_folder, "oxigenacion_export")
                        os.makedirs(export_folder, exist_ok=True)
                        for i, (img, fname) in enumerate(zip(ox_time_series, file_names)):
                            img_norm = normalize_image(img, mode=export_mode)
                            img_uint8 = (img_norm * 255).astype(np.uint8)
                            out_path = os.path.join(export_folder, os.path.splitext(fname)[0] + "_ox.png")
                            Image.fromarray(img_uint8).save(out_path)
                        st.success(f"Exportadas {len(file_names)} imágenes a: {export_folder}")

                    if st.button("Generar y mostrar video MP4"):
                        with st.spinner("Generando video..."):
                            temp_dir = tempfile.mkdtemp()
                            frame_paths = []
                            for i, (img, fname) in enumerate(zip(ox_time_series, file_names)):
                                img_norm = normalize_image(img, mode=export_mode)
                                img_uint8 = (img_norm * 255).astype(np.uint8)
                                out_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
                                Image.fromarray(img_uint8).save(out_path)
                                frame_paths.append(out_path)
                            video_path = os.path.join(temp_dir, "oxigenacion_video.mp4")
                            with imageio.get_writer(video_path, fps=10) as writer:
                                for frame in frame_paths:
                                    writer.append_data(imageio.imread(frame))
                            st.video(video_path)

                else:
                    st.warning("No se pudieron generar stacks válidos para los canales seleccionados.")
            else:
                st.warning("No se encontraron archivos .npy en la carpeta seleccionada.")

    # --- EXPANDER: Ver evolución temporal de oxigenación ---
    with st.expander("🎥 Ver evolución temporal (todas las imágenes)"):
        ox_time_series = st.session_state.get('ox_time_series', None)
        file_names = st.session_state.get('file_names', [])

        if ox_time_series is not None and len(file_names) == ox_time_series.shape[0]:
            idx = st.slider("Imagen en el tiempo", 0, ox_time_series.shape[0] - 1, 0)
            norm_mode = st.selectbox("Modo de visualización", ["lineal", "percentile", "std", "log"])
            img = normalize_image(ox_time_series[idx], mode=norm_mode)
            fig, ax = plt.subplots()
            im = ax.imshow(img, cmap='coolwarm', vmin=0, vmax=1)
            ax.set_title(f"Oxigenación estimada - Frame {idx} ({file_names[idx]})")
            plt.colorbar(im, ax=ax)
            st.pyplot(fig)
        else:
            st.info("Primero debes generar el mapa de oxigenación para activar esta vista temporal.")