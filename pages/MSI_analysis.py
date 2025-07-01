import streamlit as st
import os
import subprocess
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw
import tifffile
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from scipy.ndimage import gaussian_filter
from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift
import shutil
import tempfile
import imageio.v2 as imageio  # FIX: usar imageio.v2 para compatibilidad con video
import io
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(layout="wide", page_title="Spectral Image Processing App", page_icon=":microscope:")
tab1,tab2,tab3=st.tabs(['Image processing','Perfusion maps',"ROI analysis"])
with tab1:
    # --- NUEVA FUNCIÓN ROBUSTA ---
    def load_image_stack(folder_path, expected_channels=16, page_main=1):
        """
        Devuelve [(fname, img)] donde img es (H,W,bandas).
        page_main=1 ⇒ primero intenta la página 1 (cubo completo)
        y si no existe usa la 0 (compatibilidad con TIFF antiguos).
        """
        import tifffile, numpy as np, os, streamlit as st

        imgs=[]
        for fname in sorted(f for f in os.listdir(folder_path)
                            if f.lower().endswith(('.tif','.tiff'))):
            try:
                with tifffile.TiffFile(os.path.join(folder_path,fname)) as tif:
                    key = page_main if page_main < len(tif.pages) else 0
                    arr = tif.asarray(key=key)
                if arr.ndim==3 and arr.shape[0]==expected_channels:
                    arr=np.moveaxis(arr,0,-1)          # (H,W,C)
                if arr.ndim==3 and arr.shape[-1]==expected_channels:
                    imgs.append((fname,arr.astype(np.float32)))
                else:
                    st.warning(f"{fname}: canales={arr.shape[-1]} (omitido)")
            except Exception as e:
                st.warning(f"No se pudo cargar {fname}: {e}")
        return imgs
    # --- FIN NUEVA FUNCIÓN ---

    with st.expander("Select images", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Select folder with DARK images"):
                dark_ref_folder = subprocess.check_output([
                    "python3", "-c",
                    "import tkinter as tk; from tkinter import filedialog; root = tk.Tk(); root.withdraw(); print(filedialog.askdirectory(initialdir='/home/alonso/Desktop'))"
                ]).decode("utf-8").strip()
                st.session_state['dark_ref_folder'] = dark_ref_folder
                st.success("Dark reference folder selected correctly.")

        with col2:
            if st.button("Select folder with WHITE images"):
                white_ref_folder = subprocess.check_output([
                    "python3", "-c",
                    "import tkinter as tk; from tkinter import filedialog; root = tk.Tk(); root.withdraw(); print(filedialog.askdirectory(initialdir='/home/alonso/Desktop'))"
                ]).decode("utf-8").strip()
                st.session_state['white_ref_folder'] = white_ref_folder
                st.success("White reference folder selected correctly.")

        dark_ref_folder = st.session_state.get('dark_ref_folder', '')
        white_ref_folder = st.session_state.get('white_ref_folder', '')



        if dark_ref_folder:
            with col1:
                st.caption(f"Dark reference folder: {dark_ref_folder}")

        if white_ref_folder:
            with col2:
                st.caption(f"White reference folder: {white_ref_folder}")

    with st.expander("🔍 White and Dark Analysis", expanded=False):
        run_analysis = st.button("Iniciar análisis de white/dark")

        if run_analysis:
            if os.path.isdir(white_ref_folder) and os.path.isdir(dark_ref_folder):
                white_stack = [img for _, img in load_image_stack(white_ref_folder)]
                dark_stack = [img for _, img in load_image_stack(dark_ref_folder)]

                if white_stack and dark_stack:
                    # --- BLOQUE NUEVO: mediana robusta y sigma adaptativo ---
                    white_stack = np.stack(white_stack,0).astype(np.float32)
                    dark_stack  = np.stack(dark_stack ,0).astype(np.float32)

                    white_mean = np.median(white_stack,axis=0)
                    dark_mean  = np.median(dark_stack ,axis=0)

                    sigma_xy = max(white_mean.shape[:2]) / 50  # ~2 % del FOV
                    white_smooth = gaussian_filter(white_mean, sigma=(sigma_xy, sigma_xy, 0))
                    # --- FIN BLOQUE NUEVO ---

                    ch = st.slider("Selecciona canal a visualizar (0–15)", 0, 15, 0)

                    st.write("Visualización de canal seleccionado:")
                    col_w, col_d = st.columns(2)

                    with col_w:
                        fig1, ax1 = plt.subplots()
                        im1 = ax1.imshow(white_mean[:, :, ch], cmap='gray')
                        ax1.set_title(f"White Median - Canal {ch}")
                        plt.colorbar(im1, ax=ax1)
                        st.pyplot(fig1)

                    with col_d:
                        fig2, ax2 = plt.subplots()
                        im2 = ax2.imshow(dark_mean[:, :, ch], cmap='gray')
                        ax2.set_title(f"Dark Median - Canal {ch}")
                        plt.colorbar(im2, ax=ax2)
                        st.pyplot(fig2)

                    # Guardar los archivos mean.npy en las carpetas de origen
                    white_mean_path = os.path.join(white_ref_folder, "white_mean.npy")
                    dark_mean_path = os.path.join(dark_ref_folder, "dark_mean.npy")
                    np.save(white_mean_path, white_mean)
                    np.save(dark_mean_path, dark_mean)
                    st.success(f"Promedios guardados como:\n- '{white_mean_path}'\n- '{dark_mean_path}'")
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

        # Seleccionar archivos white_mean.npy y dark_mean.npy con tkinter
        col_white, col_dark = st.columns(2)
        with col_white:
            if st.button("Seleccionar archivo white_mean.npy"):
                white_mean_path = subprocess.check_output([
                    "python3", "-c",
                    "import tkinter as tk; from tkinter import filedialog; root = tk.Tk(); root.withdraw(); print(filedialog.askopenfilename(filetypes=[('NumPy files', '*.npy')], initialdir='/home/alonso/Desktop'))"
                ]).decode("utf-8").strip()
                st.session_state['white_mean_path'] = white_mean_path
                st.success("Archivo white_mean.npy seleccionado correctamente.")

        with col_dark:
            if st.button("Seleccionar archivo dark_mean.npy"):
                dark_mean_path = subprocess.check_output([
                    "python3", "-c",
                    "import tkinter as tk; from tkinter import filedialog; root = tk.Tk(); root.withdraw(); print(filedialog.askopenfilename(filetypes=[('NumPy files', '*.npy')], initialdir='/home/alonso/Desktop'))"
                ]).decode("utf-8").strip()
                st.session_state['dark_mean_path'] = dark_mean_path
                st.success("Archivo dark_mean.npy seleccionado correctamente.")

        white_mean_path = st.session_state.get('white_mean_path', '')
        dark_mean_path = st.session_state.get('dark_mean_path', '')

        if white_mean_path:
            with col_white:
                st.caption(f"white_mean.npy: {white_mean_path}")
        if dark_mean_path:
            with col_dark:
                st.caption(f"dark_mean.npy: {dark_mean_path}")

        folders = st.session_state.get('reflectance_folders', [])
        # Controlar el flujo según las selecciones hechas
        if folders and white_mean_path and dark_mean_path:
            try:
                white = np.load(white_mean_path)
                dark = np.load(dark_mean_path)
                sigma_xy = max(white.shape[:2]) / 50  # ~2 % del FOV
                white_smooth = gaussian_filter(white, sigma=(sigma_xy, sigma_xy, 0))

                for folder in folders:
                    st.write(f"Procesando carpeta: {folder}")
                    images = load_image_stack(folder)
                    output_dir = os.path.join(folder, "reflectance")
                    os.makedirs(output_dir, exist_ok=True)

                    for fname, img_array in images:
                        # --- BLOQUE NUEVO: reflectancia robusta ---
                        epsilon = 1e-6
                        denom  = white_smooth - dark
                        bad    = denom <= epsilon
                        denom  = np.where(bad, epsilon, denom)

                        refl = (img_array - dark) / denom
                        refl[bad] = np.nan
                        refl = np.clip(refl,0,1).astype(np.float32)

                        out_path = os.path.join(
                            output_dir,
                            fname.replace(".tif","_refl.npy").replace(".tiff","_refl.npy")
                        )
                        np.save(out_path, refl)
                        # --- FIN BLOQUE NUEVO ---
                    st.success(f"Carpeta '{folder}' procesada y guardada en '{output_dir}'")
            except Exception as e:
                st.error(f"Error al cargar los archivos de referencia: {e}")
        elif folders or white_mean_path or dark_mean_path:
            # Mostrar advertencias específicas según lo que falte
            if not folders:
                st.warning("Debes seleccionar al menos una carpeta de imágenes para procesar.")
            if not white_mean_path:
                st.warning("Debes seleccionar el archivo white_mean.npy.")
            if not dark_mean_path:
                st.warning("Debes seleccionar el archivo dark_mean.npy.")
    
with tab2:
    # --- NUEVA FUNCIÓN ROBUSTA ---
    def load_image_stack(folder_path, expected_channels=16, page_main=1):
        """
        Devuelve [(fname, img)] donde img es (H,W,bandas).
        page_main=1 ⇒ primero intenta la página 1 (cubo completo)
        y si no existe usa la 0 (compatibilidad con TIFF antiguos).
        """
        import tifffile, numpy as np, os, streamlit as st

        imgs=[]
        for fname in sorted(f for f in os.listdir(folder_path)
                            if f.lower().endswith(('.tif','.tiff'))):
            try:
                with tifffile.TiffFile(os.path.join(folder_path,fname)) as tif:
                    key = page_main if page_main < len(tif.pages) else 0
                    arr = tif.asarray(key=key)
                if arr.ndim==3 and arr.shape[0]==expected_channels:
                    arr=np.moveaxis(arr,0,-1)          # (H,W,C)
                if arr.ndim==3 and arr.shape[-1]==expected_channels:
                    imgs.append((fname,arr.astype(np.float32)))
                else:
                    st.warning(f"{fname}: canales={arr.shape[-1]} (omitido)")
            except Exception as e:
                st.warning(f"No se pudo cargar {fname}: {e}")
        return imgs
    # --- FIN NUEVA FUNCIÓN ---

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
                can1, can2 = st.columns(2)
                with can1:
                    ch_hbo2 = st.number_input("Canal para HbO2", min_value=0, max_value=15, value=5)
                with can2:
                    ch_hb = st.number_input("Canal para Hb", min_value=0, max_value=15, value=2)

                # Menú de normalización justo debajo de la selección de canales
                export_mode = st.selectbox("Modo de normalización para visualizar/exportar", ["lineal", "percentile", "std", "log"])

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

                    if st.checkbox("Aplicar estabilización de movimiento", value=False):
                        with st.spinner("Estabilizando imágenes..."):
                            stack_hbo2 = stabilize_stack(stack_hbo2)
                            stack_hb = stabilize_stack(stack_hb)

                    mean_hbo2 = np.mean(stack_hbo2, axis=0)
                    mean_hb = np.mean(stack_hb, axis=0)

                    epsilon = 1e-6
                    ox_map = mean_hbo2 / (mean_hbo2 + mean_hb + epsilon)

                    # Normalizar para visualización según el modo seleccionado
                    ox_map_norm = normalize_image(ox_map, mode=export_mode)

                    fig, ax = plt.subplots()
                    im = ax.imshow(ox_map_norm, cmap='coolwarm', vmin=0, vmax=1)
                    ax.set_title("Mapa de oxigenación estimada HbO2 / (HbO2 + Hb)")
                    plt.colorbar(im, ax=ax)
                    st.pyplot(fig)

                    ox_time_series = stack_hbo2 / (stack_hbo2 + stack_hb + epsilon)
                    st.session_state['ox_time_series'] = ox_time_series
                    st.session_state['file_names'] = file_names

                    if st.button("Exportar imágenes como NPY"):
                        export_folder = os.path.join(perf_folder, "oxigenacion_export")
                        os.makedirs(export_folder, exist_ok=True)
                        for i, (img, fname) in enumerate(zip(ox_time_series, file_names)):
                            out_path = os.path.join(export_folder, os.path.splitext(fname)[0] + "_ox.npy")
                            np.save(out_path, img)
                        st.success(f"Exportadas {len(file_names)} imágenes a: {export_folder}")
                else:
                    st.warning("No se pudieron generar stacks válidos para los canales seleccionados.")
            else:
                st.warning("No se encontraron archivos .npy en la carpeta seleccionada.")

    with st.expander("📈 View oxygenation time series (mean HbO2 / (HbO2 + Hb) over time) with anomaly detection"):
        ox_time_series = st.session_state.get('ox_time_series', None)
        file_names = st.session_state.get('file_names', [])

        if ox_time_series is not None:
            mean_oxygenation_per_frame = np.mean(ox_time_series, axis=(1,2))

            # Interactive threshold slider
            drop_threshold = st.slider(
                "Select threshold for detecting sudden drops",
                min_value=0.01, max_value=0.2, value=0.05, step=0.005,
                help="Drops larger than this fraction between consecutive frames will be flagged as anomalies"
            )

            # Compute linear trend line
            x = np.arange(len(mean_oxygenation_per_frame))
            coeffs = np.polyfit(x, mean_oxygenation_per_frame, 1)
            trend_line = np.polyval(coeffs, x)
            slope = coeffs[0]

            # Interpret trend
            if slope > 0.001:
                trend_text = f"oxygenation is **increasing** over time (slope: +{slope:.4f})"
            elif slope < -0.001:
                trend_text = f"oxygenation is **decreasing** over time (slope: {slope:.4f})"
            else:
                trend_text = f"oxygenation is **stable** over time (slope: {slope:.4f})"

            # Detect sudden drops
            drops = np.where(np.diff(mean_oxygenation_per_frame) < -drop_threshold)[0] + 1  # +1 to get the drop point

            # Plot
            fig, ax = plt.subplots()
            ax.plot(mean_oxygenation_per_frame, marker='o', label='Mean oxygenation')
            ax.plot(x, trend_line, color='red', linestyle='--', label='Trend line')
            if len(drops) > 0:
                ax.scatter(drops, mean_oxygenation_per_frame[drops], color='purple', label=f'Sudden drops (> {drop_threshold:.3f})', zorder=5)
            ax.set_xlabel("Frame index")
            ax.set_ylabel("Mean Oxygenation Index")
            ax.set_title("Oxygenation evolution over time")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            # Explanation
            drop_msg = (
                f"⚠️ Detected sudden drops at frames: {', '.join(map(str, drops))}"
                if len(drops) > 0 else f"✅ No sudden drops detected above {drop_threshold:.3f} threshold."
            )
            st.markdown(f"""
                **What does this show?**  
                This plot displays the average oxygenation index over time with a linear trend line (dashed red).

                According to this, {trend_text}.

                {drop_msg}

                **Max oxygenation:** {mean_oxygenation_per_frame.max():.3f}  
                **Min oxygenation:** {mean_oxygenation_per_frame.min():.3f}
            """)
        else:
            st.info("Generate an oxygenation map first to see the time series.")
with tab3:
    # ———————————————————————————————————————————
    # Función para normalizar imagen
    # modo "lineal", "percentile", "std", "log", "custom"
    def normalize_image(img, mode="lineal", custom_min=None, custom_max=None):
        img = np.nan_to_num(img)
        if mode == "lineal":
            return (img - np.min(img)) / (np.ptp(img) + 1e-6)
        elif mode == "percentile":
            p2, p98 = np.percentile(img, (2,98))
            return np.clip((img - p2) / (p98 - p2 + 1e-6), 0, 1)
        elif mode == "std":
            mean, std = np.mean(img), np.std(img)
            return np.clip((img - (mean - 2*std)) / (4*std + 1e-6), 0, 1)
        elif mode == "log":
            img = np.log1p(img - np.min(img))
            return (img - np.min(img)) / (np.ptp(img) + 1e-6)
        elif mode == "custom" and custom_min is not None and custom_max is not None:
            return np.clip((img - custom_min) / (custom_max - custom_min + 1e-6), 0, 1)
        else:
            return (img - np.min(img)) / (np.ptp(img) + 1e-6)
    # ———————————————————————————————————————————
    controls,imag= st.columns([0.7,2])
    with controls:
    # Expander: cargar mapas de oxigenación
        with st.expander("📂 Select folder with post-calculation oxygenation maps (.npy)"):
            perf_folder = st.session_state.get('roi_perf_folder', '')
            if st.button("Select folder with .npy files"):
                perf_folder = subprocess.check_output([
                    "python3", "-c",
                    "import tkinter as tk; from tkinter import filedialog; root=tk.Tk(); root.withdraw(); print(filedialog.askdirectory(initialdir='/home/alonso/Desktop'))"
                ]).decode().strip()
                st.session_state['roi_perf_folder'] = perf_folder
            if perf_folder and os.path.isdir(perf_folder):
                files = sorted([f for f in os.listdir(perf_folder) if f.endswith('.npy')])
                ox_time_series = np.stack([np.load(os.path.join(perf_folder,f)) for f in files], axis=0) if files else None
                if ox_time_series is None:
                    st.warning("No .npy files found.")
                else:
                    st.success(f"Loaded {len(files)} frames of oxygenation maps.")
            else:
                ox_time_series = None

        # Main
        if ox_time_series is not None:
            # Parámetros UI
            norm_mode = st.selectbox("Normalization mode", ["lineal","percentile","std","log","custom"])
            mean_ox_map = np.mean(ox_time_series, axis=0)
            roi_radius = st.slider("ROI radius (px)", 3, 100, 10)
            drop_threshold = st.slider("Drop threshold for sudden drops", 0.01, 0.2, 0.05, 0.005)
            st.session_state.setdefault('roi_points', [])
            tracking_enabled = st.checkbox("Enable tracking and video export")

            # Custom normalization sliders
            custom_min, custom_max = None, None
            if norm_mode == "custom":
                min_val = float(np.min(mean_ox_map))
                max_val = float(np.max(mean_ox_map))
                st.write("Adjust normalization range:")
                custom_min, custom_max = st.slider(
                    "Select min and max for normalization",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                    step=(max_val-min_val)/100 if max_val>min_val else 0.01,
                    format="%.4f"
                )

            # Guardar y cargar ROIs
            col_save, col_load = st.columns(2)
            with col_save:
                if st.button("💾 Save ROIs to CSV"):
                    if st.session_state['roi_points']:
                        df = pd.DataFrame(st.session_state['roi_points'], columns=['x', 'y'])
                        csv_bytes = df.to_csv(index=False).encode('utf-8')
                        st.download_button("Download ROI CSV", csv_bytes, file_name="rois.csv", mime="text/csv")
                    else:
                        st.info("No ROIs to save.")

            if st.button("📂 Load ROIs from CSV"):
                csv_path = subprocess.check_output([
                    "python3", "-c",
                    "import tkinter as tk; from tkinter import filedialog; root=tk.Tk(); root.withdraw(); print(filedialog.askopenfilename(filetypes=[('CSV files', '*.csv')], initialdir='/home/alonso/Desktop'))"
                ]).decode().strip()
                if os.path.isfile(csv_path):
                    try:
                        df = pd.read_csv(csv_path)
                        if {'x','y'}.issubset(df.columns):
                            st.session_state['roi_points'] = list(df[['x','y']].itertuples(index=False, name=None))
                            st.success(f"Loaded {len(st.session_state['roi_points'])} ROIs from CSV.")
                            st.rerun()
                        else:
                            st.error("CSV must have columns 'x' and 'y'.")
                    except Exception as e:
                        st.error(f"Error loading CSV: {e}")

            with imag:
                # Mostrar mapa medio y seleccionar ROIs
                norm_img = normalize_image(mean_ox_map, norm_mode, custom_min, custom_max)
                cmap = cm.get_cmap('coolwarm')
                rgb = (cmap(norm_img)[..., :3] * 255).astype(np.uint8)  # Ensure only RGB channels
                disp = Image.fromarray(rgb).resize((350, int(350 * rgb.shape[0] / rgb.shape[1])))
                draw = ImageDraw.Draw(disp)
                sx = mean_ox_map.shape[1] / disp.width
                sy = mean_ox_map.shape[0] / disp.height

                for i, (x, y) in enumerate(st.session_state['roi_points']):
                    xd, yd = int(x / sx), int(y / sy)
                    draw.ellipse([(xd - roi_radius, yd - roi_radius), (xd + roi_radius, yd + roi_radius)], outline="red", width=2)
                    draw.text((xd + roi_radius + 2, yd), f"ROI {i+1}", fill="red")

                st.markdown("Click to add ROIs")
                click = streamlit_image_coordinates(disp, key="coord")
                if click:
                    x0, y0 = int(click['x'] * sx), int(click['y'] * sy)
                    if min([np.hypot(x0 - x, y0 - y) for x, y in st.session_state['roi_points']] + [np.inf]) > roi_radius:
                        st.session_state['roi_points'].append((x0, y0))
                        st.rerun()

                if st.session_state['roi_points']:
                    if st.button("Clear all ROIs"):
                        st.session_state['roi_points'].clear()
                        st.rerun()
                else:
                    st.info("Select at least one ROI.")
        else:
            st.info("No oxygenation maps loaded. Please select a folder with .npy files to proceed.")

        # Tracking y export (move inside the ox_time_series block)
        if ox_time_series is not None:
            if tracking_enabled and st.session_state['roi_points']:
                n_frames, H, W = ox_time_series.shape[:3]
                # VideoWriter
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_path = '/tmp/roi_tracking.mp4'
                writer = cv2.VideoWriter(video_path, fourcc, 5, (W, H))

                # Prepara patches desde la imagen normalizada y coloreada (como la de selección de ROIs)
                first = ox_time_series[0]
                norm_first = normalize_image(first, norm_mode, custom_min, custom_max)
                cmap = cm.get_cmap('coolwarm')
                rgb_first = (cmap(norm_first)[:,:,:3] * 255).astype(np.uint8)
                patches, centers = [], []
                search_size = roi_radius * 3
                for x, y in st.session_state['roi_points']:
                    x0, y0 = int(x), int(y)
                    patch = rgb_first[y0-roi_radius:y0+roi_radius, x0-roi_radius:x0+roi_radius, :].copy()
                    patches.append(patch)
                    centers.append((x0, y0))

                yy, xx = np.ogrid[:H, :W]
                csv_data = {'Frame': np.arange(n_frames)}
                # Color list
                color_list = plt.cm.tab10.colors + plt.cm.Set3.colors + plt.cm.Dark2.colors
                colors = [(int(r*255), int(g*255), int(b*255)) for r,g,b in color_list]

                # Loop
                for i in range(n_frames):
                    frame = ox_time_series[i]
                    norm_img = normalize_image(frame, norm_mode, custom_min, custom_max)
                    rgb = (cmap(norm_img)[:,:,:3] * 255).astype(np.uint8)
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                    for j, patch in enumerate(patches):
                        cx_prev, cy_prev = centers[j]
                        x1 = max(cx_prev - search_size, 0)
                        y1 = max(cy_prev - search_size, 0)
                        x2 = min(cx_prev + search_size, W - 2*roi_radius)
                        y2 = min(cy_prev + search_size, H - 2*roi_radius)
                        win = rgb[y1:y2+2*roi_radius, x1:x2+2*roi_radius, :]

                        res = cv2.matchTemplate(win, patch, cv2.TM_CCOEFF_NORMED)
                        _, maxVal, _, maxLoc = cv2.minMaxLoc(res)
                        if maxVal > 0.5:
                            tl = (x1 + maxLoc[0], y1 + maxLoc[1])
                            cx, cy = tl[0] + roi_radius, tl[1] + roi_radius
                            centers[j] = (cx, cy)
                        else:
                            cx, cy = centers[j]

                        mask = (yy - cy)**2 + (xx - cx)**2 <= roi_radius**2
                        csv_data.setdefault(f'ROI_{j+1}', []).append(np.mean(ox_time_series[i][mask]))
                        cv2.circle(bgr, (int(cx), int(cy)), roi_radius, colors[j % len(colors)], 2)

                    writer.write(bgr)

                writer.release()

                # Exportar CSV y Video
                df = pd.DataFrame(csv_data)
                st.download_button("💾 Download CSV", df.to_csv(index=False).encode('utf-8'), file_name='rois_timeseries.csv', mime='text/csv')
                with open(video_path,'rb') as f:
                    st.download_button("🎥 Download Video", f.read(), file_name='roi_tracking.mp4', mime='video/mp4')

                # Interactividad: selección de ROIs a graficar
                roi_labels = [f"ROI {i+1}" for i in range(len(patches))]
                selected_rois = st.multiselect(
                    "Select ROIs to plot",
                    roi_labels,
                    default=roi_labels,
                    help="Select one or more ROIs to visualize their time series"
                )

                # Graficar cada ROI seleccionado en su propia gráfica
                with imag:
                    x = np.arange(n_frames)
                    for j, label in enumerate(roi_labels):
                        if label in selected_rois:
                            ts = csv_data[f'ROI_{j+1}']
                            fig, ax = plt.subplots(figsize=(8,3))
                            ax.plot(x, ts, marker='o', label=label, color=color_list[j % len(color_list)])
                            trend = np.polyfit(x, ts, 1)
                            ax.plot(x, np.polyval(trend, x), '--', color=color_list[j % len(color_list)], label='Trend')
                            drops = np.where(np.diff(ts) < -drop_threshold)[0] + 1
                            if drops.size:
                                ax.scatter(drops, np.array(ts)[drops], c='k', zorder=5, label='Sudden drop')
                            ax.set(title=f"Oxygenation in {label}", xlabel="Frame", ylabel="Mean O₂ index")
                            ax.legend(fontsize=7)
                            ax.grid(True)
                            fig.tight_layout()
                            st.pyplot(fig, use_container_width=True)
                            ax.set(title=f"Oxygenation in {label}", xlabel="Frame", ylabel="Mean O₂ index")
                            ax.legend(fontsize=7)
                            ax.grid(True)
                            fig.tight_layout()
                            st.pyplot(fig, use_container_width=True)
