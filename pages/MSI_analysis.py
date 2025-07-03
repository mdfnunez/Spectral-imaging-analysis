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
tab1,tab2,tab3,tab4=st.tabs(['Image processing','Perfusion maps',"ROI analysis","Sat determination"])
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
        print(img)
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
        with st.expander("🔬 Visualizar espectro de absorción y las bandas seleccionadas"):
            st.markdown("""
            Este gráfico muestra los coeficientes de absorción relativos (ε) de HbO₂ y Hb 
            en función de la longitud de onda.  
            Las líneas verticales indican las bandas seleccionadas actualmente para el cálculo 
            de saturación, ayudándote a verificar su posición espectral.
            """)

            # -------------------------------------------------
            # DataFrame con espectro (compacto solo ejemplo)
            data = {
                "Longitud (nm)": [540,542,544,546,548,550,552,554,556,558,560,562,564,566,568,570,572,574,576,578,580,582,584,586,588,590,592,594,596,598,600,602,604,606,608,610,612,614,616,618,620,622,624,626,628,630,632,634,636,638,640,642,644,646,648],
                "HBO2": [53236,53292,52096,49868,46660,43016,39675.2,36815.2,34476.8,33456,32613.2,32620,33915.6,36495.2,40172,44496,49172,53308,55540,54728,50104,43304,34639.6,26600.4,19763.2,14400.8,10468.4,7678.8,5683.6,4504.4,3200,2664,2128,1789.2,1647.6,1506,1364.4,1222.8,1110,1026,942,858,774,707.6,658.8,610,561.2,512.4,478.8,460.4,442,423.6,405.2,390.4,379.2],
                "HB":   [46592,48148,49708,51268,52496,53412,54080,54520,54540,54164,53788,52276,50572,48828,46948,45072,43340,41716,40092,38467.6,37020,35676.4,34332.8,32851.6,31075.2,28324.4,25470,22574.8,19800,17058.4,14677.2,13622.4,12567.6,11513.2,10477.6,9443.6,8591.2,7762,7344.8,6927.2,6509.6,6193.2,5906.8,5620,5366.8,5148.8,4930.8,4730.8,4602.4,4473.6,4345.2,4216.8,4088.4,3965.08,3857.6],
            }
            df_spec = pd.DataFrame(data)
            df_spec['eps_HBO2'] = df_spec['HBO2'] / 55000
            df_spec['eps_HB']   = df_spec['HB']   / 55000

            # -------------------------------------------------
            # UI para elegir bandas (independiente del otro tab)
            colb1, colb2 = st.columns(2)
            with colb1:
                band1 = st.number_input("Selecciona banda para HbO₂", min_value=0, max_value=15, value=2, step=1, key="band1_spec")
            with colb2:
                band2 = st.number_input("Selecciona banda para Hb", min_value=0, max_value=15, value=5, step=1, key="band2_spec")

            # -------------------------------------------------
            # Marcar λ aproximadas
            band_mapping = {
                0:540,1:548,2:554,3:562,4:568,5:576,6:584,7:590,
                8:596,9:602,10:610,11:616,12:624,13:630,14:638,15:644
            }
            λ1 = band_mapping.get(band1, 560)
            λ2 = band_mapping.get(band2, 580)

            # -------------------------------------------------
            # Plot
            fig, ax = plt.subplots(figsize=(8,5))
            ax.plot(df_spec["Longitud (nm)"], df_spec["eps_HBO2"], label="ε HbO₂", color="crimson")
            ax.plot(df_spec["Longitud (nm)"], df_spec["eps_HB"], label="ε Hb", color="royalblue")
            ax.fill_between(df_spec["Longitud (nm)"], df_spec["eps_HBO2"], df_spec["eps_HB"], color='gray', alpha=0.2)

            ax.axvline(λ1, color="crimson", linestyle="--", lw=2, label=f"Banda HbO₂ ~ {λ1} nm")
            ax.axvline(λ2, color="royalblue", linestyle="--", lw=2, label=f"Banda Hb ~ {λ2} nm")

            ax.set_xlabel("Longitud de onda (nm)")
            ax.set_ylabel("Coeficiente ε (normalizado)")
            ax.set_title("Espectro de absorción relativo y bandas seleccionadas")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
                        # ————————————————————————————————————————————————
            # ————————————————————————————————————————————————
            # Extraer coeficientes ε para las bandas seleccionadas
            row1 = df_spec.iloc[df_spec[df_spec["Longitud (nm)"]==λ1].index[0]]
            row2 = df_spec.iloc[df_spec[df_spec["Longitud (nm)"]==λ2].index[0]]

            eps_hbo2_λ1 = row1['eps_HBO2']
            eps_hb_λ1   = row1['eps_HB']
            eps_hbo2_λ2 = row2['eps_HBO2']
            eps_hb_λ2   = row2['eps_HB']

            # Construir matriz y calcular determinante
            E = np.array([
                [eps_hbo2_λ1, eps_hb_λ1],
                [eps_hbo2_λ2, eps_hb_λ2]
            ])
            det = np.linalg.det(E)

            # Mostrar mensaje interpretativo
            if abs(det) < 0.01:
                st.warning(f"⚠️ El determinante de la matriz Beer–Lambert con estas bandas es muy bajo ({det:.4f}). Esto puede provocar errores numéricos. Considera elegir bandas más separadas en absorción.")
            elif abs(det) < 0.05:
                st.info(f"ℹ️ El determinante es moderado ({det:.4f}). Es aceptable pero revisa la estabilidad.")
            else:
                st.success(f"✅ Excelente: el determinante es {det:.4f}. Buena separación espectral para calcular saturación.")
with tab4:
    with st.expander("🔬 Calcular StO₂ desde cero con ROI calibrado y espectro"):
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import os
        from PIL import Image
        from scipy.ndimage import gaussian_filter
        from streamlit_image_coordinates import streamlit_image_coordinates

        # ————————————————————————————————————————————————
        # FUNCIONES UNICAS
        def load_npy_stack(folder):
            files = sorted([f for f in os.listdir(folder) if f.endswith('.npy')])
            return np.stack([np.load(os.path.join(folder, f)) for f in files], axis=0)

        def normalize_percentile(img):
            p2, p98 = np.nanpercentile(img, (2,98))
            return np.clip((img - p2) / (p98 - p2 + 1e-6), 0, 1)

        def beer_lambert_solver(stack, band1, band2, invE):
            ox_stack = []
            for img in stack:
                A = -np.log(np.clip(np.stack([img[:,:,band1], img[:,:,band2]], axis=-1), 1e-6, 1))
                concs = A @ invE.T
                StO2 = concs[:,:,0] / (concs[:,:,0] + concs[:,:,1] + 1e-6)
                ox_stack.append(StO2)
            return np.stack(ox_stack, axis=0)
        # ————————————————————————————————————————————————

        # Selección de archivos
        col_w, col_d = st.columns(2)
        with col_w:
            if st.button("Seleccionar white_mean.npy"):
                white_mean_path = subprocess.check_output([
                    "python3","-c","import tkinter as tk; from tkinter import filedialog; root=tk.Tk(); root.withdraw(); print(filedialog.askopenfilename(filetypes=[('NumPy files', '*.npy')]))"
                ]).decode().strip()
        with col_d:
            if st.button("Seleccionar dark_mean.npy"):
                dark_mean_path = subprocess.check_output([
                    "python3","-c","import tkinter as tk; from tkinter import filedialog; root=tk.Tk(); root.withdraw(); print(filedialog.askopenfilename(filetypes=[('NumPy files', '*.npy')]))"
                ]).decode().strip()

        refl_folder = ""
        if st.button("Seleccionar carpeta con archivos .npy"):
            refl_folder = subprocess.check_output([
                "python3","-c","import tkinter as tk; from tkinter import filedialog; root=tk.Tk(); root.withdraw(); print(filedialog.askdirectory())"
            ]).decode().strip()

        if white_mean_path and dark_mean_path and refl_folder:
            white = np.load(white_mean_path)
            dark  = np.load(dark_mean_path)
            sigma_xy = max(white.shape[:2]) / 50
            white_smooth = gaussian_filter(white, sigma=(sigma_xy, sigma_xy, 0))
            refl_stack = load_npy_stack(refl_folder)
            stack_refl = []
            for img in refl_stack:
                denom = white_smooth - dark
                denom = np.where(denom <= 1e-6, 1e-6, denom)
                refl = (img - dark) / denom
                refl = np.clip(refl,0,1)
                stack_refl.append(refl)
            stack_refl = np.stack(stack_refl, axis=0)

            # Espectro y matriz Beer–Lambert
            df_spec = pd.DataFrame({
                "λ": [540,542,544,546,548,550,552,554,556,558,560,562,564,566,568,570,572,574,576,578,580,582,584,586,588,590,592,594,596,598,600,602,604,606,608,610,612,614,616,618,620,622,624,626,628,630,632,634,636,638,640,642,644,646,648],
                "HBO2": [53236,53292,52096,49868,46660,43016,39675.2,36815.2,34476.8,33456,32613.2,32620,33915.6,36495.2,40172,44496,49172,53308,55540,54728,50104,43304,34639.6,26600.4,19763.2,14400.8,10468.4,7678.8,5683.6,4504.4,3200,2664,2128,1789.2,1647.6,1506,1364.4,1222.8,1110,1026,942,858,774,707.6,658.8,610,561.2,512.4,478.8,460.4,442,423.6,405.2,390.4,379.2],
                "HB":   [46592,48148,49708,51268,52496,53412,54080,54520,54540,54164,53788,52276,50572,48828,46948,45072,43340,41716,40092,38467.6,37020,35676.4,34332.8,32851.6,31075.2,28324.4,25470,22574.8,19800,17058.4,14677.2,13622.4,12567.6,11513.2,10477.6,9443.6,8591.2,7762,7344.8,6927.2,6509.6,6193.2,5906.8,5620,5366.8,5148.8,4930.8,4730.8,4602.4,4473.6,4345.2,4216.8,4088.4,3965.08,3857.6]
            })
            df_spec['ε_HBO2'] = df_spec['HBO2'] / 55000
            df_spec['ε_HB']   = df_spec['HB']   / 55000
            band_map = {0:540,1:548,2:554,3:562,4:568,5:576,6:584,7:590,8:596,9:602,10:610,11:616,12:624,13:630,14:638,15:644}
            colb1, colb2 = st.columns(2)
            with colb1:
                band1 = st.number_input("Banda para HbO₂", 0,15,2,1)
            with colb2:
                band2 = st.number_input("Banda para Hb", 0,15,5,1)
            λ1,λ2 = band_map[band1], band_map[band2]
            fig, ax = plt.subplots()
            ax.plot(df_spec['λ'], df_spec['ε_HBO2'], label="ε HbO₂", color="crimson")
            ax.plot(df_spec['λ'], df_spec['ε_HB'],   label="ε Hb", color="royalblue")
            ax.axvline(λ1, color="crimson", linestyle="--", lw=2)
            ax.axvline(λ2, color="royalblue", linestyle="--", lw=2)
            ax.fill_between(df_spec['λ'], df_spec['ε_HBO2'], df_spec['ε_HB'], color='gray', alpha=0.2)
            ax.legend(); ax.grid(); st.pyplot(fig)
            E = np.array([[df_spec.loc[df_spec['λ']==λ1,'ε_HBO2'].values[0], df_spec.loc[df_spec['λ']==λ1,'ε_HB'].values[0]],
                          [df_spec.loc[df_spec['λ']==λ2,'ε_HBO2'].values[0], df_spec.loc[df_spec['λ']==λ2,'ε_HB'].values[0]]])
            det = np.linalg.det(E)
            if abs(det)<0.01:
                st.warning(f"⚠️ Determinante muy bajo: {det:.5f}")
            invE = np.linalg.inv(E)

            # Beer-Lambert
            ox_time_series = beer_lambert_solver(stack_refl, band1, band2, invE)
            mean_ox_map = np.nanmean(ox_time_series, axis=0)

            # ROI selector con normalización robusta
            st.write("Selecciona ROI para calibrar al 98%")
            vmin,vmax = float(np.nanmin(mean_ox_map)), float(np.nanmax(mean_ox_map))
            if abs(vmax - vmin) < 1e-5:
                vmin, vmax = vmin - 0.05, vmax + 0.05  # forzar rango mínimo
            vm1,vm2 = st.slider("Ajusta rango para visualización", vmin, vmax, (vmin, vmax))

            norm_disp = np.clip((mean_ox_map - vm1) / (vm2 - vm1 + 1e-6),0,1)
            disp = Image.fromarray((norm_disp*255).astype(np.uint8))
            click = streamlit_image_coordinates(disp, key="roi_calib_v4")
            if click:
                x,y = int(click['x']), int(click['y'])
                r = st.slider("Radio ROI",3,50,10)
                yy,xx = np.ogrid[:mean_ox_map.shape[0], :mean_ox_map.shape[1]]
                mask = (yy-y)**2 + (xx-x)**2 <= r**2
                mean_roi = np.nanmean(mean_ox_map[mask])
                st.info(f"Media ROI: {mean_roi:.3f}")
                if st.button("Calibrar stack a ≈98%"):
                    ox_calibrated = ox_time_series / mean_roi * 0.98
                    fig, ax = plt.subplots()
                    ax.imshow(np.nanmean(ox_calibrated, axis=0), cmap='coolwarm', vmin=0,vmax=1)
                    ax.set_title("Stack calibrado (media)")
                    plt.colorbar(ax.images[0], ax=ax)
                    st.pyplot(fig)
        else:
            st.info("Selecciona los archivos y la carpeta para comenzar.")
