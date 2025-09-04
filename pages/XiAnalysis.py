import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import easygui
from tifffile import imread as imread_tiff
import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import simpledialog
import tiffile as tiff
import glob


default = "/home/alonso/Desktop/"
st.set_page_config(layout="wide")

# --- Header ---
def header_agsantos():
    c1, c2 = st.columns([4, 1])
    with c1:
        st.title("Spectral analysis of Ximea cameras")
        st.caption('AG-Santos Neurovascular Research Laboratory')
    with c2:
        try:
            st.image('images/agsantos.png', width=130)
        except Exception:
            pass
    st.markdown("---")

header_agsantos()

def folder_path_acquisition():
    # .npy procesado (T,H,W)
    if st.sidebar.button('Add processed stack .npy file'):
        path = easygui.fileopenbox(
            msg="Select a .npy file with processed stacks",
            default=default,
            filetypes=["*.npy"]
        )
        if path:
            data = np.load(path, allow_pickle=False, mmap_mode="r")
            st.session_state['processed_stack'] = data
            st.session_state.setdefault('logs', []).append(
                f"✅ Processed stack: shape={data.shape}, dtype={data.dtype}"
            )
            st.session_state['processed_data'] = {'Index': data}
            st.session_state['processed_keys'] = ['Index']

    if st.sidebar.button('Add TIFF folder'):
        tiff_dir_path = easygui.diropenbox("Select folder with TIFF frames", default=default)
        if tiff_dir_path:
            tiff_files = sorted(
                [os.path.join(tiff_dir_path, f) for f in os.listdir(tiff_dir_path)
                 if f.lower().endswith((".tif", ".tiff"))]
            )
            if not tiff_files:
                st.warning("No TIFF files found in the selected folder.")
            else:
                original_stack = []
                for fp in tiff_files:
                    img = imread_tiff(fp)
                    if img.ndim == 3:
                        img = img[..., 0]   # toma canal 0 si es RGB
                    original_stack.append([img.astype(np.float32)])  # [C=1,H,W]
                st.session_state['original_stack'] = original_stack
                st.session_state['file_names_reflectance'] = [os.path.basename(f) for f in tiff_files]
                st.session_state['p_files_names'] = [os.path.basename(f) for f in tiff_files]
                st.session_state['tiff_path'] = tiff_dir_path

    # Metadata CSV -> log_map
    metadata_file = st.sidebar.file_uploader("📂 Upload CSV metadata (Timestamp,log_event)", type=".csv")
    if metadata_file is not None:
        metadata=pd.read_csv(metadata_file)
        st.session_state["metadata"]=metadata
    return (
        st.session_state.get('processed_stack',None),
        st.session_state.get('metadata',None),
        st.session_state.get('tiff_path',None)
    )

def processed_visualizer(processed_stack, tiff_folder):
    col1, col2= st.columns([0.5, 2])
    with col1:
        view_mode = st.selectbox(
        "Ver:",
        ["Superimposed","Processed", "Original"] 
    )

        n, h, w = processed_stack.shape
        num_images = st.slider('Select image', 0, max_value=n - 1)

    # --- Procesado ---
    img_proc = processed_stack[num_images, :, :]
    fig_proc, ax_proc = plt.subplots()
    vmin = np.percentile(img_proc, 20)
    vmax = np.percentile(img_proc, 80)
    ax_proc.imshow(img_proc, cmap="coolwarm", vmin=vmin, vmax=vmax)

    # --- Tiffile (original) ---
    tiff_files = sorted(glob.glob(os.path.join(tiff_folder, "*.tif")))
    st.session_state["tiff_files"] = tiff_files
    img_tiff = tiff.imread(tiff_files[num_images])

    if img_tiff.ndim == 2:
        fig_orig, ax_orig = plt.subplots()
        vmin = np.percentile(img_tiff, 2)
        vmax = np.percentile(img_tiff, 98)
        ax_orig.imshow(img_tiff, vmin=vmin, vmax=vmax, cmap="gray")
    elif img_tiff.ndim == 3:
        rgb = img_tiff[:, :, :3].astype(np.float32)
        p2, p98 = np.percentile(rgb[np.isfinite(rgb)], (2, 98))
        rgb_disp = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)
        fig_orig, ax_orig = plt.subplots()
        ax_orig.imshow(rgb_disp)

    # --- Mostrar según el modo ---
    with col2:
        if view_mode == "Processed":
            st.pyplot(fig_proc)
            st.caption("Processed stack image")
        elif view_mode == "Original":
            st.pyplot(fig_orig)
            st.caption("Original TIFF image")
        elif view_mode == "Superimposed":
            fig_overlay, ax_overlay = plt.subplots()

            # Fondo TIFF
            if img_tiff.ndim == 2:
                ax_overlay.imshow(img_tiff, cmap="gray")
            else:
                ax_overlay.imshow(rgb_disp)

            # Mismo vmin/vmax que la vista "Procesado"
            vmin = np.percentile(img_proc, 20)
            vmax = np.percentile(img_proc, 80)

            # Procesado con máscara de blancos
            img_masked = np.ma.masked_where(img_proc > 0.9, img_proc)
            ax_overlay.imshow(img_masked, cmap="coolwarm", vmin=vmin, vmax=vmax, alpha=0.7)

            st.pyplot(fig_overlay)
            st.caption("Overlay con mismos límites de intensidad")
        return tiff_files
def tracking_roi_selector(tiff_files, processed_stack,metadata, scale=3, output_video='tracking_output.avi'):
    if st.button("Select ROIs & Track"):
        # --- Primer frame para selección de ROI ---
        img0 = tiff.imread(tiff_files[0])
        if img0.ndim == 3:   # asegurar 2D
            img0 = img0[..., 0]

        img0n = normalize_img(img0)
        H, W = img0n.shape

        pil_img = Image.fromarray(img0n).resize((W*scale, H*scale))
        rois_local, use_global_percentile = [], True

        root = tk.Tk(); root.title("Select ROIs on Image 0")
        canvas = tk.Canvas(root, width=pil_img.size[0], height=pil_img.size[1]); canvas.pack()
        tk_img = ImageTk.PhotoImage(pil_img); canvas.create_image(0, 0, anchor='nw', image=tk_img)

        rect, start = None, (0, 0)
        def on_press(evt):
            nonlocal rect, start
            start = (evt.x, evt.y)
            rect = canvas.create_rectangle(evt.x, evt.y, evt.x, evt.y, outline='red', width=2)
        def on_drag(evt):
            canvas.coords(rect, start[0], start[1], evt.x, evt.y)
        def on_release(evt):
            x0, y0 = start; x1, y1 = evt.x, evt.y
            x, y, w, h = min(x0, x1)//scale, min(y0, y1)//scale, abs(x1-x0)//scale, abs(y1-y0)//scale
            if w and h:
                name = simpledialog.askstring("ROI Name", "Name for this ROI:", parent=root)
                if name:
                    rois_local.append({'name': name, 'rect': (x, y, w, h)})
        canvas.bind('<ButtonPress-1>', on_press)
        canvas.bind('<B1-Motion>', on_drag)
        canvas.bind('<ButtonRelease-1>', on_release)

        tk.Button(root, text="Finish Selection & Start Tracking", command=root.destroy).pack(pady=10)
        root.mainloop()

        if not rois_local:
            st.info("No ROIs selected.")
            return

        # --- Percentiles globales ---
        global_p1, global_p99 = np.nanpercentile(processed_stack, (1, 99))

        # --- Tracking con template matching ---
        roi_tracks = []
        T = len(tiff_files)
        for roi in rois_local:
            name = roi['name']; x, y, w, h = roi['rect']
            template = normalize_img(img0[y:y+h, x:x+w])
            coords = []
            for i, f in enumerate(tiff_files):
                img = tiff.imread(f)
                if img.ndim == 3:  # asegurar 2D
                    img = img[..., 0]
                img = normalize_img(img)
                res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
                _, _, _, max_loc = cv2.minMaxLoc(res)
                coords.append((i, *max_loc, w, h))
            roi_tracks.append({'name': name, 'coords': coords})

        # --- Crear video con overlay ---
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_name = f"{'global' if use_global_percentile else 'fixed'}_{output_video}"
        out = cv2.VideoWriter(video_name, fourcc, 10, (W, H))

        for i, f in enumerate(tiff_files):
            img = tiff.imread(f)
            if img.ndim == 3:
                img = img[..., 0]
            img = normalize_img(img)
            frame_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            mask = processed_stack[i].astype(np.float32)
            if use_global_percentile:
                mask = np.clip(mask, global_p1, global_p99)
                mask = (255 * (mask - global_p1) / (global_p99 - global_p1 + 1e-5)).astype(np.uint8)
            else:
                mask = (255 * np.clip(mask, 0, 1)).astype(np.uint8)

            frame_bgr = cv2.addWeighted(frame_bgr, 0.6, cv2.applyColorMap(mask, cv2.COLORMAP_JET), 0.4, 0)

            for roi in roi_tracks:
                _, x, y, w, h = roi['coords'][i]
                cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), (0,255,0), 2)
            out.write(frame_bgr)

        out.release()
        st.success(f"✅ Tracking done. Video saved as: {video_name}")
        st.session_state["roi_tracks"] = roi_tracks
        st.session_state["video_file"] = video_name
        compute_mean_in_tracked_rois(processed_stack,roi_tracks,metadata)
        return roi_tracks

def normalize_img(img, p1=1, p99=99):
    """Escala imagen a 8-bit con percentiles."""
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, (p1, p99))
    img = np.clip(img, lo, hi)
    return (255 * (img - lo) / (hi - lo + 1e-5)).astype(np.uint8)


def compute_mean_in_tracked_rois(processed_stack, roi_tracks, metadata=None):
    import io

    height, width = processed_stack.shape[1:]
    rows = []

    for roi in roi_tracks:
        name = roi['name']
        for (frame_id, x, y, w, h) in roi['coords']:
            y1, y2 = max(0, y), min(y + h, height)
            x1, x2 = max(0, x), min(x + w, width)
            roi_data = processed_stack[frame_id][y1:y2, x1:x2]
            mean_val = float(np.nanmean(roi_data)) if roi_data.size > 0 else None
            if np.isnan(mean_val):
                mean_val = None

            rows.append({
                "frame": frame_id,
                "roi_name": name,
                "mean_value": mean_val
            })

    # --- DataFrame largo
    df_long = pd.DataFrame(rows)

    # --- Convertir a ancho → cada ROI es una columna
    df_wide = df_long.pivot(index="frame", columns="roi_name", values="mean_value").reset_index()

    # --- Mostrar tabla
    st.write("📊 Intensidad media en ROIs:")
    st.dataframe(df_wide)

    # --- Gráfica dinámica con todas las columnas
    if len(df_wide.columns) > 1:  # hay al menos frame + 1 ROI
        st.line_chart(df_wide.set_index("frame"))
    else:
        st.warning("⚠️ No se encontraron ROIs para graficar.")

    # --- Si hay metadata, unir
    df_final = df_wide
    if metadata is not None:
        try:
            if not isinstance(metadata, pd.DataFrame):
                metadata = pd.read_csv(metadata)

            if "frame" in metadata.columns:
                df_final = df_wide.merge(metadata, on="frame", how="left")
            else:
                df_final = pd.concat([df_wide, metadata], axis=1)

            st.write("📊 Datos con metadata añadida:")
            st.dataframe(df_final)
        except Exception as e:
            st.error(f"Error leyendo metadata: {e}")

    # --- Botón de descarga
    csv_buffer = io.StringIO()
    df_final.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Descargar resultados en CSV",
        data=csv_buffer.getvalue(),
        file_name="roi_mean_values.csv",
        mime="text/csv"
    )

    return df_final




def app_main():

    processed_stack,metadata,tiff_path=folder_path_acquisition()

    if processed_stack is not None and tiff_path:
        tiff_files=processed_visualizer(processed_stack,tiff_path)
        roi_tracks=tracking_roi_selector(tiff_files,processed_stack,metadata)
    
    else:
        st.info("Aún no hay índices para el viewer (se crean al cargar el .npy).")

app_main()
