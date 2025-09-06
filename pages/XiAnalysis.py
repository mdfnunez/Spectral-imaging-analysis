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

import os, math, cv2, numpy as np, tifffile as tiff
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import simpledialog
import streamlit as st

# Ruta por defecto para el video
DEFAULT_VIDEO_PATH = default  # <- tu variable/carpeta por defecto

def tracking_roi_selector(tiff_files, processed_stack, metadata, scale=3, output_video='tracking_output.avi'):
    if st.button("Select ROIs & Track"):
        # --- Frame 0 crudo ---
        img0 = tiff.imread(tiff_files[0])
        if img0.ndim == 3:
            img0 = img0[..., 0]
        H, W = img0.shape

        # Vista para UI (solo mostrar; cálculos usan crudo)
        ui_preview = (img0 >> 8).astype(np.uint8) if img0.dtype == np.uint16 else np.clip(img0,0,255).astype(np.uint8)

        # ============================================================
        # Ventana 1: ROI GRANDE (tracking) con escala EXACTA = 'scale'
        # ============================================================
        rois_local = []
        root = tk.Tk(); root.title("Select TRACKING ROIs (large)")
        dispW, dispH = int(W*scale), int(H*scale)
        pil_img = Image.fromarray(ui_preview).resize((dispW, dispH), resample=Image.NEAREST)
        canvas = tk.Canvas(root, width=dispW, height=dispH); canvas.pack()
        tk_img = ImageTk.PhotoImage(pil_img); canvas.create_image(0, 0, anchor='nw', image=tk_img)

        rect, start = None, (0, 0)
        def on_press(evt):
            nonlocal rect, start
            start = (evt.x, evt.y)
            rect = canvas.create_rectangle(evt.x, evt.y, evt.x, evt.y, outline='red', width=2)
        def on_drag(evt):
            canvas.coords(rect, start[0], start[1], evt.x, evt.y)
        def on_release(evt):
            x0s, y0s = start; x1s, y1s = evt.x, evt.y
            xs, ys = min(x0s, x1s), min(y0s, y1s)
            ws, hs = abs(x1s - x0s), abs(y1s - y0s)
            if ws > 0 and hs > 0:
                name = simpledialog.askstring("ROI Name", "Name for this LARGE ROI:", parent=root)
                if name:
                    # Volver a coords originales
                    x0 = int(math.floor(xs / scale)); y0 = int(math.floor(ys / scale))
                    x1 = int(math.ceil((xs + ws) / scale)); y1 = int(math.ceil((ys + hs) / scale))
                    x0, y0 = max(0,x0), max(0,y0)
                    x1, y1 = min(W,x1), min(H,y1)
                    rois_local.append({'name': name, 'rect': (x0, y0, x1 - x0, y1 - y0)})
        canvas.bind('<ButtonPress-1>', on_press)
        canvas.bind('<B1-Motion>', on_drag)
        canvas.bind('<ButtonRelease-1>', on_release)
        tk.Button(root, text="Finish Selection (Large ROIs)", command=root.destroy).pack(pady=10)
        root.mainloop()

        if not rois_local:
            st.info("No ROIs selected.")
            return

        # ============================================================
        # Ventana 2: múltiples ROIs PEQUEÑAS por cada GRANDE (zoom)
        # ============================================================
        TARGET_MIN_WIDTH = 800
        MAX_INNER_SCALE = 8
        MIN_INNER_SCALE = 2

        for roi in rois_local:
            name = roi['name']; x, y, w, h = roi['rect']
            crop = ui_preview[y:y+h, x:x+w]
            roi['inners'] = []
            if crop.size == 0:
                continue

            inner_scale = max(MIN_INNER_SCALE, min(MAX_INNER_SCALE, math.ceil(TARGET_MIN_WIDTH / max(1, w))))
            dispw = int(w * inner_scale); disph = int(h * inner_scale)

            pil_crop = Image.fromarray(crop).resize((dispw, disph), resample=Image.NEAREST)
            inner_root = tk.Tk(); inner_root.title(f"Small ROIs inside '{name}' (zoom x{inner_scale})")
            inner_canvas = tk.Canvas(inner_root, width=dispw, height=disph); inner_canvas.pack()
            inner_tk_img = ImageTk.PhotoImage(pil_crop); inner_canvas.create_image(0, 0, anchor='nw', image=inner_tk_img)

            irect, istart = None, (0, 0)
            def i_on_press(evt):
                nonlocal irect, istart
                istart = (evt.x, evt.y)
                irect = inner_canvas.create_rectangle(evt.x, evt.y, evt.x, evt.y, outline='red', width=2)
            def i_on_drag(evt):
                inner_canvas.coords(irect, istart[0], istart[1], evt.x, evt.y)
            def i_on_release(evt):
                x0s, y0s = istart; x1s, y1s = evt.x, evt.y
                xs, ys = min(x0s, x1s), min(y0s, y1s)
                ws, hs = abs(x1s - x0s), abs(y1s - y0s)
                if ws > 0 and hs > 0:
                    ix0 = int(math.floor(xs / inner_scale)); iy0 = int(math.floor(ys / inner_scale))
                    ix1 = int(math.ceil((xs + ws) / inner_scale));  iy1 = int(math.ceil((ys + hs) / inner_scale))
                    ix0, iy0 = max(0, min(ix0, w-1)), max(0, min(iy0, h-1))
                    ix1, iy1 = max(1, min(ix1, w)),             max(1, min(iy1, h))
                    small_name = simpledialog.askstring("Small ROI Name (optional)", "Name:", parent=inner_root)
                    small_name = (small_name.strip() if small_name else None)
                    roi['inners'].append({'name': small_name, 'rect': (ix0, iy0, ix1 - ix0, iy1 - iy0)})
            def done():
                inner_root.destroy()
            def clear_last():
                if roi['inners']: roi['inners'].pop()
            inner_canvas.bind('<ButtonPress-1>', i_on_press)
            inner_canvas.bind('<B1-Motion>', i_on_drag)
            inner_canvas.bind('<ButtonRelease-1>', i_on_release)
            btn_box = tk.Frame(inner_root); btn_box.pack(pady=8)
            tk.Button(btn_box, text="Undo last", command=clear_last).pack(side='left', padx=6)
            tk.Button(btn_box, text="Done (use these)", command=done).pack(side='left', padx=6)
            tk.Button(btn_box, text="Skip (none)", command=lambda: (roi['inners'].clear(), done())).pack(side='left', padx=6)
            inner_root.mainloop()

        # ============================================================
        # Tracking (template = ROI grande del frame 0)
        # ============================================================
        roi_tracks = []
        for roi in rois_local:
            name = roi['name']; x, y, w, h = roi['rect']
            template = img0[y:y+h, x:x+w].astype(np.float32)
            coords = []
            for i, f in enumerate(tiff_files):
                img = tiff.imread(f)
                if img.ndim == 3: img = img[..., 0]
                res = cv2.matchTemplate(img.astype(np.float32), template, cv2.TM_CCOEFF_NORMED)
                _, _, _, max_loc = cv2.minMaxLoc(res)
                coords.append((i, max_loc[0], max_loc[1], w, h))
            roi_tracks.append({'name': name, 'coords': coords, 'inners_rel': roi['inners']})

        # ============================================================
        # Video (gris) + etiquetas: grande y pequeñas en ROJO; texto pequeño
        # ============================================================
        if os.path.isabs(output_video) or os.path.dirname(output_video):
            video_path = output_video
        else:
            os.makedirs(DEFAULT_VIDEO_PATH, exist_ok=True)
            video_path = os.path.join(DEFAULT_VIDEO_PATH, output_video)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_path, fourcc, 10, (W, H))

        T = len(tiff_files)
        for i in range(T):
            img = tiff.imread(tiff_files[i])
            if img.ndim == 3: img = img[..., 0]
            frame8 = (img >> 8).astype(np.uint8) if img.dtype == np.uint16 else np.clip(img,0,255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame8, cv2.COLOR_GRAY2BGR)

            for roi in roi_tracks:
                _, x0, y0, w0, h0 = roi['coords'][i]
                # ROI grande (rojo)
                cv2.rectangle(frame_bgr, (x0, y0), (x0+w0, y0+h0), (0,0,255), 2)

                # ROIs pequeñas (rojo) + etiqueta (nombre o índice)
                inners = roi.get('inners_rel', [])
                for idx, inner in enumerate(inners, start=1):
                    ix, iy, iw, ih = inner['rect']
                    cx, cy = x0 + ix, y0 + iy
                    cv2.rectangle(frame_bgr, (cx, cy), (cx+iw, cy+ih), (0,0,255), 2)
                    label_text = inner['name'].strip() if inner.get('name') else str(idx)
                    cv2.putText(frame_bgr, label_text, (cx, max(0, cy-3)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,255), 1, cv2.LINE_AA)

            out.write(frame_bgr)
        out.release()

        st.success(f"✅ Tracking done. Video saved as: {video_path}")
        st.session_state["roi_tracks"] = roi_tracks
        st.session_state["video_file"] = video_path

        # ============================================================
        # Métricas: usar nombre de ROI pequeño si existe; si no, BIG_smallN
        # ============================================================
        measure_tracks = []
        for roi in roi_tracks:
            big_name = roi['name']
            inners = roi.get('inners_rel', [])
            if inners:
                for idx, inner in enumerate(inners, start=1):
                    ix, iy, iw, ih = inner['rect']
                    series = []
                    for (i, x0, y0, w0, h0) in roi['coords']:
                        series.append((i, x0+ix, y0+iy, iw, ih))
                    metric_name = inner['name'].strip() if inner.get('name') else f"{big_name}_small{idx}"
                    measure_tracks.append({'name': metric_name, 'coords': series})
            else:
                measure_tracks.append({'name': big_name, 'coords': roi['coords']})

        # Pasa nombres correctos a tu cálculo
        compute_mean_in_tracked_rois(processed_stack, measure_tracks, metadata)
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
