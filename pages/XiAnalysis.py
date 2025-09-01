# =========================
# XIMEA Spectral Analyzer + ROI Tracking (simplified)
# =========================
import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import easygui
from bisect import bisect_right

from tifffile import imread as imread_tiff
import skimage.exposure
import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import simpledialog

# --- Config ---
DEFAULT_DIR = "/home/alonso/Desktop/"
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

# =========================
# Loader (NPY + TIFF folder + metadata CSV)
# =========================
def folder_path_acquisition():
    # .npy procesado (T,H,W)
    if st.sidebar.button('Add processed stack .npy file'):
        path = easygui.fileopenbox(
            msg="Select a .npy file with processed stacks",
            default=DEFAULT_DIR,
            filetypes=["*.npy"]
        )
        if path:
            data = np.load(path, allow_pickle=False, mmap_mode="r")
            st.session_state['processed_stack'] = data
            st.session_state.setdefault('logs', []).append(
                f"✅ Processed stack: shape={data.shape}, dtype={data.dtype}"
            )
            # Viewer mínimo
            st.session_state['processed_data'] = {'Index': data}
            st.session_state['processed_keys'] = ['Index']

    # Carpeta con TIFFs (original_stack)
    if st.sidebar.button('Add TIFF folder'):
        tiff_dir_path = easygui.diropenbox("Select folder with TIFF frames", default=DEFAULT_DIR)
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
        df_meta = pd.read_csv(metadata_file)
        if not {'Timestamp', 'log_event'}.issubset(df_meta.columns):
            st.error("CSV must contain columns: Timestamp, log_event")
        else:
            st.session_state['metadata'] = df_meta
            st.session_state['log_map'] = dict(zip(df_meta['Timestamp'].astype(str),
                                                   df_meta['log_event'].astype(str)))

    return (
        st.session_state.get('processed_stack'),
        st.session_state.get('metadata'),
        st.session_state.get('tiff_path')
    )

# Carga (no rompe si aún no hay nada)
try:
    processed_stack, metadata, tiff_path = folder_path_acquisition()
except Exception:
    st.info('Load processed .npy and metadata (.csv) if available.')

# =========================
# Helpers
# =========================
def _norm01(img):
    img = img.astype(np.float32)
    vmin = np.nanmin(img)
    vmax = np.nanmax(img)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax - vmin == 0:
        return np.zeros_like(img, dtype=np.float32), 0.0, 1.0
    out = (img - vmin) / (vmax - vmin + 1e-8)
    return out, float(vmin), float(vmax)

def _load_tiff_list(tiff_path):
    if not tiff_path: return []
    if os.path.isdir(tiff_path):
        return sorted([os.path.join(tiff_path, f) for f in os.listdir(tiff_path)
                       if f.lower().endswith((".tif", ".tiff"))])
    elif os.path.isfile(tiff_path) and tiff_path.lower().endswith((".tif", ".tiff")):
        return [tiff_path]
    return []

# =========================
# Visualizer: processed over TIFF background
# =========================
def visualizer(processed_stack, tiff_path):
    col1, col2 = st.columns([0.55, 1.45])
    with col1:
        T, H, W = processed_stack.shape
        t = st.slider("Frame (t)", 0, T-1, 0, 1)
        alpha = st.slider("Transparencia máscara (α)", 0.0, 1.0, 0.6, 0.05)
        cmap = st.selectbox("Colormap", ["coolwarm", "seismic", "bwr", "magma", "viridis"])
        use_thresh = st.checkbox("Usar umbral en máscara", value=False)
        thr_mode, thr_p, thr_val = None, None, None
        if use_thresh:
            thr_mode = st.radio("Modo umbral", ["Percentil (%)", "Valor absoluto"], horizontal=True)
            if thr_mode == "Percentil (%)":
                thr_p = st.slider("Percentil", 0, 100, 90, 1)
            else:
                thr_val = st.number_input("Valor (absoluto)", value=0.0, format="%.6f")
        edge_only = st.checkbox("Mostrar solo máscara (ocultar fondo fuera de ROI)", value=False)

    files = _load_tiff_list(tiff_path)
    if not files or t >= len(files):
        st.error(f"No hay TIFF para t={t}. Encontrados: {len(files)}")
        return

    bg = imread_tiff(files[t])  # (H,W) o (H,W,C)
    if bg.ndim == 3: bg = bg[..., 0]

    # Ajustar tamaños si difieren
    if bg.shape != (H, W):
        st.warning(f"Tamaño distinto: TIFF {bg.shape} vs stack {(H,W)}. Recortando al mínimo.")
        Hm, Wm = min(bg.shape[0], H), min(bg.shape[1], W)
        bg = bg[:Hm, :Wm]
        proc = processed_stack[t, :Hm, :Wm].astype(np.float32)
        H, W = Hm, Wm
    else:
        proc = processed_stack[t].astype(np.float32)

    bg01, _, _ = _norm01(bg)
    _, vmin, vmax = _norm01(proc)

    # Umbral opcional
    if use_thresh:
        if thr_mode == "Percentil (%)":
            thr_val = np.nanpercentile(proc, thr_p)
        mask = proc >= thr_val
        mask_alpha = alpha * mask.astype(np.float32)
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

# =========================
# Viewer of processed indices (minimal, robust)
# =========================
def viewer_npy():
    with st.expander('Viewer for StO2 and indexes'):
        if 'processed_stack' not in st.session_state or 'processed_data' not in st.session_state:
            st.warning("⚠️ No processed .npy loaded yet.")
            return

        index_data = st.session_state['processed_data']
        keys = st.session_state['processed_keys']
        if 'viewer_state' not in st.session_state:
            st.session_state['viewer_state'] = {
                'selected_index': keys[0],
                'image_selection': 0,
                'mask_toggle': True,
                'threshold': 0.01,
                'enhancement': "None",
                'brightness': 1.0,
                'overlay_alpha': 0.6,
                'exposure_slider': 1.0
            }
        S = st.session_state['viewer_state']

        st.caption('Display')
        S['selected_index'] = st.selectbox('Select type', keys, index=keys.index(S['selected_index']))
        data = index_data[S['selected_index']]
        num_frames = data.shape[0]
        st.write(f"Number of frames: {num_frames}")
        S['image_selection'] = st.slider('Frame', 0, num_frames - 1, value=S['image_selection'], step=1)

        # Máscara solo si existen HbO2 y Hb y seleccionas StO2
        if S['selected_index'] == 'StO2' and {'HbO2','Hb'}.issubset(keys):
            S['mask_toggle'] = st.toggle("Mask low tHb pixels", value=S['mask_toggle'])
            S['threshold'] = st.slider("tHb threshold", 0.0, 0.1, value=S['threshold'], step=0.005)
            tHb = index_data['HbO2'] + index_data['Hb']
            img_to_show = np.where(tHb[S['image_selection']] < S['threshold'],
                                   np.nan, data[S['image_selection']]) if S['mask_toggle'] else data[S['image_selection']]
        else:
            img_to_show = data[S['image_selection']]

        # Dar forma 2D
        if img_to_show.ndim == 1:
            st.warning(f"Shape 1D no visualizable: {img_to_show.shape}")
            return
        elif img_to_show.ndim == 3:
            img_to_show = img_to_show[..., 0]

        # Base desde processed_stack
        processed_stack = st.session_state['processed_stack']
        base_img = processed_stack[S['image_selection']] if processed_stack.ndim == 3 else processed_stack[:, :, 0]

        # Enhancements
        S['enhancement'] = st.radio("Enhancement", ["None", "Brightness", "Auto contrast", "Stretch dynamic range"],
                                    index=["None", "Brightness", "Auto contrast", "Stretch dynamic range"].index(S['enhancement']),
                                    horizontal=True)
        if S['enhancement'] == "Brightness":
            S['brightness'] = st.slider("Brightness", 0.1, 5.0, value=S['brightness'], step=0.1)
            enhanced_base = np.clip(base_img * S['brightness'], 0, 1)
        elif S['enhancement'] == "Auto contrast":
            enhanced_base = skimage.exposure.equalize_adapthist(base_img, clip_limit=0.03)
        elif S['enhancement'] == "Stretch dynamic range":
            enhanced_base = (base_img - base_img.min()) / (base_img.max() - base_img.min() + 1e-8)
        else:
            enhanced_base = base_img

        S['overlay_alpha'] = st.slider("Overlay alpha", 0.0, 1.0, value=S['overlay_alpha'], step=0.05)
        S['exposure_slider'] = st.slider("Exposure slider", 0.1, 5.0, value=S['exposure_slider'], step=0.1)

        # Mostrar
        fig, ax = plt.subplots()
        ax.imshow(enhanced_base, cmap='gray')
        im = ax.imshow(img_to_show, cmap='coolwarm', alpha=S['overlay_alpha'], vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        ax.axis('off')
        st.pyplot(fig)

        # Export PNGs
        if st.button("Export current overlay series as PNGs"):
            output_folder = easygui.diropenbox('Select output folder', default=DEFAULT_DIR)
            if output_folder:
                ts = time.strftime("%Y%m%d-%H%M")
                outdir = os.path.join(output_folder, f"{S['selected_index']}_OverlaySeries_{ts}")
                os.makedirs(outdir, exist_ok=True)
                for i in range(num_frames):
                    fig2, ax2 = plt.subplots()
                    base_img_exp = np.clip(processed_stack[i] * S['exposure_slider'], 0, 1) if processed_stack.ndim == 3 \
                                   else np.clip(processed_stack[:, :, 0] * S['exposure_slider'], 0, 1)
                    ax2.imshow(base_img_exp, cmap='gray')
                    if S['selected_index'] == 'StO2' and {'HbO2','Hb'}.issubset(keys) and S['mask_toggle']:
                        overlay_data = np.where((index_data['HbO2'][i] + index_data['Hb'][i]) < S['threshold'], np.nan, data[i])
                    else:
                        overlay_data = data[i]
                    im2 = ax2.imshow(overlay_data, cmap='coolwarm', alpha=S['overlay_alpha'], vmin=0, vmax=1)
                    ax2.axis('off')
                    fig2.colorbar(im2, ax=ax2, fraction=0.025, pad=0.01)
                    plt.savefig(os.path.join(outdir, f"{S['selected_index']}_overlay_{i}.png"), dpi=300, bbox_inches='tight')
                    plt.close(fig2)
                st.success(f"✅ Saved overlays in: {outdir}")
                st.rerun()

# =========================
# ROI Tracking + CSV/AVI export
# =========================
def tracking_roi_selector(original_stack, processed_stack, p_files_names, log_map, scale=3, output_video='tracking_output.avi'):
    with st.expander('ROI tracker'):
        if st.button("Select ROIs & Track"):
            rois_local = []
            use_global_percentile = True
            global_p1, global_p99 = np.nanpercentile(processed_stack, (1, 99))

            # frame0 canal 0
            frame0 = original_stack[0]
            img0 = frame0[0] if isinstance(frame0, (list, tuple)) else original_stack[0, 0]
            img0 = img0.astype(np.float32)
            p1, p99 = np.percentile(img0, (1, 99))
            img0n = np.clip(img0, p1, p99)
            img0n = (255 * (img0n - p1) / (p99 - p1 + 1e-5)).astype(np.uint8)

            H, W = img0n.shape
            pil_img = Image.fromarray(img0n).resize((W*scale, H*scale))
            root = tk.Tk(); root.title("Select ROIs on Image 0")
            canvas = tk.Canvas(root, width=pil_img.size[0], height=pil_img.size[1]); canvas.pack()
            tk_img = ImageTk.PhotoImage(pil_img); canvas.create_image(0, 0, anchor='nw', image=tk_img)

            rect, start = None, (0, 0)
            def on_press(evt):
                nonlocal rect, start
                start = (evt.x, evt.y)
                rect = canvas.create_rectangle(evt.x, evt.y, evt.x, evt.y, outline='green', width=2)
            def on_drag(evt):
                canvas.coords(rect, start[0], start[1], evt.x, evt.y)
            def on_release(evt):
                x0, y0 = start; x1, y1 = evt.x, evt.y
                x, y = min(x0, x1), min(y0, y1)
                w, h = abs(x1 - x0), abs(y1 - y0)
                if w == 0 or h == 0: return
                x, y, w, h = x//scale, y//scale, w//scale, h//scale
                name = simpledialog.askstring("ROI Name", "Name for this ROI:", parent=root)
                if not name: return
                rois_local.append({'name': name, 'rect': (x, y, w, h)})

            canvas.bind('<ButtonPress-1>', on_press)
            canvas.bind('<B1-Motion>', on_drag)
            canvas.bind('<ButtonRelease-1>', on_release)

            def on_toggle():
                nonlocal use_global_percentile
                use_global_percentile = not use_global_percentile
                btn_toggle.config(text=f"Use global percentiles: {use_global_percentile}")

            btn_toggle = tk.Button(root, text=f"Use global percentiles: {use_global_percentile}", command=on_toggle)
            btn_toggle.pack(pady=5)
            tk.Button(root, text="Finish Selection & Start Tracking", command=root.destroy).pack(pady=10)
            root.mainloop()

            if not rois_local:
                st.info("No ROIs selected.")
                return

            # Frames helper
            if isinstance(original_stack, np.ndarray):
                T, C, H, W = original_stack.shape
                get_frame = lambda i: original_stack[i, 0]
            else:
                T = len(original_stack); H, W = original_stack[0][0].shape
                get_frame = lambda i: original_stack[i][0]

            # Template matching
            roi_tracks = []
            for roi in rois_local:
                name = roi['name']; x, y, w, h = roi['rect']
                template = get_frame(0)[y:y+h, x:x+w].astype(np.float32)
                tp1, tp99 = np.percentile(template, (1,99))
                template = np.clip(template, tp1, tp99)
                template = (255 * (template - tp1) / (tp99 - tp1 + 1e-5)).astype(np.uint8)

                coords = []
                for i in range(T):
                    img = get_frame(i).astype(np.float32)
                    p1, p99 = np.percentile(img, (1,99))
                    img_norm = np.clip(img, p1, p99)
                    img_norm = (255 * (img_norm - p1) / (p99 - p1 + 1e-5)).astype(np.uint8)

                    res = cv2.matchTemplate(img_norm, template, cv2.TM_CCOEFF_NORMED)
                    _, _, _, max_loc = cv2.minMaxLoc(res)
                    x_new, y_new = max_loc
                    coords.append((i, x_new, y_new, w, h))
                roi_tracks.append({'name': name, 'coords': coords})

            # Video overlay
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_name = f"{'global' if use_global_percentile else 'fixed'}_{output_video}"
            out = cv2.VideoWriter(video_name, fourcc, 10, (W, H))

            for i in range(T):
                img = get_frame(i).astype(np.float32)
                p1, p99 = np.percentile(img, (1,99))
                img_norm = np.clip(img, p1, p99)
                img_norm = (255 * (img_norm - p1) / (p99 - p1 + 1e-5)).astype(np.uint8)
                frame_bgr = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)

                mask = processed_stack[i].astype(np.float32)
                if use_global_percentile:
                    mask_norm = np.clip(mask, global_p1, global_p99)
                    mask_norm = (255 * (mask_norm - global_p1) / (global_p99 - global_p1 + 1e-5)).astype(np.uint8)
                else:
                    mask_norm = np.clip(mask, 0, 1)
                    mask_norm = (255 * mask_norm).astype(np.uint8)

                color_mask = cv2.applyColorMap(mask_norm, cv2.COLORMAP_JET)
                frame_bgr = cv2.addWeighted(frame_bgr, 0.6, color_mask, 0.4, 0)

                for roi in roi_tracks:
                    _, x, y, w, h = roi['coords'][i]
                    cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), (0,255,0), 2)

                out.write(frame_bgr)
            out.release()
            st.success(f"✅ Tracking done. Video saved as: {video_name}")

            st.session_state["roi_tracks"] = roi_tracks
            st.session_state["video_file"] = video_name

            # Export CSV
            df = compute_mean_in_tracked_rois(processed_stack, roi_tracks,
                                              st.session_state.get('log_map', {}), p_files_names)
            st.write(df)
            st.download_button("📥 Download CSV of mean ROI values",
                               df.to_csv(index=False).encode('utf-8'),
                               "mean_roi_values.csv", "text/csv")

            # Descargar video
            with open(video_name, 'rb') as f:
                st.download_button("📥 Download tracking video", f,
                                   file_name=os.path.basename(video_name), mime="video/avi")

# =========================
# Mean over tracked ROIs + log alignment
# =========================
def compute_mean_in_tracked_rois(processed_stack, roi_tracks, log_map, p_files_names):
    height, width = processed_stack.shape[1:]
    sorted_ts = sorted(log_map.keys())
    rows = []

    for roi in roi_tracks:
        name = roi['name']
        for (frame_id, x, y, w, h) in roi['coords']:
            y1, y2 = max(0, y), min(y + h, height)
            x1, x2 = max(0, x), min(x + w, width)
            roi_data = processed_stack[frame_id][y1:y2, x1:x2]
            mean_val = float(np.nanmean(roi_data)) if roi_data.size > 0 else None
            if np.isnan(mean_val): mean_val = None

            fname = p_files_names[frame_id]
            ts = fname.split("_imagen")[0] if "_imagen" in fname else fname

            idx = bisect_right(sorted_ts, ts) - 1
            log_text = log_map[sorted_ts[idx]] if idx >= 0 else ""

            rows.append({'frame': fname, 'log_event': log_text, 'roi_name': name, 'mean_value': mean_val})

    df = pd.DataFrame(rows)
    return df.pivot(index=['frame', 'log_event'], columns='roi_name', values='mean_value').reset_index()

# =========================
# Main App
# =========================
def app_main():
    col1, col2, col3 = st.columns([1, 1, 0.6])

    # Datos disponibles
    processed_stack = st.session_state.get("processed_stack")
    original_stack  = st.session_state.get("original_stack")
    p_files_names   = st.session_state.get("p_files_names")
    log_map         = st.session_state.get("log_map")
    file_names_reflectance = st.session_state.get("file_names_reflectance")

    with col2:
        st.caption('Viewers')

    with col1:
        st.subheader("Visualizer")
        tiff_path_default = st.session_state.get("tiff_path", "")
        tiff_path = st.text_input("Ruta de TIFF (carpeta o archivo)", value=tiff_path_default)

        if processed_stack is not None and tiff_path:
            visualizer(processed_stack, tiff_path)
        elif processed_stack is None:
            st.info("Carga primero el processed .npy")
        elif not tiff_path:
            st.info("Selecciona la carpeta con TIFFs para el fondo.")

        if 'processed_data' in st.session_state and 'processed_keys' in st.session_state:
            viewer_npy()
        else:
            st.info("Aún no hay índices para el viewer (se crean al cargar el .npy).")

        if all(v is not None for v in (original_stack, processed_stack, p_files_names)):
            tracking_roi_selector(original_stack, processed_stack, p_files_names, st.session_state.get('log_map', {}))
        else:
            st.info("Para el tracking: carga TIFFs (original_stack) y processed .npy (y opcionalmente CSV para log).")

    with col3:
        st.write('Log')
        try:
            if file_names_reflectance is not None:
                st.dataframe(pd.DataFrame({'Reflectance frame names': file_names_reflectance}),
                             hide_index=True, use_container_width=True)
            if p_files_names is not None:
                st.dataframe(pd.DataFrame({"frame names": p_files_names}),
                             hide_index=True, use_container_width=True)
            if log_map is not None and len(log_map):
                df_log = pd.DataFrame(list(log_map.items()), columns=["Timestamp", "log_event"])
                st.dataframe(df_log, use_container_width=True, hide_index=True)
        except Exception as e:
            st.info(f'Nothing to show ({e})')

# ---- Run ----
app_main()
