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
import tifffile as tiff
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
    import os, math, base64, json
    import numpy as np, tifffile as tiff, cv2
    from PIL import Image, ImageTk
    import tkinter as tk
    from tkinter import simpledialog

    # Ruta por defecto para el video si no existe la constante externa
    DEFAULT_VIDEO_PATH = os.path.join(os.getcwd(), "videos")

    # ---------- Helpers ----------
    def _ensure_gray(img):
        return img[..., 0] if (img.ndim == 3) else img

    def _to_uint8(img):
        if img.dtype == np.uint16:
            return (img >> 8).astype(np.uint8)
        return np.clip(img, 0, 255).astype(np.uint8)

    def _encode_png_u8(img_u8):
        ok, buf = cv2.imencode(".png", img_u8)
        if not ok:
            raise RuntimeError("No se pudo codificar PNG.")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def _decode_png_to_u8(b64):
        arr = np.frombuffer(base64.b64decode(b64.encode("ascii")), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        return img

    def _sanitize_inner_rel(inners, bw, bh):
        fixed = []
        for inn in inners:
            name = inn.get("name")
            ix, iy, iw, ih = map(int, inn["rect"])
            ix = max(0, min(ix, max(0, bw - 1)))
            iy = max(0, min(iy, max(0, bh - 1)))
            iw = max(1, min(iw, max(1, bw - ix)))
            ih = max(1, min(ih, max(1, bh - iy)))
            fixed.append({"name": name, "rect": (ix, iy, iw, ih)})
        return fixed

    def _dedup_inner_rects(inners):
        seen = set()
        dedup = []
        for inn in inners:
            key = tuple(map(int, inn["rect"]))
            if key in seen:
                ix, iy, iw, ih = key
                ix += 2; iy += 2
                inn = {**inn, "rect": (ix, iy, iw, ih)}
                key = (ix, iy, iw, ih)
            seen.add(key)
            dedup.append(inn)
        return dedup

    def _build_saved_payload_from_rois(img0_u8, rois_local):
        payload = {"image_shape": img0_u8.shape, "rois": []}
        for roi in rois_local:
            bx, by, bw, bh = map(int, roi["rect"])
            big_tpl = img0_u8[by:by+bh, bx:bx+bw].copy()

            inners_out = []
            for inn in roi.get("inners", []):
                ix, iy, iw, ih = map(int, inn["rect"])
                ix = max(0, min(ix, max(0, bw - 1)))
                iy = max(0, min(iy, max(0, bh - 1)))
                iw = max(1, min(iw, max(1, bw - ix)))
                ih = max(1, min(ih, max(1, bh - iy)))
                inner_tpl = big_tpl[iy:iy+ih, ix:ix+iw].copy()
                inners_out.append({
                    "name": inn.get("name") or None,
                    "rect": [ix, iy, iw, ih],
                    "inner_template_png_b64": _encode_png_u8(inner_tpl)
                })

            payload["rois"].append({
                "name": roi["name"],
                "rect": [bx, by, bw, bh],
                "inners": inners_out,
                "template_png_b64": _encode_png_u8(big_tpl)
            })
        return payload

    def _rois_from_saved_on_new_img0(saved_payload, img0_u8, img0_f32):
        rois_local = []
        for r in saved_payload.get("rois", []):
            name = r["name"]
            bx0, by0, bw, bh = map(int, r["rect"])
            big_tpl_u8 = _decode_png_to_u8(r["template_png_b64"]).astype(np.float32)
            res_big = cv2.matchTemplate(img0_f32, big_tpl_u8, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(res_big)
            nx, ny = int(max_loc[0]), int(max_loc[1])

            big_now = img0_u8[ny:ny+bh, nx:nx+bw].copy().astype(np.float32)

            inners_rel = []
            for inn in r.get("inners", []):
                ix, iy, iw, ih = map(int, inn["rect"])
                if "inner_template_png_b64" in inn and inn["inner_template_png_b64"]:
                    inner_tpl = _decode_png_to_u8(inn["inner_template_png_b64"]).astype(np.float32)
                    if inner_tpl.shape[0] > 0 and inner_tpl.shape[1] > 0 and \
                       big_now.shape[0] >= inner_tpl.shape[0] and big_now.shape[1] >= inner_tpl.shape[1]:
                        res_in = cv2.matchTemplate(big_now, inner_tpl, cv2.TM_CCOEFF_NORMED)
                        _, _, _, in_loc = cv2.minMaxLoc(res_in)
                        ix, iy = int(in_loc[0]), int(in_loc[1])
                inners_rel.append({"name": inn.get("name"), "rect": (ix, iy, iw, ih)})

            inners_rel = _sanitize_inner_rel(_dedup_inner_rects(inners_rel), bw, bh)
            rois_local.append({"name": name, "rect": (nx, ny, bw, bh), "inners": inners_rel})
        return rois_local

    # ---------- Screen & Scroll helpers ----------
    def _fit_scale_to_screen(img_w, img_h, req_scale=3, screen_frac=0.9):
        tmp = tk.Tk()
        try:
            sw = tmp.winfo_screenwidth()
            sh = tmp.winfo_screenheight()
        finally:
            tmp.destroy()
        max_w = int(sw * screen_frac)
        max_h = int(sh * screen_frac)
        disp_w_req = int(img_w * req_scale)
        disp_h_req = int(img_h * req_scale)
        if disp_w_req <= max_w and disp_h_req <= max_h:
            return req_scale
        scale_w = max_w / max(1, img_w)
        scale_h = max_h / max(1, img_h)
        return max(1.0, min(scale_w, scale_h))

    def _make_scrollable_canvas_with_sidepanel(root, dispW, dispH, pil_img, panel_width=240, title=None):
        """
        Crea una grilla 2 columnas: [Canvas con scrollbars] | [Panel de controles].
        Devuelve (canvas, panel_frame).
        """
        if title:
            root.title(title)

        # Contenedor principal
        main = tk.Frame(root)
        main.pack(fill='both', expand=True)

        # --- Columna 0: área de canvas con scroll ---
        canvas_area = tk.Frame(main)
        canvas_area.grid(row=0, column=0, sticky='nsew')

        xscroll = tk.Scrollbar(canvas_area, orient='horizontal')
        yscroll = tk.Scrollbar(canvas_area, orient='vertical')

        vis_w = min(dispW, root.winfo_screenwidth() - panel_width - 80)  # restamos panel y margen
        vis_h = min(dispH, root.winfo_screenheight() - 140)

        canvas = tk.Canvas(
            canvas_area,
            width=vis_w,
            height=vis_h,
            scrollregion=(0, 0, dispW, dispH),
            xscrollcommand=xscroll.set,
            yscrollcommand=yscroll.set
        )

        xscroll.config(command=canvas.xview)
        yscroll.config(command=canvas.yview)

        canvas.grid(row=0, column=0, sticky='nsew')
        yscroll.grid(row=0, column=1, sticky='ns')
        xscroll.grid(row=1, column=0, sticky='ew')

        canvas_area.grid_rowconfigure(0, weight=1)
        canvas_area.grid_columnconfigure(0, weight=1)

        tk_img = ImageTk.PhotoImage(pil_img)
        canvas.create_image(0, 0, anchor='nw', image=tk_img)
        canvas.img_ref = tk_img  # evitar GC

        # Scroll con rueda
        def _on_mousewheel(event):
            if event.state & 0x0001:  # Shift -> horizontal
                canvas.xview_scroll(-1 if event.delta > 0 else 1, "units")
            else:
                canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

        # --- Columna 1: panel de controles ---
        panel = tk.Frame(main, width=panel_width, padx=8, pady=8, relief='groove', borderwidth=2)
        panel.grid(row=0, column=1, sticky='ns')
        panel.grid_propagate(False)  # mantener ancho fijo

        main.grid_rowconfigure(0, weight=1)
        main.grid_columnconfigure(0, weight=1)

        return canvas, panel

    # ---------- UI (Streamlit) ----------
    mode = st.radio(
        "ROIs a usar:",
        options=["🆕 Nuevos ROIs", "🗂️ Usar ROIs en sesión"],
        horizontal=True
    )

    go = st.button("Select ROIs & Track")
    if not go:
        return

    # --- Frame 0 crudo ---
    img0_raw = _ensure_gray(tiff.imread(tiff_files[0]))
    H, W = img0_raw.shape
    img0_u8 = _to_uint8(img0_raw)
    img0_f32 = img0_u8.astype(np.float32)

    rois_local = []

    # ---------- Opción A: usar ROIs guardados ----------
    if mode == "🗂️ Usar ROIs en sesión":
        saved = st.session_state.get("saved_rois", None)
        if not saved or not saved.get("rois"):
            st.warning("No hay ROIs guardados en sesión. Dibuja nuevos ROIs.")
            mode = "🆕 Nuevos ROIs"
        else:
            try:
                rois_local = _rois_from_saved_on_new_img0(saved, img0_u8, img0_f32)
            except Exception as e:
                st.error(f"No se pudieron reubicar los ROIs guardados: {e}")
                mode = "🆕 Nuevos ROIs"

    # ---------- Opción B: dibujar ROIs nuevos ----------
    if mode == "🆕 Nuevos ROIs":
        ui_preview = img0_u8

        # ====== Ventana 1: ROI GRANDE ======
        rois_local = []
        root = tk.Tk()
        auto_scale = _fit_scale_to_screen(W, H, req_scale=scale, screen_frac=0.9)
        dispW, dispH = int(W * auto_scale), int(H * auto_scale)

        pil_img = Image.fromarray(ui_preview).resize((dispW, dispH), resample=Image.NEAREST)
        canvas, panel = _make_scrollable_canvas_with_sidepanel(
            root, dispW, dispH, pil_img, panel_width=260, title="Select TRACKING ROIs (large)"
        )

        rect, start = None, (0, 0)
        def on_press(evt):
            nonlocal rect, start
            x = int(canvas.canvasx(evt.x)); y = int(canvas.canvasy(evt.y))
            start = (x, y)
            rect = canvas.create_rectangle(x, y, x, y, outline='red', width=2)

        def on_drag(evt):
            x = int(canvas.canvasx(evt.x)); y = int(canvas.canvasy(evt.y))
            canvas.coords(rect, start[0], start[1], x, y)

        def on_release(evt):
            x1 = int(canvas.canvasx(evt.x)); y1 = int(canvas.canvasy(evt.y))
            xs, ys = min(start[0], x1), min(start[1], y1)
            ws, hs = abs(x1 - start[0]), abs(y1 - start[1])
            if ws > 0 and hs > 0:
                name = simpledialog.askstring("ROI Name", "Name for this LARGE ROI:", parent=root)
                if name:
                    x0 = int(math.floor(xs / auto_scale)); y0 = int(math.floor(ys / auto_scale))
                    x2 = int(math.ceil((xs + ws) / auto_scale)); y2 = int(math.ceil((ys + hs) / auto_scale))
                    x0, y0 = max(0, x0), max(0, y0)
                    x2, y2 = min(W, x2), min(H, y2)
                    rois_local.append({'name': name, 'rect': (x0, y0, x2 - x0, y2 - y0)})

        canvas.bind('<ButtonPress-1>', on_press)
        canvas.bind('<B1-Motion>', on_drag)
        canvas.bind('<ButtonRelease-1>', on_release)

        # ---- CONTROLES A LA DERECHA (panel) ----
        lbl_info = tk.Label(panel, text="Controles", font=("TkDefaultFont", 10, "bold"))
        lbl_info.pack(pady=(0,6), anchor='n')

        tip = ("Arrastra para dibujar ROI grande.\n"
               "Shift + rueda = scroll horizontal.\n"
               "Rueda = scroll vertical.")
        tk.Label(panel, text=tip, justify='left').pack(pady=(0,10), anchor='w')

        tk.Button(panel, text="Finish Selection (Large ROIs)", command=root.destroy).pack(fill='x', pady=6)

        tk.Button(panel, text="Clear Last Rectangle", command=lambda: (
            canvas.delete(rect) if rect else None
        )).pack(fill='x', pady=6)

        root.mainloop()

        if not rois_local:
            st.info("No ROIs selected.")
            return

        # ====== Ventana 2: ROIs PEQUEÑAS (relativas) ======
        TARGET_MIN_WIDTH = 800
        MAX_INNER_SCALE = 8
        MIN_INNER_SCALE = 2

        for roi in rois_local:
            name = roi['name']; x, y, w, h = roi['rect']
            crop = img0_u8[y:y+h, x:x+w]
            roi['inners'] = []
            if crop.size == 0:
                continue

            desired_inner_scale = max(MIN_INNER_SCALE, min(MAX_INNER_SCALE, math.ceil(TARGET_MIN_WIDTH / max(1, w))))
            inner_scale = _fit_scale_to_screen(w, h, req_scale=desired_inner_scale, screen_frac=0.9)
            dispw = int(w * inner_scale); disph = int(h * inner_scale)

            pil_crop = Image.fromarray(crop).resize((dispw, disph), resample=Image.NEAREST)
            inner_root = tk.Tk()
            inner_canvas, inner_panel = _make_scrollable_canvas_with_sidepanel(
                inner_root, dispw, disph, pil_crop, panel_width=260,
                title=f"Small ROIs inside '{name}' (zoom x{inner_scale:.2f})"
            )

            irect, istart = None, (0, 0)
            def i_on_press(evt):
                nonlocal irect, istart
                x = int(inner_canvas.canvasx(evt.x)); y = int(inner_canvas.canvasy(evt.y))
                istart = (x, y)
                irect = inner_canvas.create_rectangle(x, y, x, y, outline='red', width=2)

            def i_on_drag(evt):
                x = int(inner_canvas.canvasx(evt.x)); y = int(inner_canvas.canvasy(evt.y))
                inner_canvas.coords(irect, istart[0], istart[1], x, y)

            def i_on_release(evt):
                x1 = int(inner_canvas.canvasx(evt.x)); y1 = int(inner_canvas.canvasy(evt.y))
                xs, ys = min(istart[0], x1), min(istart[1], y1)
                ws, hs = abs(x1 - istart[0]), abs(y1 - istart[1])
                if ws > 0 and hs > 0:
                    ix0 = int(math.floor(xs / inner_scale)); iy0 = int(math.floor(ys / inner_scale))
                    ix1 = int(math.ceil((xs + ws) / inner_scale));  iy1 = int(math.ceil((ys + hs) / inner_scale))
                    ix0, iy0 = max(0, min(ix0, w-1)), max(0, min(iy0, h-1))
                    ix1, iy1 = max(1, min(ix1, w)),             max(1, min(iy1, h))
                    small_name = simpledialog.askstring("Small ROI Name (optional)", "Name:", parent=inner_root)
                    small_name = (small_name.strip() if small_name else None)
                    roi['inners'].append({'name': small_name, 'rect': (ix0, iy0, ix1 - ix0, iy1 - iy0)})

            def done():
                roi['inners'] = _sanitize_inner_rel(_dedup_inner_rects(roi.get('inners', [])), w, h)
                inner_root.destroy()

            def clear_last_rect():
                if irect is not None:
                    inner_canvas.delete(irect)

            inner_canvas.bind('<ButtonPress-1>', i_on_press)
            inner_canvas.bind('<B1-Motion>', i_on_drag)
            inner_canvas.bind('<ButtonRelease-1>', i_on_release)

            # ---- CONTROLES A LA DERECHA (panel) ----
            tk.Label(inner_panel, text=f"Inners in '{name}'", font=("TkDefaultFont", 10, "bold")).pack(pady=(0,6), anchor='n')
            tip2 = ("Dibuja ROIs pequeños dentro del ROI grande.\n"
                    "Shift + rueda = scroll horizontal.\n"
                    "Rueda = scroll vertical.")
            tk.Label(inner_panel, text=tip2, justify='left').pack(pady=(0,10), anchor='w')

            btn_box = tk.Frame(inner_panel); btn_box.pack(fill='x', pady=8)
            tk.Button(btn_box, text="Undo last", command=lambda: (roi['inners'].pop() if roi['inners'] else None)).pack(fill='x', pady=4)
            tk.Button(btn_box, text="Clear last rect drawn", command=clear_last_rect).pack(fill='x', pady=4)
            tk.Button(btn_box, text="Done (use these)", command=done).pack(fill='x', pady=4)
            tk.Button(btn_box, text="Skip (none)", command=lambda: (roi['inners'].clear(), done())).pack(fill='x', pady=4)

            inner_root.mainloop()

    # ---------- Tracking del BIG (por template) ----------
    roi_tracks = []
    for roi in rois_local:
        name = roi['name']; x, y, w, h = map(int, roi['rect'])
        template = _ensure_gray(img0_raw[y:y+h, x:x+w]).astype(np.float32)
        coords = []
        for i, f in enumerate(tiff_files):
            img = _ensure_gray(tiff.imread(f)).astype(np.float32)
            res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(res)
            coords.append((i, int(max_loc[0]), int(max_loc[1]), w, h))
        roi_tracks.append({'name': name, 'coords': coords, 'inners_rel': roi.get('inners', [])})

    # ---------- Video con overlays ----------
    if os.path.isabs(output_video) or os.path.dirname(output_video):
        video_path = output_video
    else:
        os.makedirs(DEFAULT_VIDEO_PATH, exist_ok=True)
        video_path = os.path.join(DEFAULT_VIDEO_PATH, output_video)

    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), 10, (W, H))
    for i in range(len(tiff_files)):
        img = _ensure_gray(tiff.imread(tiff_files[i]))
        frame8 = _to_uint8(img)
        frame_bgr = cv2.cvtColor(frame8, cv2.COLOR_GRAY2BGR)

        for roi in roi_tracks:
            _, x0, y0, w0, h0 = roi['coords'][i]
            cv2.rectangle(frame_bgr, (x0, y0), (x0+w0, y0+h0), (0,0,255), 2)

            for idx, inner in enumerate(roi.get('inners_rel', []), start=1):
                ix, iy, iw, ih = map(int, inner['rect'])  # relativos
                cx, cy = x0 + ix, y0 + iy                 # absolutos
                cv2.rectangle(frame_bgr, (cx, cy), (cx+iw, cy+ih), (0,0,255), 2)
                label_text = inner['name'].strip() if inner.get('name') else str(idx)
                cv2.putText(frame_bgr, label_text, (cx, max(0, cy-3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,255), 1, cv2.LINE_AA)
        out.write(frame_bgr)
    out.release()

    st.success(f"✅ Tracking done. Video saved as: {video_path}")
    st.session_state["roi_tracks"] = roi_tracks
    st.session_state["video_file"] = video_path

    # ---------- Guardar/actualizar ROIs en sesión ----------
    saved_rois_input = []
    for r in roi_tracks:
        _, bx, by, bw, bh = r["coords"][0]
        inners_rel = _sanitize_inner_rel(r.get("inners_rel", []), bw, bh)
        saved_rois_input.append({
            "name": r["name"],
            "rect": (bx, by, bw, bh),
            "inners": inners_rel
        })
    saved_payload = _build_saved_payload_from_rois(img0_u8, saved_rois_input)
    st.session_state["saved_rois"] = saved_payload

    # ---------- Métricas ----------
    measure_tracks = []
    for roi in roi_tracks:
        big_name = roi['name']
        inners = roi.get('inners_rel', [])
        if inners:
            for idx, inner in enumerate(inners, start=1):
                ix, iy, iw, ih = map(int, inner['rect'])
                series = []
                for (i, x0, y0, w0, h0) in roi['coords']:
                    series.append((i, x0+ix, y0+iy, iw, ih))
                metric_name = inner['name'].strip() if inner.get('name') else f"{big_name}_small{idx}"
                measure_tracks.append({'name': metric_name, 'coords': series})
        else:
            measure_tracks.append({'name': big_name, 'coords': roi['coords']})

    compute_mean_in_tracked_rois(processed_stack, measure_tracks, metadata)
    return roi_tracks




def normalize_img(img, p1=1, p99=99):
    """Escala imagen a 8-bit con percentiles."""
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, (p1, p99))
    img = np.clip(img, lo, hi)
    return (255 * (img - lo) / (hi - lo + 1e-5)).astype(np.uint8)

def compute_mean_in_tracked_rois(processed_stack, roi_tracks, metadata=None, n_base=50, percent=True):
    """
    Calcula la media por ROI y normaliza cada serie con ΔF/F0 usando los primeros n_base frames de ese ROI.
    - n_base: número de frames iniciales por ROI para F0.
    - percent: si True, expresa ΔF/F0 en %.
    """
    import io
    import numpy as np
    import pandas as pd
    import streamlit as st

    height, width = processed_stack.shape[1:]
    rows = []

    # --- Recorrer ROIs y armar tabla larga (frame, roi_name, mean_value)
    for roi in roi_tracks:
        name = roi['name']
        for (frame_id, x, y, w, h) in roi['coords']:
            y1, y2 = max(0, y), min(y + h, height)
            x1, x2 = max(0, x), min(x + w, width)
            roi_data = processed_stack[frame_id][y1:y2, x1:x2]
            mean_val = float(np.nanmean(roi_data)) if roi_data.size > 0 else None
            if mean_val is not None and np.isnan(mean_val):
                mean_val = None

            rows.append({
                "frame": int(frame_id),
                "roi_name": name,
                "mean_value": mean_val
            })

    # --- DataFrame largo
    df_long = pd.DataFrame(rows).dropna(subset=["mean_value"])
    if df_long.empty:
        st.warning("⚠️ No hay datos para mostrar.")
        return pd.DataFrame()

    # --- Orden por frame para definir F0 correctamente
    df_long = df_long.sort_values(["roi_name", "frame"])

    # --- F0 por ROI = media de los primeros n_base valores disponibles
    f0_map = (
        df_long.groupby("roi_name")["mean_value"]
        .apply(lambda s: float(np.nanmean(s.iloc[:n_base])) if len(s) > 0 else np.nan)
        .to_dict()
    )

    # --- ΔF/F0 (o %). Si F0==0 o NaN, dejar NaN y avisar.
    def _norm(row):
        f0 = f0_map.get(row["roi_name"], np.nan)
        if f0 is None or np.isnan(f0) or f0 == 0:
            return np.nan
        val = (row["mean_value"] - f0) / f0
        if percent:
            val *= 100.0
        return float(val)

    df_long["norm_value"] = df_long.apply(_norm, axis=1)

    # --- Avisos de F0 problemático
    bad_rois = [r for r, f0 in f0_map.items() if (f0 is None or np.isnan(f0) or f0 == 0)]
    if bad_rois:
        st.warning(f"⚠️ F₀ no válido (0 o NaN) en: {', '.join(bad_rois)}. Se omiten en la normalización.")

    # --- Tablas anchas
    df_wide_raw = df_long.pivot(index="frame", columns="roi_name", values="mean_value").reset_index()
    df_wide_norm = df_long.pivot(index="frame", columns="roi_name", values="norm_value").reset_index()

    # --- Mostrar tablas
    st.write("📊 Intensidad media en ROIs (sin normalizar):")
    st.dataframe(df_wide_raw)

    st.write(f"📊 Normalizado con primeros {n_base} frames por ROI (ΔF/F₀{' %' if percent else ''}):")
    st.dataframe(df_wide_norm)

    # --- Gráficas
    if len(df_wide_norm.columns) > 1:
        st.write("📈 Gráfica normalizada (ΔF/F₀):")
        st.line_chart(df_wide_norm.set_index("frame"))
    else:
        st.warning("⚠️ No se encontraron ROIs para graficar.")

    # --- Unir metadata al normalizado (que es lo que probablemente exportarás)
    df_final = df_wide_norm.copy()
    if metadata is not None:
        try:
            if not isinstance(metadata, pd.DataFrame):
                metadata = pd.read_csv(metadata)

            if "frame" in metadata.columns:
                df_final = df_wide_norm.merge(metadata, on="frame", how="left")
            else:
                df_final = pd.concat([df_wide_norm, metadata], axis=1)

            st.write("📊 Datos normalizados con metadata añadida:")
            st.dataframe(df_final)
        except Exception as e:
            st.error(f"Error leyendo metadata: {e}")

    # --- Descarga
    csv_buffer = io.StringIO()
    df_final.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Descargar resultados normalizados (CSV)",
        data=csv_buffer.getvalue(),
        file_name="roi_mean_values_normalized.csv",
        mime="text/csv"
    )

    # Para los curiosos: también retorno raw por si luego quieres comparar (ΔF/F₀ no muerde).
    df_final.attrs["raw"] = df_wide_raw
    df_final.attrs["f0_map"] = f0_map
    return df_final




def app_main():

    processed_stack,metadata,tiff_path=folder_path_acquisition()

    if processed_stack is not None and tiff_path:
        tiff_files=processed_visualizer(processed_stack,tiff_path)
        roi_tracks=tracking_roi_selector(tiff_files,processed_stack,metadata)
    
    else:
        st.info("Aún no hay índices para el viewer (se crean al cargar el .npy).")

app_main()
