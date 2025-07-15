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
    with st.sidebar.expander('Add reflectance file (.npz)'):
        if st.button('Add reflectance .npz file'):
            path = easygui.fileopenbox(
                msg="Select a .npz file with reflectance stacks",
                default="/home/alonso/Desktop/",
                filetypes=["*.npz"]
            )
            if path:
                data = np.load(path, allow_pickle=True)
                images_list = [data[key] for key in data.files]

                # stack de todas las claves del .npz
                reflectance_list = [data[key] for key in sorted(data.files)]
                reflectance_stack = np.stack(reflectance_list, axis=0)
                
                st.session_state['reflectance_stack'] = reflectance_stack
                st.session_state['reflectance_npz_path'] = path
                st.success(f"✅ Reflectance cargada con shape: {reflectance_stack.shape}")

    # ———————————————————————————————
    # 2) Original TIFF folder
    # ———————————————————————————————
    with st.sidebar.expander('Add original TIFF images folder'):
        if st.button("Add folder with .tiff files"):
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
                    st.success(f"✅ Original TIFF stack: {stack.shape}")
                else:
                    st.error("⚠️ No valid TIFFs found.")

    # ———————————————————————————————
    # 3) Processed NPZ
    # ———————————————————————————————
    with st.sidebar.expander('Add processed indices (.npz)'):
        if st.button("Add processed .npz file"):
            p = easygui.fileopenbox(
                msg="Select a .npz file with processed indices",
                default="/home/alonso/Desktop/",
                filetypes=["*.npz"]
            )
            if p:
                data = np.load(p, allow_pickle=True)
                st.session_state['processed_data'] = data
                st.session_state['processed_files_names']=data["file_names"].tolist()
                st.session_state['processed_keys'] = list(data.files)
                st.session_state['processed_file_path'] = p
                st.success(f"✅ Processed keys: {st.session_state['processed_keys']}")

        if 'processed_data' in st.session_state:
            sel = st.selectbox(
                "Select index to use",
                st.session_state['processed_keys'],
                key="processed_key_selector"
            )
            st.session_state['processed_stack'] = st.session_state['processed_data'][sel]
            st.caption(f"Selected `{sel}` shape: {st.session_state['processed_stack'].shape}")

    # ———————————————————————————————
    # 4) Devuelve los tres stacks
    # ———————————————————————————————
    return (
        st.session_state.get('reflectance_stack', None),
        st.session_state.get('original_stack', None),
        st.session_state.get('processed_stack', None),
        st.session_state.get("processed_files_names",None)
    )


def band_selection():
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.sidebar.expander('Band selection'):
            # table of extinction coefficients
            data = {
                "Wavelengths (nm)": [540,542,544,546,548,550,552,554,556,558,560,562,564,566,568,570,572,574,576,578,580,582,584,586,588,590,592,594,596,598,600,602,604,606,608,610,612,614,616,618,620,622,624,626,628,630,632,634,636,638,640,642,644,646,648],
                "HBO2": [53236,53292,52096,49868,46660,43016,39675.2,36815.2,34476.8,33456,32613.2,32620,33915.6,36495.2,40172,44496,49172,53308,55540,54728,50104,43304,34639.6,26600.4,19763.2,14400.8,10468.4,7678.8,5683.6,4504.4,3200,2664,2128,1789.2,1647.6,1506,1364.4,1222.8,1110,1026,942,858,774,707.6,658.8,610,561.2,512.4,478.8,460.4,442,423.6,405.2,390.4,379.2],
                "HB":   [46592,48148,49708,51268,52496,53412,54080,54520,54540,54164,53788,52276,50572,48828,46948,45072,43340,41716,40092,38467.6,37020,35676.4,34332.8,32851.6,31075.2,28324.4,25470,22574.8,19800,17058.4,14677.2,13622.4,12567.6,11513.2,10477.6,9443.6,8591.2,7762,7344.8,6927.2,6509.6,6193.2,5906.8,5620,5366.8,5148.8,4930.8,4730.8,4602.4,4473.6,4345.2,4216.8,4088.4,3965.08,3857.6],
            }
            df_spec = pd.DataFrame(data)
            
            # Normalization with the maximum value per column
        
            df_spec['eps_HBO2'] = df_spec['HBO2'] / df_spec['HBO2'].max()
            df_spec['eps_HB']   = df_spec['HB']   / df_spec['HB'].max()
            st.caption(f"Normalized using maxima: HbO₂ / {df_spec['HBO2'].max():.1f}, Hb / {df_spec['HB'].max():.1f}")
           
            # Band selection
            colb1, colb2 = st.columns(2)
            with colb1:
                band1 = st.number_input("Select band for HbO₂", min_value=0, max_value=15, value=2, step=1, key="band1_spec")
            with colb2:
                band2 = st.number_input("Select band for Hb", min_value=0, max_value=15, value=5, step=1, key="band2_spec")

            band_mapping = {
                0:540,1:548,2:554,3:562,4:568,5:576,6:584,7:590,
                8:596,9:602,10:610,11:616,12:624,13:630,14:638,15:644
            }
            λ1 = band_mapping.get(band1)
            λ2 = band_mapping.get(band2)

            # -------------------------------------------------
            # Plot
            fig, ax = plt.subplots(figsize=(8,5))
            ax.plot(df_spec["Wavelengths (nm)"], df_spec["eps_HBO2"], label="ε HbO₂", color="crimson")
            ax.plot(df_spec["Wavelengths (nm)"], df_spec["eps_HB"], label="ε Hb", color="royalblue")
            ax.fill_between(df_spec["Wavelengths (nm)"], df_spec["eps_HBO2"], df_spec["eps_HB"], color='gray', alpha=0.2)

            ax.axvline(λ1, color="crimson", linestyle="--", lw=2, label=f"Band HbO₂ ~ {λ1} nm")
            ax.axvline(λ2, color="royalblue", linestyle="--", lw=2, label=f"Band Hb ~ {λ2} nm")

            ax.set_xlabel("Wavelengths (nm)")
            ax.set_ylabel("Coefficient ε (normalized)")
            ax.legend()
            ax.grid(True)
            st.caption('Coefficients and selected bands')
            st.pyplot(fig)

            # -------------------------------------------------
            # Extract coefficients ε for the selected bands
            row1 = df_spec.iloc[df_spec[df_spec["Wavelengths (nm)"]==λ1].index[0]]
            row2 = df_spec.iloc[df_spec[df_spec["Wavelengths (nm)"]==λ2].index[0]]

            st.write(row1)
            st.write(row2)

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

            # Display interpretative message
            if abs(det) < 0.01:
                st.warning(f"⚠️ The determinant of the Beer–Lambert matrix with these bands is very low ({det}). This may cause numerical instability. Consider selecting bands with more distinct absorption.")
            elif abs(det) < 0.05:
                st.info(f"ℹ️ The determinant is moderate ({det}). Acceptable, but monitor stability.")
            else:
                st.success(f"The determinant is {det}. ")

            st.caption("The determinant tells us how well the selected wavelengths can distinguish between HbO₂ and Hb. "
                    "A low determinant means the absorption properties at those wavelengths are too similar, making it hard to separate "
                    "the contributions of each molecule. A higher determinant ensures more reliable and stable oxygenation calculations.")

            return λ1, λ2, eps_hbo2_λ1, eps_hb_λ1, eps_hbo2_λ2, eps_hb_λ2,E,band1,band2
#Sidebar selection of band
λ1, λ2, eps_hbo2_λ1, eps_hb_λ1, eps_hbo2_λ2, eps_hb_λ2,E,band1,band2 = band_selection()

def beer_lambert_sat_calculations():
    with st.expander('Beer-Lambert StO2 calculations (DEBUG MODE)'):
        refl = st.session_state.get('reflectance_stack')
        if refl is None:
            st.warning('❌ No reflectance data loaded. Carga primero el .npz.')
            return

        st.write(f"Reflectance shape: {refl.shape}")

        # Extracción de bandas
        if refl.ndim == 4:
            R1, R2 = refl[..., band1], refl[..., band2]
        elif refl.ndim == 3:
            R1, R2 = refl[..., band1], refl[..., band2]
        else:
            st.error(f"⚠️ Shape no soportado: {refl.shape}")
            return

        st.write(f"🔎 R1 shape: {R1.shape}, R2 shape: {R2.shape}")

        # Clipping y absorbancia
        eps = 1e-8
        R1c, R2c = np.clip(R1, eps, 1), np.clip(R2, eps, 1)
        A1, A2 = -np.log10(R1c), -np.log10(R2c)

        Abs = np.stack([A1, A2], axis=-1)
        E_inv = np.linalg.inv(E)
        conc = Abs @ E_inv.T

        HbO2, Hb = conc[..., 0], conc[..., 1]
        StO2 = HbO2 / (HbO2 + Hb + eps)

        st.write(f"StO2 shape: {StO2.shape}")
        st.write(f"StO2 stats: min {StO2.min()}, max {StO2.max()}, mean {StO2.mean()}")

        # Guardado
        if st.button('Save Beer–Lambert .npz'):
            out = easygui.filesavebox(msg="Guardar .npz", default="~/StO2_data.npz")
            if out:
                np.savez_compressed(out, StO2=StO2, HbO2=HbO2, Hb=Hb, tHb=HbO2+Hb, DeltaHb=HbO2-Hb)
                size = os.path.getsize(out)
                st.success(f"✅ Saved {out} ({size/1e3:.1f} kB)")

def viewer_npy():
    with st.expander('Viewer for StO2 and indexes'):
        # Validar que el archivo ya esté cargado
        if 'processed_data' not in st.session_state or 'processed_keys' not in st.session_state:
            st.warning("⚠️ No processed .npz file loaded yet. Please add it in the sidebar first.")
            return

        index_data = st.session_state['processed_data']
        keys = st.session_state['processed_keys']

        # Iniciar el estado persistente
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
        
        state = st.session_state['viewer_state']

        st.caption('Display')
        state['selected_index'] = st.selectbox('Select type', keys, index=keys.index(state['selected_index']))
        test = index_data[state['selected_index']]
        num_frames = test.shape[0]
        st.write(f"Number of frames in selected data: {num_frames}")

        state['image_selection'] = st.slider('Select image to display', 0, num_frames - 1, value=state['image_selection'], step=1)

        # Mask si es StO2 con tHb
        if state['selected_index'] == 'StO2' and {'HbO2','Hb'}.issubset(keys):
            state['mask_toggle'] = st.toggle("Mask low tHb pixels", value=state['mask_toggle'])
            state['threshold'] = st.slider("Threshold for tHb masking", 0.0, 0.1, value=state['threshold'], step=0.005)
            HbO2 = index_data['HbO2']
            Hb = index_data['Hb']
            tHb = HbO2 + Hb
            masked = np.where(
                tHb[state['image_selection']] < state['threshold'], np.nan, test[state['image_selection']]
            ) if state['mask_toggle'] else test[state['image_selection']]
        else:
            masked = test[state['image_selection']]

        # --- Corrige para stacks 1D, 2D, 3D, 4D ---
        # Si masked es 1D y el stack es (frames, width), intenta reshape a (H, W)
        if masked.ndim == 1:
            # Si el stack original es (N, W) y sabemos el ancho, intentar reshape
            if hasattr(test, "shape") and test.ndim == 2:
                width = test.shape[1]
                # Suponiendo cuadrado
                h = int(np.sqrt(width))
                if h * h == width:
                    img_to_show = masked.reshape((h, h))
                else:
                    st.warning(f"El stack seleccionado es 1D (shape={masked.shape}), no es visualizable como imagen.")
                    return
            else:
                st.warning(f"El stack seleccionado es 1D (shape={masked.shape}), no es visualizable como imagen.")
                return
        elif masked.ndim == 2:
            img_to_show = masked
        elif masked.ndim == 3:
            # Si es (frames, H, W), selecciona el frame
            if masked.shape[0] == num_frames:
                img_to_show = masked[state['image_selection']]
            else:
                img_to_show = masked[..., 0]
        else:
            st.warning(f"Forma de datos no soportada para visualización: {masked.shape}")
            return

        # Obtener imagen base
        if 'reflectance_stack' not in st.session_state:
            st.warning("No reflectance data loaded. Load your folder first.")
            return

        reflectance_stack = st.session_state['reflectance_stack']
        if reflectance_stack.ndim == 4:
            base_img = reflectance_stack[state['image_selection'], :, :, 0]
        elif reflectance_stack.ndim == 3:
            base_img = reflectance_stack[:, :, 0]
        else:
            st.warning("Reflectance stack tiene una forma no soportada.")
            return

        # Enhancements
        state['enhancement'] = st.radio(
            "Enhancement", ["None", "Brightness", "Auto contrast", "Stretch dynamic range"],
            index=["None", "Brightness", "Auto contrast", "Stretch dynamic range"].index(state['enhancement']),
            horizontal=True
        )

        if state['enhancement'] == "Brightness":
            state['brightness'] = st.slider("Adjust brightness", 0.1, 5.0, value=state['brightness'], step=0.1)
            enhanced_base = np.clip(base_img * state['brightness'], 0, 1)
        elif state['enhancement'] == "Auto contrast":
            enhanced_base = skimage.exposure.equalize_adapthist(base_img, clip_limit=0.03)
        elif state['enhancement'] == "Stretch dynamic range":
            enhanced_base = (base_img - base_img.min()) / (base_img.max() - base_img.min() + 1e-8)
        else:
            enhanced_base = base_img

        state['overlay_alpha'] = st.slider("Overlay alpha", 0.0, 1.0, value=state['overlay_alpha'], step=0.05)
        state['exposure_slider'] = st.slider("Exposure slider", 0.1, 5.0, value=state['exposure_slider'], step=0.1)

        # Mostrar imagen
        fig, ax = plt.subplots()
        ax.imshow(enhanced_base, cmap='gray')
        im = ax.imshow(img_to_show, cmap='coolwarm', alpha=state['overlay_alpha'], vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        ax.axis('off')
        with col2:
            st.pyplot(fig)

        # Export series
        if st.button("Export current overlay series as PNGs"):
            output_folder = easygui.diropenbox('Select output folder to save overlay series', default="/home/alonso/Desktop")
            if output_folder:
                timestamp = time.strftime("%Y%m%d-%H%M")
                new_folder = os.path.join(output_folder, f"{state['selected_index']}_OverlaySeries_{timestamp}")
                os.makedirs(new_folder, exist_ok=True)

                for i in range(num_frames):
                    fig2, ax2 = plt.subplots()
                    if reflectance_stack.ndim == 4:
                        base_img_exp = np.clip(reflectance_stack[i, :, :, 0] * state['exposure_slider'], 0, 1)
                    elif reflectance_stack.ndim == 3:
                        base_img_exp = np.clip(reflectance_stack[:, :, 0] * state['exposure_slider'], 0, 1)
                    else:
                        continue
                    ax2.imshow(base_img_exp, cmap='gray')
                    overlay_data = np.where(
                        tHb[i] < state['threshold'], np.nan, test[i]
                    ) if (state['mask_toggle'] and state['selected_index'] == 'StO2') else test[i]

                    im2 = ax2.imshow(
                        overlay_data, cmap='coolwarm', alpha=state['overlay_alpha'], vmin=0, vmax=1
                    )
                    ax2.axis('off')
                    fig2.colorbar(im2, ax=ax2, fraction=0.025, pad=0.01)
                    plt.savefig(
                        os.path.join(new_folder, f"{state['selected_index']}_overlay_{i}.png"),
                        dpi=300, bbox_inches='tight'
                    )
                    plt.close(fig2)
                
                st.success(f"✅ Saved overlays in: {new_folder}")
                st.rerun()


def compute_mean_in_tracked_rois(processed_stack, roi_tracks):
    height, width = processed_stack.shape[1:]
    data = []

    for roi in roi_tracks:
        name = roi['name']
        for (frame_id, x, y, w, h) in roi['coords']:
            y1, y2 = max(0,y), min(y+h,height)
            x1, x2 = max(0,x), min(x+w,width)
            roi_data = processed_stack[frame_id][y1:y2, x1:x2]

            if roi_data.size > 0:
                mean_val = np.nanmean(roi_data)
                if np.isnan(mean_val):
                    mean_val = None
            else:
                mean_val = None
            data.append({'frame': frame_id, 'roi_name': name, 'mean_value': mean_val})

    df = pd.DataFrame(data)
    return df

def tracking_roi_selector(original_stack, processed_stack, scale=3, output_video='tracking_output.avi'):
    with st.expander('ROI tracker'):
        if st.button("Select ROIs & Track"):
            rois_local = []

            def on_toggle():
                nonlocal use_global_percentile
                use_global_percentile = not use_global_percentile
                toggle_btn.config(text=f"Use global percentiles: {use_global_percentile}")

            use_global_percentile = True
            global_p1, global_p99 = np.nanpercentile(processed_stack, (1,99))

            img0 = original_stack[0][0]
            p1, p99 = np.percentile(img0, (1, 99))
            img0 = np.clip(img0, p1, p99)
            img0 = (255 * (img0 - p1) / (p99 - p1 + 1e-5)).astype(np.uint8)

            H, W = img0.shape
            pil_img = Image.fromarray(img0).resize((W*scale, H*scale))
            new_W, new_H = pil_img.size

            root = tk.Tk()
            root.title("Select ROIs on Image 0")

            canvas = tk.Canvas(root, width=new_W, height=new_H)
            canvas.pack()
            tk_img = ImageTk.PhotoImage(pil_img)
            canvas.create_image(0, 0, anchor='nw', image=tk_img)

            rect = None
            start = (0, 0)

            def on_press(evt):
                nonlocal rect, start
                start = (evt.x, evt.y)
                rect = canvas.create_rectangle(evt.x, evt.y, evt.x, evt.y, outline='green', width=2)

            def on_drag(evt):
                canvas.coords(rect, start[0], start[1], evt.x, evt.y)

            def on_release(evt):
                x0, y0 = start
                x1, y1 = evt.x, evt.y
                x, y = min(x0, x1), min(y0, y1)
                w, h = abs(x1 - x0), abs(y1 - y0)
                if w == 0 or h == 0:
                    return
                x, y, w, h = x//scale, y//scale, w//scale, h//scale
                name = simpledialog.askstring("ROI Name", "Name for this ROI:")
                if not name:
                    return
                rois_local.append({'name': name, 'rect': (x, y, w, h)})

            canvas.bind('<ButtonPress-1>', on_press)
            canvas.bind('<B1-Motion>', on_drag)
            canvas.bind('<ButtonRelease-1>', on_release)

            toggle_btn = tk.Button(root, text=f"Use global percentiles: {use_global_percentile}", command=on_toggle)
            toggle_btn.pack(pady=5)

            finish_btn = tk.Button(root, text="Finish Selection & Start Tracking", command=root.destroy)
            finish_btn.pack(pady=10)
            root.mainloop()

            # --- Tracking
            height, width = original_stack[0][0].shape
            roi_tracks = []

            for roi in rois_local:
                name = roi['name']
                x, y, w, h = roi['rect']
                template = original_stack[0][0, y:y+h, x:x+w].astype(np.float32)
                p1, p99 = np.percentile(template, (1,99))
                template = np.clip(template, p1, p99)
                template = (255 * (template - p1) / (p99 - p1 + 1e-5)).astype(np.uint8)

                coords_per_frame = []
                for i, frame in enumerate(original_stack):
                    img = frame[0].astype(np.float32)
                    p1, p99 = np.percentile(img, (1,99))
                    img_norm = np.clip(img, p1, p99)
                    img_norm = (255 * (img_norm - p1) / (p99 - p1 + 1e-5)).astype(np.uint8)

                    res = cv2.matchTemplate(img_norm, template, cv2.TM_CCOEFF_NORMED)
                    _, _, _, max_loc = cv2.minMaxLoc(res)
                    x_new, y_new = max_loc
                    coords_per_frame.append( (i, x_new, y_new, w, h) )

                roi_tracks.append({'name': name, 'coords': coords_per_frame})

            # --- Generar video
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_name = f"{'global' if use_global_percentile else 'fixed'}_{output_video}"
            out = cv2.VideoWriter(video_name, fourcc, 10, (width, height))

            for i in range(len(original_stack)):
                img = original_stack[i][0].astype(np.float32)
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

            # --- Guardar
            st.session_state["roi_tracks"] = roi_tracks
            st.session_state["video_file"] = video_name

            # --- Calcular y exportar mean automáticamente
            df = compute_mean_in_tracked_rois(processed_stack, roi_tracks)
            st.write(df)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV of mean ROI values", csv, "mean_roi_values.csv", "text/csv")

            # --- Descargar video
            with open(video_name, 'rb') as f:
                st.download_button("📥 Download tracking video", f, file_name=os.path.basename(video_name), mime="video/avi")

def compute_mean_in_tracked_rois(processed_stack, roi_tracks):
    height, width = processed_stack.shape[1:]
    data = []
    file_names = p_files_names

    for roi in roi_tracks:
        name = roi['name']
        for (frame_id, x, y, w, h) in roi['coords']:
            y1, y2 = max(0, y), min(y + h, height)
            x1, x2 = max(0, x), min(x + w, width)
            roi_data = processed_stack[frame_id][y1:y2, x1:x2]

            # Debug
            print(f"Frame {frame_id} ({file_names[frame_id]}), ROI {name}: bounds=({y1}:{y2}, {x1}:{x2}), shape={roi_data.shape}")
            if roi_data.size > 0:
                print(f"ROI data sample: {roi_data.flatten()[:5]}")

            if roi_data.size > 0:
                mean_val = np.nanmean(roi_data)
                if np.isnan(mean_val):
                    mean_val = None
            else:
                mean_val = None

            data.append({'frame': file_names[frame_id], 'roi_name': name, 'mean_value': mean_val})

    df = pd.DataFrame(data)

    # 🚀 Pivot para que cada ROI sea columna
    df_wide = df.pivot(index='frame', columns='roi_name', values='mean_value').reset_index()
    return df_wide



col1,col2,col3=st.columns([1,1,0.5])

reflectance_stack,original_stack,processed_stack,p_files_names=folder_path_acquisition()
with col2:
    st.caption('Viewers')
with col1:
    beer_lambert_sat_calculations()
    viewer_npy()
    tracking_roi_selector(original_stack,processed_stack)
with col3:
    st.write('Log')
    st.write(p_files_names)