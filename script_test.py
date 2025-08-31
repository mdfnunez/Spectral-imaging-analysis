def viewer_npy():
    with st.expander('Viewer for StO2 and indexes'):
        # Validar que el archivo ya esté cargado
        if 'processed_stack' not in st.session_state:
            st.warning("⚠️ No processed .npy file loaded yet. Please add it in the sidebar first.")
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
        if 'processed_stack' not in st.session_state:
            st.warning("No reflectance data loaded. Load your folder first.")
            return

        processed_stack = st.session_state['processed_stack']
        if processed_stack.ndim == 4:
            base_img = processed_stack[state['image_selection'], :, :, 0]
        elif processed_stack.ndim == 3:
            base_img = processed_stack[:, :, 0]
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
                    if processed_stack.ndim == 4:
                        base_img_exp = np.clip(processed_stack[i, :, :, 0] * state['exposure_slider'], 0, 1)
                    elif processed_stack.ndim == 3:
                        base_img_exp = np.clip(processed_stack[:, :, 0] * state['exposure_slider'], 0, 1)
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
            df = compute_mean_in_tracked_rois(processed_stack, roi_tracks,log_map)
            st.write(df)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV of mean ROI values", csv, "mean_roi_values.csv", "text/csv")

            # --- Descargar video
            with open(video_name, 'rb') as f:
                st.download_button("📥 Download tracking video", f, file_name=os.path.basename(video_name), mime="video/avi")



def compute_mean_in_tracked_rois(processed_stack, roi_tracks, log_map):
    height, width = processed_stack.shape[1:]
    data = []
    file_names = p_files_names
    sorted_ts = sorted(log_map.keys())

    for roi in roi_tracks:
        name = roi['name']
        for (frame_id, x, y, w, h) in roi['coords']:
            y1, y2 = max(0, y), min(y + h, height)
            x1, x2 = max(0, x), min(x + w, width)
            roi_data = processed_stack[frame_id][y1:y2, x1:x2]

            if roi_data.size > 0:
                mean_val = np.nanmean(roi_data)
                if np.isnan(mean_val):
                    mean_val = None
            else:
                mean_val = None

            ts = file_names[frame_id].split("_imagen")[0]

            # buscar evento más reciente <= ts
            idx = bisect_right(sorted_ts, ts) - 1
            if idx >= 0:
                nearest_ts = sorted_ts[idx]
                log_text = log_map[nearest_ts]
            else:
                log_text = ""

            data.append({
                'frame': file_names[frame_id],
                'log_event': log_text,
                'roi_name': name,
                'mean_value': mean_val
            })

    df = pd.DataFrame(data)
    df_wide = df.pivot(index=['frame', 'log_event'], columns='roi_name', values='mean_value').reset_index()
    return df_wide





col1,col2,col3=st.columns([1,1,0.5])

processed_stack,original_stack,processed_stack,p_files_names,log_map,file_names_reflectance=folder_path_acquisition()
with col2:
    st.caption('Viewers')
with col1:
    beer_lambert_sat_calculations()
    viewer_npy()
    tracking_roi_selector(original_stack,processed_stack)
with col3:
    st.write('Log')
    try:
        df_names_reflectance=pd.DataFrame({'Reflectance frame names':file_names_reflectance})
        st.dataframe(df_names_reflectance,hide_index=True)
        df_files = pd.DataFrame({"frame names": p_files_names})
        st.dataframe(df_files, use_container_width=True, hide_index=True)
        df_log = pd.DataFrame(list(log_map.items()), columns=["Timestamp", "log_event"])
        st.dataframe(df_log, use_container_width=True,hide_index=True)

    except:
        st.info('Nothing to show')

st.write(file_names_reflectance)