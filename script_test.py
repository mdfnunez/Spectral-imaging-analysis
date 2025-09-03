import PyQt6
def tracking_roi_selector(tiff_path, processed_stack, scale=3, output_video='tracking_output.avi'):
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
                rect = canvas.create_rectangle(evt.x, evt.y, evt.x, evt.y, outline='red', width=2)
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
                                              )
            st.write(df)
            st.download_button("📥 Download CSV of mean ROI values",
                               df.to_csv(index=False).encode('utf-8'),
                               "mean_roi_values.csv", "text/csv")

            # Descargar video
            with open(video_name, 'rb') as f:
                st.download_button("📥 Download tracking video", f,
                                   file_name=os.path.basename(video_name), mime="video/avi")

def compute_mean_in_tracked_rois(processed_stack, roi_tracks):
    height, width = processed_stack.shape[1:]
    rows = []

    for roi in roi_tracks:
        name = roi['name']
        for (frame_id, x, y, w, h) in roi['coords']:
            y1, y2 = max(0, y), min(y + h, height)
            x1, x2 = max(0, x), min(x + w, width)
            roi_data = processed_stack[frame_id][y1:y2, x1:x2]
            mean_val = float(np.nanmean(roi_data)) if roi_data.size > 0 else None
            if np.isnan(mean_val): mean_val = None


    df = pd.DataFrame(rows)
    return df.pivot(index=['frame', 'log_event'], columns='roi_name', values='mean_value').reset_index()
