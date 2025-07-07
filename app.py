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
from tkinter import simpledialog
from PIL import Image, ImageTk
import cv2


st.set_page_config(layout="wide")

def header_agsantos():
    iol1, iol2 = st.columns([4, 1])
    with iol1:
        st.title("Spectral analysis of Ximea cameras")
        st.caption('AG-Santos Neurovascular research Lab')
    with iol2:
        st.image('images/agsantos.png', width=130)
    st.markdown("_______________________")
header_agsantos()


def folder_path_acquisition():
    # ---- Reflectance files (.npy)
    with st.sidebar.expander('Add reflectance files'):
        st.caption('Add folder with .npy reflectance files')
        add_data_button = st.button('Add folder data path')
        if add_data_button:
            folder_path = easygui.diropenbox(
                msg="Select folder with reflectance files (.npy)",
                default="/home/alonso/Desktop"
            )
            if folder_path:
                st.session_state['folder_path'] = folder_path

                file_list = sorted([f for f in os.listdir(folder_path) if f.endswith(".npy")])
                arrays = [np.load(os.path.join(folder_path, f)) for f in file_list]
                stack = np.stack(arrays, axis=0)
                st.session_state['reflectance_stack'] = stack

                st.success(f"Loaded reflectance stack: {stack.shape}")

        # Mostrar estado actual siempre
        folder_path = st.session_state.get('folder_path', None)
        if folder_path:
            st.caption(f"✅ Reflectance folder: {folder_path}")
            st.caption(f"Stack shape: {st.session_state['reflectance_stack'].shape}")

    # ---- Original images (.tiff)
    with st.sidebar.expander('Add original TIFF images folder'):
        if st.button("Add folder with .tiff files"):
            original_folder_path = easygui.diropenbox('Select folder with original .tiff images', default="/home/alonso/Desktop/")
            if original_folder_path:
                st.session_state['original_folder_path'] = original_folder_path

                original_files_list = sorted([f for f in os.listdir(original_folder_path) if f.lower().endswith(('.tiff', '.tif'))])
                original_stack = np.stack([tiff.imread(os.path.join(original_folder_path, f)) for f in original_files_list], axis=0)
                st.session_state['original_stack'] = original_stack

                st.success(f"Loaded original stack: {original_stack.shape}")

        # Mostrar estado actual siempre
        if 'original_folder_path' in st.session_state:
            st.caption(f"✅ Original images folder: {st.session_state['original_folder_path']}")
            st.caption(f"Original stack shape: {st.session_state['original_stack'].shape}")

    # ---- Processed indices (.npy / .npz)
    with st.sidebar.expander('Add processed indices folder'):
        if st.button("Add folder with processed files"):
            processed_folder_path = easygui.diropenbox('Select folder with processed index files', default="/home/alonso/Desktop/")
            if processed_folder_path:
                st.session_state['processed_folder_path'] = processed_folder_path

                processed_files_list = sorted([f for f in os.listdir(processed_folder_path) if f.endswith((".npy", ".npz"))])
                st.session_state['processed_files_list'] = processed_files_list

                if processed_files_list:
                    first_file = processed_files_list[0]
                    path_file = os.path.join(processed_folder_path, first_file)
                    if first_file.endswith(".npz"):
                        data = np.load(path_file)
                        keys = data.files
                        st.caption(f"First .npz contains: {keys}")
                    else:
                        data = np.load(path_file)
                        st.caption(f"First .npy shape: {data.shape}")

                    st.success(f"Loaded processed folder with {len(processed_files_list)} files")

        # Mostrar estado actual siempre
        if 'processed_folder_path' in st.session_state:
            st.caption(f"✅ Processed folder: {st.session_state['processed_folder_path']}")
            st.caption(f"Files: {len(st.session_state.get('processed_files_list', []))} files loaded")

    # Retornar los stacks y paths
    reflectance_stack = st.session_state.get('reflectance_stack', None)
    original_stack = st.session_state.get('original_stack', None)
    processed_files_list = st.session_state.get('processed_files_list', [])

    return folder_path, reflectance_stack, original_stack, processed_files_list    # ---- Reflectance files (.npy)
    with st.sidebar.expander('Add reflectance files'):
        st.caption('Add folder with .npy reflectance files')
        add_data_button = st.button('Add folder data path')
        if add_data_button:
            folder_path = easygui.diropenbox(
                msg="Select folder with reflectance files (.npy)",
                default="/home/alonso/Desktop"
            )
            st.session_state['folder_path'] = folder_path

            file_list = sorted([f for f in os.listdir(folder_path) if f.endswith(".npy")])
            st.caption(f"Found {len(file_list)} .npy files")

            arrays = [np.load(os.path.join(folder_path, f)) for f in file_list]
            stack = np.stack(arrays, axis=0)
            st.session_state['reflectance_stack'] = stack

            st.success(f"Reflectance stack shape: {stack.shape}")

        folder_path = st.session_state.get('folder_path', '')
        stack = st.session_state.get('reflectance_stack', None)
        if folder_path:
            st.caption(f"✅ Selected folder: {folder_path}")

    # ---- Original images (.tiff)
    with st.sidebar.expander('Add original TIFF images folder'):
        if st.button ("Add folder with .tiff files"):
            original_folder_path = easygui.diropenbox('Select folder with original .tiff images', default="/home/alonso/Desktop/")
            st.session_state['original_folder_path'] = original_folder_path

            original_files_list = sorted([f for f in os.listdir(original_folder_path) if f.lower().endswith(('.tiff', '.tif'))])
            st.caption(f"Found {len(original_files_list)} TIFF files")
            
            # Load stack
            original_stack = np.stack([tiff.imread(os.path.join(original_folder_path, f)) for f in original_files_list], axis=0)
            st.session_state['original_stack'] = original_stack
            st.success(f"Original image stack shape: {original_stack.shape}")

    # ---- Processed indices folder (.npy or .npz)
    with st.sidebar.expander('Add processed indices folder'):
        if st.button ("Add folder with processed files"):
            processed_folder_path = easygui.diropenbox('Select folder with processed index files', default="/home/alonso/Desktop/")
            st.session_state['processed_folder_path'] = processed_folder_path

            processed_files_list = sorted([f for f in os.listdir(processed_folder_path) if f.endswith((".npy", ".npz"))])
            st.caption(f"Found {len(processed_files_list)} processed files")

            # Example: load first file to show shape
            first_file = processed_files_list[0]
            if first_file.endswith(".npz"):
                data = np.load(os.path.join(processed_folder_path, first_file))
                keys = data.files
                st.caption(f"First .npz contains: {keys}")
            else:
                data = np.load(os.path.join(processed_folder_path, first_file))
                st.caption(f"First .npy shape: {data.shape}")

            st.session_state['processed_files_list'] = processed_files_list

    # Always return from session_state
    original_stack = st.session_state.get('original_stack', None)
    processed_files_list = st.session_state.get('processed_files_list', [])

    return folder_path, stack, original_stack, processed_files_list
folder_path, reflectance_stack, original_stack, processed_files_list = folder_path_acquisition()

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
λ1, λ2, eps_hbo2_λ1, eps_hb_λ1, eps_hbo2_λ2, eps_hb_λ2,E,band1,band2 = band_selection()

def beer_lambert_sat_calculations():
    with st.expander('Beer-Lambert StO2 calculations'):
        absorbance_help_str = 'Absorbance measures how much light is absorbed by the tissue'
        st.latex('A(λ)=−log10 (R(λ))', help=absorbance_help_str)

        if reflectance_stack is None:
            st.warning('❌ No reflectance data loaded. Please add folder path with .npy files first.')
            return

        st.caption(f"✅ Loaded reflectance stack shape: {reflectance_stack.shape}")
        
        # Procesar solo si el stack existe
        R1 = reflectance_stack[:, :, :, band1]
        R2 = reflectance_stack[:, :, :, band2]
        st.write(f"HBO2 band {band1} shape: {R1.shape}")
        st.write(f"HB band {band2} shape: {R2.shape}")

        # Clipping para evitar log(0)
        epsilon_clip = 1e-8
        R1 = np.clip(R1, epsilon_clip, 1)
        R2 = np.clip(R2, epsilon_clip, 1)

        # Absorbance
        A1 = -np.log10(R1)
        A2 = -np.log10(R2)

        # Inverse matrix
        E_inv = np.linalg.inv(E)
        Abs = np.stack([A1, A2], axis=-1)
        st.write(f"Absorbance images shape: {Abs.shape} (last dim 0=HbO2, 1=Hb)")

        # Concentrations
        conc = Abs @ E_inv.T
        HbO2 = conc[:, :, :, 0]
        Hb   = conc[:, :, :, 1]
        tHb = HbO2 + Hb
        deltaHb = HbO2 - Hb

        # Saturation
        epsilon = 1e-8  # para evitar divisiones por cero
        StO2 = HbO2 / (tHb + epsilon)
        st.session_state['StO2_stack'] = StO2

        st.latex(r"StO_2 = \frac{[HbO_2]}{[HbO_2] + [Hb]}")
        st.write(f"StO2 shape: {StO2.shape}")

        with st.popover('StO2 image pixel values'):
            st.write(StO2[0, :, :])

        # Guardar como stack único .npz
        if st.button('Save all as single .npz file'):
            output_file = easygui.filesavebox(
                msg="Save compressed .npz file",
                default="/home/alonso/Desktop/StO2_data.npz"
            )
            if output_file:
                np.savez_compressed(
                    output_file,
                    StO2=StO2,
                    HbO2=HbO2,
                    Hb=Hb,
                    tHb=tHb,
                    DeltaHb=deltaHb
                )
                st.success(f"Saved as compressed stack at: {output_file}")
def viewer_npy():
    with st.expander('Viewer for StO2 and indexes'):     
        if st.button('Select file .npz'):
            index_file_path = easygui.fileopenbox(
                'Select file with .npz index files',
                default="/home/alonso/Desktop/"
            )
            if index_file_path:
                st.session_state['index_file_path'] = index_file_path
                st.session_state['index_data'] = np.load(index_file_path)
        
        if 'index_data' in st.session_state:
            index_data = st.session_state['index_data']
            st.write(index_data.files)
        else:
            st.empty()
            return

    with col3:
        st.caption('Display')
        selected_index = st.selectbox('Select type', index_data.files)
        num_frames = index_data[selected_index].shape[0]
        st.write(f"Number of frames in selected data: {num_frames}")
        
        test = index_data[selected_index]
        image_selection = st.slider('Select image to display', 0, num_frames - 1, step=1)

        # Threshold si hay tHb
        mask_toggle = False
        threshold = 0.0
        if selected_index == 'StO2' and {'HbO2','Hb'}.issubset(index_data.files):
            mask_toggle = st.toggle("Mask low tHb pixels", value=True)
            threshold = st.slider("Threshold for tHb masking", 0.0, 0.1, 0.01, 0.005)
            HbO2 = index_data['HbO2']
            Hb = index_data['Hb']
            tHb = HbO2 + Hb
            if mask_toggle:
                masked = np.where(tHb[image_selection] < threshold, np.nan, test[image_selection])
            else:
                masked = test[image_selection]
        else:
            masked = test[image_selection]

        # Obtener imagen base
        if 'reflectance_stack' not in st.session_state:
            st.warning("No reflectance data loaded. Load your folder first.")
            return

        reflectance_stack = st.session_state['reflectance_stack']
        base_img = reflectance_stack[image_selection, :, :, 0]

        # Sliders para enhancements
        option = st.radio("Enhancement", ["None", "Brightness", "Auto contrast", "Stretch dynamic range"], horizontal=True)
        if option == "Brightness":
            brightness_factor = st.slider("Adjust brightness", 0.1, 5.0, 1.0, 0.1)
            enhanced_base = np.clip(base_img * brightness_factor, 0, 1)
        elif option == "Auto contrast":
            enhanced_base = skimage.exposure.equalize_adapthist(base_img, clip_limit=0.03)
        elif option == "Stretch dynamic range":
            enhanced_base = (base_img - base_img.min()) / (base_img.max() - base_img.min() + 1e-8)
        else:
            enhanced_base = base_img

        overlay_alpha = st.slider("Overlay alpha", 0.0, 1.0, 0.6, 0.05)
        exposure_slider = st.slider("Exposure slider", 0.1, 5.0, 1.0, 0.1)

        # Mostrar imagen actual
        fig, ax = plt.subplots()
        ax.imshow(enhanced_base, cmap='gray')
        im = ax.imshow(masked, cmap='coolwarm', alpha=overlay_alpha, vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        ax.axis('off')
        st.pyplot(fig)

        # Export series a folder único con timestamp
        if st.button("Export current overlay series as PNGs"):
            output_folder = easygui.diropenbox(
                'Select output folder to save overlay series',
                default="/home/alonso/Desktop"
            )
            if output_folder:
                # Crear subfolder único con fecha-hora
                timestamp = time.strftime("%Y%m%d-%H%M")
                new_folder = os.path.join(output_folder, f"{selected_index}_OverlaySeries_{timestamp}")
                os.makedirs(new_folder, exist_ok=True)

                # Guardar cada imagen en el subfolder
                for i in range(num_frames):
                    fig2, ax2 = plt.subplots()
                    ax2.imshow(
                        np.clip(reflectance_stack[i, :, :, 0] * exposure_slider, 0, 1),
                        cmap='gray'
                    )
                    overlay_data = np.where(
                        tHb[i] < threshold, np.nan, test[i]
                    ) if (mask_toggle and selected_index == 'StO2') else test[i]

                    im2 = ax2.imshow(
                        overlay_data,
                        cmap='coolwarm',
                        alpha=overlay_alpha,
                        vmin=0, vmax=1
                    )
                    ax2.axis('off')
                    fig2.colorbar(im2, ax=ax2, fraction=0.025, pad=0.01)
                    plt.savefig(
                        os.path.join(new_folder, f"{selected_index}_overlay_{i}.png"),
                        dpi=300, bbox_inches='tight'
                    )
                    plt.close(fig2)
                
                st.success(f"✅ Saved overlays in: {new_folder}")
#tracking and selection of ROIs


def run_tracking_on_stacks_in_streamlit():
    import streamlit as st
    import tkinter as tk
    from tkinter import simpledialog
    from PIL import Image, ImageTk, ImageDraw
    import numpy as np
    import cv2
    import os
    import time
    import pandas as pd
    import easygui

    # ---- Cargar datos desde el session_state
    original_stack = st.session_state.get('original_stack', None)
    processed_folder_path = st.session_state.get('processed_folder_path', None)
    processed_files_list = st.session_state.get('processed_files_list', [])

    if original_stack is None or processed_folder_path is None or not processed_files_list:
        st.warning("Please load both the original TIFF stack and the processed NPZ files first.")
        return

    # ---- Cargar índice desde primer npz
    npz_path = os.path.join(processed_folder_path, processed_files_list[0])
    npz_data = np.load(npz_path)
    index_key = "StO2"
    index_stack = npz_data[index_key]

    # ---- Extraer canal 0
    if original_stack.ndim != 4 or original_stack.shape[1] != 16:
        st.error(f"Expected shape (T,16,H,W), got {original_stack.shape}")
        return
    first_frame = original_stack[0,0,:,:]
    H, W = first_frame.shape
    first_frame_disp = cv2.normalize(first_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # ---- ROI selector con nombres
    root = tk.Tk()
    root.withdraw()
    win = tk.Toplevel()
    win.title("Select ROIs - drag and name")
    scale_factor = 1.5
    new_H, new_W = int(H*scale_factor), int(W*scale_factor)
    img = Image.fromarray(first_frame_disp).resize((new_W, new_H))
    tk_img = ImageTk.PhotoImage(img)
    canvas = tk.Canvas(win, width=new_W, height=new_H)
    canvas.pack()
    canvas_img = canvas.create_image(0, 0, anchor='nw', image=tk_img)

    rect, start = None, (0,0)
    roi_boxes = []

    def on_press(evt):
        nonlocal rect, start
        start = (evt.x, evt.y)
        rect = canvas.create_rectangle(evt.x, evt.y, evt.x, evt.y, outline='green', width=2)

    def on_drag(evt):
        canvas.coords(rect, start[0], start[1], evt.x, evt.y)

    def on_release(evt):
        x0,y0 = start
        x1,y1 = evt.x, evt.y
        x,y = min(x0,x1), min(y0,y1)
        w,h = abs(x1-x0), abs(y1-y0)
        if w > 5 and h > 5:
            name = simpledialog.askstring("ROI Name", "Name for this ROI:")
            if name:
                roi_boxes.append({'name': name, 'rect':[x,y,w,h]})
                canvas.create_text(x,y-10, text=name, fill='green', font=('Helvetica',12))
                canvas.create_rectangle(x, y, x+w, y+h, outline='green', width=2)
        canvas.delete(rect)

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    tk.Button(win, text="Done", command=win.destroy).pack()
    root.mainloop()

    # ---- Escalar ROIs de regreso
    for roi in roi_boxes:
        x,y,w,h = roi['rect']
        roi['rect'] = [int(x/scale_factor), int(y/scale_factor), int(w/scale_factor), int(h/scale_factor)]

    if not roi_boxes:
        st.warning("❌ No ROIs selected.")
        return

    # ---- Elegir carpeta de salida
    output_folder = easygui.diropenbox("Select output folder")
    if not output_folder:
        st.warning("Cancelled output folder.")
        return

    # ---- Inicializar tracking
    lk_params = dict(winSize=(15,15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    p0_list = []
    for roi in roi_boxes:
        x,y,w,h = roi['rect']
        pts = cv2.goodFeaturesToTrack(first_frame_disp[y:y+h,x:x+w], maxCorners=20, qualityLevel=0.01, minDistance=2)
        if pts is not None:
            pts[:,0,0] += x
            pts[:,0,1] += y
        p0_list.append(pts)

    prev_gray = first_frame_disp.copy()
    timestamp = time.strftime("%Y%m%d-%H%M")
    out_video_path = os.path.join(output_folder, f"TrackingOverlay_{timestamp}.mp4")
    out_csv_path = os.path.join(output_folder, f"TrackingData_{timestamp}.csv")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video_path, fourcc, 5, (W,H))
    records = []

    progress = st.progress(0, text="Starting tracking...")

    for t in range(1, len(original_stack)):
        curr_frame = original_stack[t,0,:,:]
        curr_disp = cv2.normalize(curr_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        overlay = cv2.cvtColor(curr_disp, cv2.COLOR_GRAY2BGR)

        mask_frame = index_stack[t]
        mask_norm = cv2.normalize(mask_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        mask_color = cv2.applyColorMap(mask_norm, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(overlay, 0.7, mask_color, 0.3, 0)

        st.write(f"📌 Frame {t}/{len(original_stack)-1}")
        for idx, (pts, roi) in enumerate(zip(p0_list, roi_boxes)):
            x,y,w,h = roi['rect']
            if pts is not None and len(pts):
                p1, st_, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_disp, pts, None, **lk_params)
                good_new = p1[st_==1]
                good_old = pts[st_==1]

                st.write(f"  ROI {roi['name']}: {len(good_new)} good points")
                if len(good_new) >= 3:
                    dx, dy = np.mean(good_new - good_old, axis=0)
                    x, y = int(np.clip(x+dx,0,W-1)), int(np.clip(y+dy,0,H-1))
                    roi['rect'][:2] = [x,y]
                    p0_list[idx] = good_new.reshape(-1,1,2)
                    mean_val = np.nanmean(mask_frame[y:y+h,x:x+w])
                    records.append({"Frame": t, "ROI": roi['name'], "MeanStO2": mean_val})
                    cv2.rectangle(overlay, (x,y), (x+w,y+h), (0,255,0), 2)
                    cv2.putText(overlay, roi['name'], (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                else:
                    st.write(f"  ⚠️ Not enough points for {roi['name']}")
            else:
                st.write(f"  🚫 No points init for {roi['name']}")

        out.write(overlay)
        prev_gray = curr_disp.copy()
        progress.progress(t/len(original_stack), text=f"Processing frame {t}/{len(original_stack)}")

    out.release()
    progress.progress(1.0, text="Done ✅")

    # ---- Guardar CSV
    df = pd.DataFrame(records)
    df.to_csv(out_csv_path, index=False)
    st.success(f"✅ Video & CSV saved at {output_folder}")
    with open(out_video_path, "rb") as f:
        st.download_button("Download tracking video", f, file_name=os.path.basename(out_video_path))


col1,col2,col3=st.columns([1,0.5,1])
with col1:
    beer_lambert_sat_calculations()
with col2:
    viewer_npy()

            

