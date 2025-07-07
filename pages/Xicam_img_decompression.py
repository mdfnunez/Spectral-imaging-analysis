import getpass
import numpy as np
import blosc2
import tifffile
from datetime import datetime
import streamlit as st
import os, re, unicodedata
from typing import Optional, Dict
import tkinter as tk
from tkinter import filedialog

col1,col2,col3=st.columns(3)
# ────────────────────────────────
# utilidades
# ────────────────────────────────
def demosaic(image: np.ndarray, mosaic_size: int = 4) -> np.ndarray:
    """Convierte mosaico 2D → cubo (rows, cols, m² bandas)."""
    rows, cols = image.shape[0] // mosaic_size, image.shape[1] // mosaic_size
    bands = mosaic_size * mosaic_size
    out = np.zeros((rows, cols, bands), dtype=image.dtype)
    k = 0
    for r in range(mosaic_size):
        for c in range(mosaic_size):
            out[:, :, k] = image[r::mosaic_size, c::mosaic_size]
            k += 1
    return out


def _slugify(txt: str) -> str:
    txt = unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode()
    txt = re.sub(r"[^0-9A-Za-z]+", "_", txt).strip("_")
    return txt[:60]


def _parse_log(path: str) -> Dict[str, str]:
    m: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                ts, msg = ln.split(" ", 1)
                m[ts.strip()] = msg.strip()
    return m


# ────────────────────────────────
# SAVE IMAGES AS MULTI-CHANNEL TIFF
# ────────────────────────────────
def save_images_to_tiff(
    b2nd_path: str,
    output_folder: str,
    grayscale: bool = False,
    mosaic_size: int = 4,
    band_idx: int = 0,
    log_path: Optional[str] = None,
) -> None:
    """
    Guarda un único TIFF contiguo con N canales.

    - Cada pixel lleva todos los canales (shape -> (rows, cols, bands)).
    - Admite uint8/uint12/uint16/float32  y activa BigTIFF si hace falta.
    """
    log_map = _parse_log(log_path) if log_path and os.path.isfile(log_path) else {}

    data = blosc2.open(b2nd_path, mode="r")
    os.makedirs(output_folder, exist_ok=True)
    print(f"Total de cuadros: {data.shape[0]}")

    for i in range(data.shape[0]):
        img = data[i][...]
        try:
            ts = data.schunk.vlmeta["time_stamp"][i]
        except (AttributeError, KeyError, IndexError):
            ts = "N-A"

        log_msg = log_map.get(ts, "")
        core = f"{ts}_{_slugify(log_msg)}" if log_msg else ts

        # ── reorganizar bandas ─────────────────────────────
        if grayscale:
            if img.ndim == 3:
                img = img[:, :, band_idx]
            # (rows, cols, 1) para que siga siendo contiguo
            img = img[..., np.newaxis]
        else:
            if img.ndim == 2:                    # mosaico crudo
                img = demosaic(img, mosaic_size)  # (rows, cols, bands)
            # si viene (bands, rows, cols) → contiguo
            if img.ndim == 3 and img.shape[0] < img.shape[-1]:
                img = np.transpose(img, (1, 2, 0))

        # ── tipo de datos sin re-escalar ──────────────────
        if np.issubdtype(img.dtype, np.integer):
            bits = img.dtype.itemsize * 8
            if bits <= 16:
                img = img.astype(np.uint16) << (16 - bits)
            else:
                img = np.clip(img, 0, 65535).astype(np.uint16)
        elif np.issubdtype(img.dtype, np.floating):
            img = img.astype(np.float32)
        else:
            raise TypeError(f"Tipo no soportado: {img.dtype}")

        # ── escribir TIFF contiguo ────────────────────────
        out = os.path.join(output_folder, f"{core}_frame_{i}.tif")
        tifffile.imwrite(
            out,
            img,                       # (rows, cols, N)
            bigtiff=True,
            photometric="minisblack",  # no RGB ⇒ escala de grises por canal
            planarconfig="CONTIG",     # <— cada pixel contiene los N valores
            metadata={"time_stamp": str(ts), "log_msg": log_msg},
        )
        print("Guardado:", out)

    print("✅ Descompresión y guardado completados.")


def pick_folder_tk():
    
    # Intentamos /media/<usuario> primero
    username = getpass.getuser()
    media_path = os.path.join("/media", username)  # /media/tu_usuario

    # Si aún no existe, fallback al HOME
    if not os.path.exists(media_path):
        media_path = os.path.expanduser("~")

    root = tk.Tk()
    root.withdraw()  # Oculta la ventana principal
    folder_path = filedialog.askdirectory(
        initialdir=media_path,
        title="Selecciona la carpeta de salida (discos externos suelen estar en /media)"
    )
    root.destroy()
    return folder_path

def main():
    with col1:
            st.subheader("Decompression of .b2nd files")
            option = st.selectbox("Proveer el archivo .b2nd", ["Subir archivo", "Ruta manual"])
            if option == "Subir archivo":
                uploaded_file = st.file_uploader("Sube tu archivo .b2nd", type=["b2nd"])
                b2nd_path = None
            else:
                b2nd_path = st.text_input("Ruta al archivo .b2nd")
                uploaded_file = None
                if b2nd_path and not os.path.isfile(b2nd_path):
                    st.error("¡La ruta especificada no es válida!")
    with col2:
        st.caption('Select the folder to save the files')
        # 2) Botón para seleccionar carpeta con Tkinter (mostrando discos externos)
        if st.button("Output folder"):
            selected_folder = pick_folder_tk()
            if selected_folder:
                st.session_state["selected_folder"] = selected_folder
                st.success(f"Carpeta seleccionada: {selected_folder}")
            else:
                st.warning("No se seleccionó ninguna carpeta (cerraste el diálogo).")


        subfolder_name = st.text_input("Name of subfolder", "")

        # 5) Tipo de cámara
        camera_type = st.selectbox("Tipo de cámara", ["Multiespectral", "Grayscale"])
        grayscale = (camera_type == "Grayscale")

        # 6) Botón final para descomprimir
        if st.button("Descomprimir y Guardar Imágenes"):
            # Verificar archivo
            if uploaded_file is not None or (b2nd_path and os.path.isfile(b2nd_path)):
                # Si se subió archivo, guardarlo temporalmente
                if uploaded_file is not None:
                    with open("uploaded_temp.b2nd", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    b2nd_path = "uploaded_temp.b2nd"

                # Resolver la carpeta de salida
                if "selected_folder" in st.session_state and st.session_state["selected_folder"]:
                    output_folder = st.session_state["selected_folder"]
                elif manual_output.strip():
                    output_folder = manual_output.strip()
                else:
                    # Carpeta por defecto: data + fecha/hora
                    output_folder = os.path.join(
                        "data",
                        datetime.now().strftime("%Y%m%d_%H%M%S")
                    )

                # Si hay subcarpeta, la anexamos
                if subfolder_name.strip():
                    output_folder = os.path.join(output_folder, subfolder_name.strip())

                os.makedirs(output_folder, exist_ok=True)

                with st.spinner("Descomprimiendo..."):
                    save_images_to_tiff(b2nd_path, output_folder, grayscale=grayscale)

                st.success(f"¡Imágenes guardadas en {output_folder}!")
            else:
                st.error("Por favor, especifica o sube un archivo .b2nd válido.")
    with col3:   
        with st.expander('Viewer for .tiff files'):
            st.caption("Visor rápido de TIFF multibanda")

            # 1️⃣ ––– Uploader
            uploaded = st.file_uploader("Sube tu archivo .tif/.tiff", type=["tif", "tiff"])

            if uploaded:
                # Guarda a disco para que tifffile lo abra
                tmp_path = "tmp_uploaded.tif"
                with open(tmp_path, "wb") as f:
                    f.write(uploaded.getbuffer())

                # 2️⃣ ––– Leer TIFF y mostrar metadatos básicos
                im = tf.imread(tmp_path)
                st.write(f"**Shape**: {im.shape}   |   **dtype**: {im.dtype}")

                # ¿Bandas en primer o último eje?
                if im.ndim == 2:  # solo una banda
                    st.image(im, caption="Imagen monobanda")
                    os.remove(tmp_path)
                    st.stop()

                # Reorientar a (rows, cols, bands) para visualizar fácil
                if im.shape[0] <= 32:             # Heurística: eje 0 = bandas
                    im = np.transpose(im, (1, 2, 0))  # (rows, cols, bands)

                num_bands = im.shape[2]
                st.write(f"El archivo contiene **{num_bands}** bandas.")

                # 3️⃣ ––– Selector de bandas para RGB
                default = [0, 1, 2] if num_bands >= 3 else list(range(num_bands))
                sel = st.multiselect(
                    "Elige 3 bandas para componer un RGB",
                    options=list(range(num_bands)),
                    default=default,
                    help="Si seleccionas menos de 3 se replicarán para completar RGB.",
                )

                if len(sel) == 0:
                    st.warning("Selecciona al menos una banda.")
                    st.stop()

                # 4️⃣ ––– Construir quick-look RGB
                while len(sel) < 3:
                    sel.append(sel[-1])  # Rellena con la última elegida

                rgb = im[:, :, sel[:3]].astype(np.float32)
                rgb -= rgb.min()
                rgb /= rgb.max() + 1e-9
                rgb = (rgb * 255).astype(np.uint8)

                st.image(rgb, caption=f"Quick-look RGB – bandas {sel[:3]}")

                # Limpieza
                os.remove(tmp_path)
main()

 # --- NUEVA FUNCIÓN ROBUSTA ---
def load_image_stack(folder_path, expected_channels=16, page_main=1):
    """
    Devuelve [(fname, img)] donde img es (H,W,bandas).
    page_main=1 ⇒ primero intenta la página 1 (cubo completo)
    y si no existe usa la 0 (compatibilidad con TIFF antiguos).
    """

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
def sidebar_files_upload():
    with st.sidebar.expander('Upload files'):
        st.caption('Add image files')
        dark=st.file_uploader('Upload dark image averaged in npy format',key="dark", type="npy")
        if dark is not None:
            dark=np.load(dark)
        white=st.file_uploader('Upload white image average in npy format',key="white",type="npy")
        if white is not None:
            white=np.load(white)
        folder_path_acquisition=st.button('Add folder data path')
        if folder_path_acquisition:
            data = easygui.diropenbox(msg="Select folder with reflectance files (.npy)",default="/home/alonso/Desktop")

        else:
            data=None

        return dark,white,data 

    ### Ends block for uploading files
#Acquire global variables from uploaders
dark, white, data = sidebar_files_upload()