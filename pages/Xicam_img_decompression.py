import os
import getpass
import numpy as np
import blosc2
import tifffile
from datetime import datetime
import streamlit as st

# Importaciones de Tkinter
import tkinter as tk
from tkinter import filedialog

def demosaic(image, mosaic_size):
    """Convierte una imagen en patrón de mosaico en una imagen demosaiced con canales separados."""
    rows = image.shape[0] // mosaic_size
    cols = image.shape[1] // mosaic_size
    bands = mosaic_size * mosaic_size
    img_demosaiced = np.zeros((rows, cols, bands), dtype=image.dtype)
    i_wavelength = 0
    for i_row in range(mosaic_size):
        for i_col in range(mosaic_size):
            img_demosaiced[:, :, i_wavelength] = image[i_row::mosaic_size, i_col::mosaic_size]
            i_wavelength += 1
    return img_demosaiced

def save_images_to_tiff(b2nd_path, output_folder, grayscale=False):
    """Descomprime imágenes de un archivo .b2nd y guarda solo la banda requerida en caso de grayscale."""
    BLOSC_DATA = blosc2.open(b2nd_path, mode="r")
    os.makedirs(output_folder, exist_ok=True)

    total_frames = BLOSC_DATA.shape[0]
    print(f"Total de cuadros en el archivo .b2nd: {total_frames}")

    for i in range(total_frames):
        image_data = BLOSC_DATA[i][...]
        print(f"Imagen {i}: forma={image_data.shape}, tipo={image_data.dtype}")

        # Intentar leer time_stamp
        try:
            time_stamp = BLOSC_DATA.schunk.vlmeta["time_stamp"][i]
            print(f"Marca de tiempo: {time_stamp}")
        except (AttributeError, KeyError, IndexError):
            time_stamp = "N-A"

        # Procesar la imagen
        if grayscale:
            # Si es 3D, tomar la primera banda
            if len(image_data.shape) > 2:
                image_data = image_data[:, :, 0]
            # Expandir a 3D (1 canal) para TIFF
            image_data = np.expand_dims(image_data, axis=0)
        else:
            # Si es 2D, asumir mosaico
            if len(image_data.shape) == 2:
                mosaic_size = 4
                image_data = demosaic(image_data, mosaic_size)
            # Si es 3D, mover el eje de bandas al inicio
            if len(image_data.shape) == 3:
                image_data = np.transpose(image_data, (2, 0, 1))

        # Normalizar a 16 bits
        max_value = np.max(image_data)
        if max_value > 0:
            image_data = (image_data / max_value * 65535).astype(np.uint16)
        else:
            image_data = image_data.astype(np.uint16)

        outname = os.path.join(output_folder, f"{time_stamp}_imagen_{i}.tif")
        tifffile.imwrite(outname, image_data, photometric='minisblack')
        print(f"Guardado: {outname}")

    print("Descompresión y guardado completados.")

def pick_folder_tk():
    """
    Abre un diálogo de selección de carpeta con Tkinter
    y configura un directorio inicial donde suelen montarse los discos externos.
    En muchas distros GNOME, es /media/<usuario> o /run/media/<usuario>.
    Ajusta la ruta según tu sistema.
    """
    # Intentamos /media/<usuario> primero
    username = getpass.getuser()
    media_path = os.path.join("/media", username)  # /media/tu_usuario

    # Si no existe esa carpeta, probamos /run/media/<usuario>
    if not os.path.exists(media_path):
        alt_media_path = os.path.join("/run", "media", username)
        if os.path.exists(alt_media_path):
            media_path = alt_media_path

    # Si tampoco existe, usamos /media directamente
    if not os.path.exists(media_path):
        media_path = "/media"

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
    st.title("Exportador de Imágenes .b2nd (Tkinter + Ruta Montaje)")

    # 1) Seleccionar el archivo b2nd
    option = st.selectbox("Proveer el archivo .b2nd", ["Subir archivo", "Ruta manual"])
    if option == "Subir archivo":
        uploaded_file = st.file_uploader("Sube tu archivo .b2nd", type=["b2nd"])
        b2nd_path = None
    else:
        b2nd_path = st.text_input("Ruta al archivo .b2nd")
        uploaded_file = None
        if b2nd_path and not os.path.isfile(b2nd_path):
            st.error("¡La ruta especificada no es válida!")

    # 2) Botón para seleccionar carpeta con Tkinter (mostrando discos externos)
    st.markdown("### Selecciona la carpeta de salida")
    if st.button("Seleccionar carpeta en /media/<usuario>"):
        selected_folder = pick_folder_tk()
        if selected_folder:
            st.session_state["selected_folder"] = selected_folder
            st.success(f"Carpeta seleccionada: {selected_folder}")
        else:
            st.warning("No se seleccionó ninguna carpeta (cerraste el diálogo).")

    # 3) Opción de escribir manualmente
    manual_output = st.text_input("O introduce manualmente la carpeta de salida", "")

    # 4) Subcarpeta
    subfolder_name = st.text_input("Nombre de subcarpeta (opcional)", "")

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

if __name__ == "__main__":
    main()
