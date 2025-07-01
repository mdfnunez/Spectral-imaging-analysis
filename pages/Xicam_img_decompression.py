import os
import getpass
import numpy as np
import blosc2
import tifffile
from datetime import datetime
import streamlit as st
import os, re, unicodedata
from typing import Optional, Dict

import blosc2
import numpy as np
import tifffile
# Importaciones de Tkinter
import tkinter as tk
from tkinter import filedialog
import os, re, unicodedata
from typing import Optional, Dict

import blosc2
import numpy as np
import tifffile


import os, re, unicodedata
from typing import Optional, Dict

import blosc2
import numpy as np
import tifffile


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
with st.expander("Viewer multiband"):
    # viewer_multibanda.py
    import os
    import streamlit as st
    import tifffile as tf
    import numpy as np


    st.title("Visor rápido de TIFF multibanda")

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


if __name__ == "__main__":
    main()
