import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import easygui
import os
import cv2
import tifffile as tiff
from tiffile import imwrite
from bisect import bisect_right
import xml.etree.ElementTree as ET
from datetime import datetime


st.set_page_config(layout="wide")

#Desktop folder path
global default
default="/home/alonso/Desktop/"
date=datetime.now()

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
    # 1) Reflectance NPY (memmap)
    # ———————————————————————————————
    if st.sidebar.button('Add reflectance .npy file', key=9815):
        path = easygui.fileopenbox(
            msg="Select a .npy file with reflectance stacks",
            default="/home/alonso/Desktop/",
            filetypes=["*.npy"]
        )
        if path:
            data = np.load(path, allow_pickle=False, mmap_mode="r")
            reflectance_stack = data  # memmap-like

            if reflectance_stack.dtype == np.float64:
                st.warning("Reflectance en float64. Convertiré por bloques a float32 si procesas todo.")

            # Guardar en session_state
            st.session_state['reflectance_stack'] = reflectance_stack
            st.session_state['reflectance_path'] = path

            if "logs" not in st.session_state:
                st.session_state.logs = []
            st.session_state.logs.append(
                f"✅ Reflectance (memmap) shape: {reflectance_stack.shape}, dtype: {reflectance_stack.dtype}"
            )
            st.sidebar.caption(st.session_state["reflectance_path"])
            st.sidebar.caption(st.session_state["reflectance_stack"].shape)
            # ← Return explícito
            

    # Si no se seleccionó nada, devuelve None
    return st.session_state["reflectance_stack"]
#load variables
try:
    reflectance_stack=folder_path_acquisition()
except:
    st.info('Load .npy file')
# Ends load variables
    
#Band selection 
def band_selection():
    # --- 1) Load wavelengts from XML ---
    tree = ET.parse("ximea files/CMV2K-SSM4x4-460_600-15.7.20.6.xml")
    root = tree.getroot()
    bands = sorted(root.findall(".//band"), key=lambda b: int(b.get("index", 0)))
    wavelengths = np.array([float(b.find("peaks/peak/wavelength_nm").text) for b in bands], dtype=np.float32)
    
    # Arranged from lower to maximum wavelengths
    wavelengths = np.sort(wavelengths)
    df_wavelenght = pd.DataFrame({"Bands": range(16), "Wavelenght": wavelengths})
    with st.expander("Bands wavelenghts"):
        st.dataframe(df_wavelenght, hide_index=True)

    # load absorption coefficients HbO2 and Hb
    df_spec = pd.read_excel("ximea files/HbO2_Hb_spectrum_full.xlsx")
    with st.expander('Molar extinction coefficients', expanded=False):
        st.dataframe(df_spec, hide_index=True)

    # --- 3) Band selection interface ---
    with col2:
        band1 = st.slider("Select band for HbO₂", min_value=0, max_value=15, value=13, step=1, key="band1_spec")
    with col2:
        band2 = st.slider("Select band for Hb", min_value=0, max_value=15, value=11, step=1, key="band2_spec")

    # Evitar que elijan la misma banda (no hay separación)
    if band1 == band2:
        st.error("Selecciona dos bandas distintas. Con la misma banda no hay separación.")
        return None, None, None, None, None, None, None, band1, band2

    # Selection of wavelength in wavelengths with the sliders
    λ1 = int(wavelengths[band1])
    λ2 = int(wavelengths[band2])

    # --- 5) Select wavelengths within 450 and 650 nm ---
    df_zoom = df_spec[(df_spec["lambda"] >= 450) & (df_spec["lambda"] <= 650)]

    # --- 6) Graph with molar coefficients and selection of bands ---
    with col2:
        with st.expander('Molar extinction graph and selected bands', expanded=True):
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(df_zoom["lambda"], df_zoom["Hb02"], label="ε HbO₂", color="crimson")
            ax.plot(df_zoom["lambda"], df_zoom["Hb"], label="ε Hb", color="royalblue")
            ax.fill_between(df_zoom["lambda"], df_zoom["Hb02"], df_zoom["Hb"], color='gray', alpha=0.4)

            ax.axvline(λ1, color="crimson", linestyle="--", lw=2, label=f"Band HbO₂ ~ {λ1} nm")
            ax.axvline(λ2, color="royalblue", linestyle="--", lw=2, label=f"Band Hb ~ {λ2} nm")

            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Coefficient ε (normalized)")
            ax.set_title("Molar extinction spectrum")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)

    # --- 7) Tomar la fila más cercana a cada λ (si no hay match exacto) ---
    # Buscar coincidencia cercana a λ1
    match1 = df_zoom[np.isclose(df_zoom["lambda"], λ1, atol=2)]
    if match1.empty:
        idx1 = (df_zoom["lambda"] - λ1).abs().idxmin()
        row1 = df_zoom.loc[idx1]
    else:
        row1 = match1.iloc[0]

    # Buscar coincidencia cercana a λ2
    match2 = df_zoom[np.isclose(df_zoom["lambda"], λ2, atol=2)]
    if match2.empty:
        idx2 = (df_zoom["lambda"] - λ2).abs().idxmin()
        row2 = df_zoom.loc[idx2]
    else:
        row2 = match2.iloc[0]

    # --- 8) Acquire absorbance coefficients ---
    Hb02_λ1 = float(row1['Hb02'])
    Hb_λ1   = float(row1['Hb'])
    Hb02_λ2 = float(row2['Hb02'])
    Hb_λ2   = float(row2['Hb'])
    data = pd.DataFrame([{"HbO2-1": Hb02_λ1, "HbO2-2": Hb02_λ2, "Hb1": Hb_λ1, "Hb2": Hb_λ2}])
    st.dataframe(data, hide_index=True)

    # --- 9) Matriz de coeficientes (para diagnóstico de condición) ---
    E = np.array([
        [Hb02_λ1, Hb_λ1],
        [Hb02_λ2, Hb_λ2]
    ], dtype=float)

    df_E = pd.DataFrame(E, columns=["HbO₂", "Hb"], index=["W1", "W2"])
    df_E["Δ(HbO₂-Hb)"] = df_E["HbO₂"] - df_E["Hb"]
    st.dataframe(df_E)

    # --- 10) MÉTRICA RATIO (Δε) INLINE: sin funciones auxiliares ---
    # Diferencias absolutas
    delta1 = Hb02_λ1 - Hb_λ1
    delta2 = Hb02_λ2 - Hb_λ2

    # Normalización local (escala dominante por λ)
    scale1 = max(abs(Hb02_λ1), abs(Hb_λ1), 1e-12)
    scale2 = max(abs(Hb02_λ2), abs(Hb_λ2), 1e-12)
    rel1 = abs(delta1) / scale1
    rel2 = abs(delta2) / scale2

    # ¿Las diferencias tienen signos opuestos?
    signs_opposite = (np.sign(delta1) != 0) and (np.sign(delta2) != 0) and (np.sign(delta1) != np.sign(delta2))

    # Puntajes
    score_mean = 0.5 * (rel1 + rel2)
    score_geo  = float(np.sqrt(rel1 * rel2))

    # Mostrar tabla compacta de la métrica
    df_ratio = pd.DataFrame([{
        "λ1 Δ(HbO₂−Hb)": delta1,
        "λ1 Δ_rel": rel1,
        "λ2 Δ(HbO₂−Hb)": delta2,
        "λ2 Δ_rel": rel2,
        "Signos opuestos": "Sí" if signs_opposite else "No",
        "Score (media)": score_mean,
        "Score (geom)": score_geo,
    }])
    st.dataframe(
        df_ratio.style.format({
            "λ1 Δ(HbO₂−Hb)": "{:.4g}", "λ1 Δ_rel": "{:.3f}",
            "λ2 Δ(HbO₂−Hb)": "{:.4g}", "λ2 Δ_rel": "{:.3f}",
            "Score (media)": "{:.3f}", "Score (geom)": "{:.3f}",
        }),
        hide_index=True
    )

    # --- 11) Diagnóstico de estabilidad: número de condición ---
    condition_number = float(np.linalg.cond(E))
    st.write(f"Número de condición: {condition_number:.2f}")

    # --- 12) Semáforos (ajusta umbrales según tus datos reales) ---
    # Semáforo de separación espectral
    score = score_geo  # más exigente; usa score_mean si prefieres
    if signs_opposite and score >= 0.30:
        st.success(f"✅ Excelente separación espectral. Score={score:.3f} (signos opuestos).")
    elif score >= 0.15:
        st.info(f"ℹ️ Separación aceptable. Score={score:.3f}{' (signos opuestos)' if signs_opposite else ''}")
    else:
        st.warning(f"⚠️ Separación pobre. Score={score:.3f}. Prueba bandas más alejadas o cruza signo.")

    # Semáforo de condición numérica
    if condition_number > 1000:
        st.warning(f"⚠️ Alta inestabilidad numérica. cond(E) = {condition_number:.2f}")
    elif condition_number > 100:
        st.info(f"ℹ️ Condición moderada. cond(E) = {condition_number:.2f}")
    else:
        st.success(f"✅ Buena condición numérica: cond(E) = {condition_number:.2f}")

    return λ1, λ2, Hb02_λ1, Hb_λ1, Hb02_λ2, Hb_λ2, E, band1, band2

def mbll_2w_save_npy(
    R,                 # reflectance_stack: (T, C, H, W)  float o uint16
    band1, band2,      # índices de banda (λ1, λ2)
    eps_HbO2_1, eps_Hb_1,   # ε en λ1
    eps_HbO2_2, eps_Hb_2,   # ε en λ2
    baseline_frames=60,
    StO2_0=0.70,
    DPF1=0.5, DPF2=0.5,
    chunk_size=32      # nº de frames por bloque en el bucle
):
    
    eps = 1e-8
    T, C, H, W = R.shape
    n = min(baseline_frames, T)

    # --- Baseline explícito (promedio primeros n frames por banda)
    I01 = R[:n, band1].mean(axis=0, dtype=np.float64).astype(np.float32) + eps  # (H,W)
    I02 = R[:n, band2].mean(axis=0, dtype=np.float64).astype(np.float32) + eps  # (H,W)

    # --- Matriz E con DPF incluido e inversa cerrada (2x2)
    a11 = DPF1 * eps_HbO2_1; a12 = DPF1 * eps_Hb_1
    a21 = DPF2 * eps_HbO2_2; a22 = DPF2 * eps_Hb_2
    det = a11*a22 - a12*a21
    if abs(det) < 1e-12:
        raise ValueError("Matriz mal condicionada (det≈0). Cambia λ1/λ2 o DPF.")
    inv11 =  a22 / det; inv12 = -a12 / det
    inv21 = -a21 / det; inv22 =  a11 / det

    # --- Ancla basal para StO2 absoluta (tHb0=1)
    HbO2_0 = float(StO2_0)
    Hb_0   = float(1.0 - StO2_0)

    # --- Crea archivos memmap de salida (evita usar RAM)
    dHbO2_mm   = np.lib.format.open_memmap(f"{default}_dHbO2.npy", mode='w+', dtype=np.float32, shape=(T, H, W))
    dHb_mm     = np.lib.format.open_memmap(f"{default}_dHb.npy",   mode='w+', dtype=np.float32, shape=(T, H, W))
    StO2_mm    = np.lib.format.open_memmap(f"{default}_StO2.npy",  mode='w+', dtype=np.float32, shape=(T, H, W))
    StO2mean_mm= np.lib.format.open_memmap(f"{default}_StO2_mean.npy", mode='w+', dtype=np.float32, shape=(T,))

    # --- Procesa por bloques en T
    for t0 in range(0, T, chunk_size):
        t1 = min(t0 + chunk_size, T)

        I1 = R[t0:t1, band1].astype(np.float32)    # (B,H,W)
        I2 = R[t0:t1, band2].astype(np.float32)    # (B,H,W)

        # ΔOD = -ln(I/I0), con clip de seguridad
        dOD1 = -np.log(np.clip(I1 / I01, eps, None))
        dOD2 = -np.log(np.clip(I2 / I02, eps, None))

        # x = E^{-1} y  (vectorizado)
        dHbO2_blk = inv11 * dOD1 + inv12 * dOD2     # (B,H,W)
        dHb_blk   = inv21 * dOD1 + inv22 * dOD2     # (B,H,W)

        # StO2(t) con ancla basal
        HbO2_t = HbO2_0 + dHbO2_blk
        Hb_t   = Hb_0   + dHb_blk
        tHb_t  = np.clip(HbO2_t + Hb_t, eps, None)
        StO2_blk = np.clip(HbO2_t / tHb_t, 0.0, 1.0)

        # Guarda al memmap
        dHbO2_mm[t0:t1] = dHbO2_blk
        dHb_mm[t0:t1]   = dHb_blk
        StO2_mm[t0:t1]  = StO2_blk
        StO2mean_mm[t0:t1] = StO2_blk.reshape(t1 - t0, -1).mean(axis=1)

        # opcional: asegura flush del bloque (útil en procesos largos)
        dHbO2_mm.flush(); dHb_mm.flush(); StO2_mm.flush(); StO2mean_mm.flush()

def _to_uint16(img):
    # autocontraste simple por frame
    lo = np.nanmin(img)
    hi = np.nanmax(img)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(img, dtype=np.uint16)
    img = (img - lo) / (hi - lo + 1e-12)
    return (img * 65535).astype(np.uint16)

def tiffiles_export(reflectance_stack, default_dir="/home/alonso/Desktop"):
    with st.expander("Download tiffiles"):
        gen_tiff = st.button("Generate all TIFFs")
        n, c, h, w = reflectance_stack.shape
        st.write(f"Frames encontrados: {n}, canales: {c}")

        if gen_tiff:
            # carpeta con timestamp
            date = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = os.path.join(default_dir, f"TIFFs_{date}")
            out_gray = os.path.join(out_dir, "grayscale")
            out_rgb  = os.path.join(out_dir, "rgb")
            os.makedirs(out_gray, exist_ok=True)
            os.makedirs(out_rgb, exist_ok=True)

            # --- Export GRAYSCALE (canal 0, 16-bit) ---
            for i in range(n):
                img = reflectance_stack[i, 0, :, :]   # frame i, canal 0 (ajusta si quieres otro)
                img_u16 = _to_uint16(img)
                out_path = os.path.join(out_gray, f"frame_{i:04d}.tif")
                imwrite(out_path, img_u16, photometric="minisblack")

            # --- Export RGB (elige 3 canales; aquí 3,7,12 a modo de ejemplo) ---
            R_idx, G_idx, B_idx = 15, 10, 0  # cámbialos a tus bandas favoritas
            for i in range(n):
                r = reflectance_stack[i, R_idx, :, :]
                g = reflectance_stack[i, G_idx, :, :]
                b = reflectance_stack[i, B_idx, :, :]

                # normalización conjunta para mantener el color
                rgb = np.stack([r, g, b], axis=-1)   # (H, W, 3)
                lo = np.nanmin(rgb)
                hi = np.nanmax(rgb)
                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                    rgb_u16 = np.zeros_like(rgb, dtype=np.uint16)
                else:
                    rgb_n = (rgb - lo) / (hi - lo + 1e-12)
                    rgb_u16 = (np.clip(rgb_n, 0, 1) * 65535).astype(np.uint16)

                out_path = os.path.join(out_rgb, f"frame_{i:04d}.tif")
                # OJO: photometric="rgb" para color
                imwrite(out_path, rgb_u16, photometric="rgb")  # visores normales lo leen bien

            st.success(f"{n} TIFFs guardados en:\n{out_dir}")






col1,col2,col3=st.columns([1,1,0.5])
with col1:
    λ1, λ2, Hb02_λ1, Hb_λ1, Hb02_λ2, Hb_λ2, E, band1, band2=band_selection()
    run_calculations=st.toggle('Run calculations')
    if run_calculations and reflectance_stack is not None:
        mbll_2w_save_npy(reflectance_stack,band1,band2,Hb02_λ1,Hb_λ1,Hb02_λ2,Hb_λ2)
    elif run_calculations and reflectance_stack is None:
        st.warning("Load reflectance stack")
with col2:
    tiffiles_export(reflectance_stack)
with col3:
    st.subheader('Logs')
    if "logs" not in st.session_state:
        st.session_state.logs = []
    st.write(st.session_state.get("logs"))

    




