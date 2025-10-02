import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import easygui
import os
from tiffile import imwrite
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
            # Save in session_state
            st.session_state['reflectance_stack'] = data
            st.session_state['reflectance_path'] = path

            if "logs" not in st.session_state:
                st.session_state.logs = []
            st.session_state.logs.append(
                f"✅ Reflectance (memmap) shape: {reflectance_stack.shape}, dtype: {reflectance_stack.dtype}"
            )
            st.sidebar.caption(st.session_state["reflectance_path"])
            st.sidebar.caption(st.session_state["reflectance_stack"].shape)
            
    return st.session_state["reflectance_stack"]

def reflectance_visualization(reflectance_stack):
    with st.expander('Visualization of reflectance images'):
        n,c,h,w=reflectance_stack.shape
        select_image=st.slider('Select image',0,n-1,step=1)
        select_band=st.slider("Select band",0,c-1)
        image = reflectance_stack[select_image, select_band, :, :]  # (ej: primer frame, primer canal)
        
        fig, ax = plt.subplots()
        im_clean = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        vmin = np.percentile(im_clean, 2)
        vmax = np.percentile(im_clean, 98)

        im = ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
        ax.axis("off")  # oculta ejes si quieres
        
        st.pyplot(fig)

def band_selection():
    # --- 1) Load wavelengts from XML ---
    tree = ET.parse("ximea files/CMV2K-SSM4x4-460_600-15.7.20.6.xml")
    root = tree.getroot()
    bands = sorted(root.findall(".//band"), key=lambda b: int(b.get("index", 0)))
    wavelengths = np.array([float(b.find("peaks/peak/wavelength_nm").text) for b in bands], dtype=np.float32)
    
    # Arranged from lower to maximum wavelengths
    wavelengths = np.sort(wavelengths)
    df_wavelenght = pd.DataFrame({"Bands": range(len(wavelengths)), "Wavelength": wavelengths})
    with st.expander("Bands wavelengths"):
        st.dataframe(df_wavelenght, hide_index=True)

    # load absorption coefficients HbO2 and Hb
    df_spec = pd.read_excel("ximea files/HbO2_Hb_spectrum_full.xlsx")
    with st.expander('Molar extinction coefficients', expanded=False):
        st.dataframe(df_spec, hide_index=True)
    # --- 3) Band selection interface ---
    with col2:
        band1 = st.slider("Select band for HbO₂", min_value=0, max_value=15, value=13, step=1, key="band1_spec")
        band2 = st.slider("Select band for Hb", min_value=0, max_value=15, value=11, step=1, key="band2_spec")

    # Evitar que elijan la misma banda (no hay separación)
    if band1 == band2:
        st.error("Select different bands")
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
    st.info(f"1st wavelength {row1[0]}   | second wavelength {row2[0]}")

    # --- 8) Acquire absorbance coefficients ---
    Hb02_λ1 = float(row1['Hb02'])
    Hb_λ1   = float(row1['Hb'])
    Hb02_λ2 = float(row2['Hb02'])
    Hb_λ2   = float(row2['Hb'])

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

def mbll_2w_delta_pixel(
    R_abs,                 # reflectancia absoluta (T, C, H, W) ya corregida con blanco/negro
    band1, band2,         # índices de canal (coinciden con el stack)
    eps_HbO2_1, eps_Hb_1, # ε(λ1) HbO2, Hb
    eps_HbO2_2, eps_Hb_2, # ε(λ2) HbO2, Hb
    baseline_frames=60,   # n frames para basal
    DPF1=1.0, DPF2=1.0,   # DPF por λ (ajusta si tienes literatura específica)
    L=1.0,                # camino óptico efectivo (no afecta StO2)
    out_root="/home/alonso/Desktop",
    chunk_size=32,
    mask_min_R=1e-4,      # evita log de reflectancias ínfimas
    clip_nonneg=True      # clipeo suave de concentraciones a ≥0
):
    import numpy as np, os

    eps = 1e-12
    T, C, H, W = R_abs.shape
    n = min(baseline_frames, T)

    # --- Matriz E e inversa ---
    a11 = DPF1 * L * float(eps_HbO2_1); a12 = DPF1 * L * float(eps_Hb_1)
    a21 = DPF2 * L * float(eps_HbO2_2); a22 = DPF2 * L * float(eps_Hb_2)
    det = a11*a22 - a12*a21
    if abs(det) < 1e-12:
        raise ValueError("Matriz E mal condicionada (det≈0). Cambia λ1/λ2 o DPF.")
    inv11 =  a22 / det; inv12 = -a12 / det
    inv21 = -a21 / det; inv22 =  a11 / det

    # --- Basal por píxel con OD absolutas ---
    R1_base = np.clip(R_abs[:n, band1].astype(np.float32), mask_min_R, 1.0)
    R2_base = np.clip(R_abs[:n, band2].astype(np.float32), mask_min_R, 1.0)
    OD1_0 = -np.log(R1_base + eps).mean(axis=0)   # (H,W)
    OD2_0 = -np.log(R2_base + eps).mean(axis=0)   # (H,W)

    HbO2_0 = inv11*OD1_0 + inv12*OD2_0
    Hb_0   = inv21*OD1_0 + inv22*OD2_0
    if clip_nonneg:
        HbO2_0 = np.clip(HbO2_0, 0.0, None)
        Hb_0   = np.clip(Hb_0,   0.0, None)
    tHb_0  = np.clip(HbO2_0 + Hb_0, eps, None)
    good   = (tHb_0 > 10*eps).astype(np.float32)

    # --- Salidas (memmap) ---
    dHbO2_mm = np.lib.format.open_memmap(os.path.join(out_root, "dHbO2.npy"), mode='w+', dtype=np.float32, shape=(T,H,W))
    dHb_mm   = np.lib.format.open_memmap(os.path.join(out_root, "dHb.npy"),   mode='w+', dtype=np.float32, shape=(T,H,W))
    HbO2_mm  = np.lib.format.open_memmap(os.path.join(out_root, "HbO2.npy"),  mode='w+', dtype=np.float32, shape=(T,H,W))
    Hb_mm    = np.lib.format.open_memmap(os.path.join(out_root, "Hb.npy"),    mode='w+', dtype=np.float32, shape=(T,H,W))
    StO2_mm  = np.lib.format.open_memmap(os.path.join(out_root, "StO2.npy"),  mode='w+', dtype=np.float32, shape=(T,H,W))
    StO2mean = np.lib.format.open_memmap(os.path.join(out_root, "StO2_mean.npy"), mode='w+', dtype=np.float32, shape=(T,))

    # --- Procesamiento por bloques ---
    for t0 in range(0, T, chunk_size):
        t1 = min(t0+chunk_size, T)

        I1 = np.clip(R_abs[t0:t1, band1].astype(np.float32), mask_min_R, 1.0)
        I2 = np.clip(R_abs[t0:t1, band2].astype(np.float32), mask_min_R, 1.0)
        OD1 = -np.log(I1 + eps); OD2 = -np.log(I2 + eps)

        dOD1 = OD1 - OD1_0  # (B,H,W) - (H,W)
        dOD2 = OD2 - OD2_0

        dHbO2_blk = inv11*dOD1 + inv12*dOD2
        dHb_blk   = inv21*dOD1 + inv22*dOD2

        HbO2_blk = HbO2_0 + dHbO2_blk
        Hb_blk   = Hb_0   + dHb_blk
        if clip_nonneg:
            HbO2_blk = np.clip(HbO2_blk, 0.0, None)
            Hb_blk   = np.clip(Hb_blk,   0.0, None)

        tHb_blk  = np.clip(HbO2_blk + Hb_blk, eps, None)
        StO2_blk = np.clip(HbO2_blk / tHb_blk, 0.0, 1.0)

        # máscara de píxeles malos
        HbO2_blk *= good; Hb_blk *= good; StO2_blk *= good
        dHbO2_blk *= good; dHb_blk *= good

        dHbO2_mm[t0:t1] = dHbO2_blk
        dHb_mm[t0:t1]   = dHb_blk
        HbO2_mm[t0:t1]  = HbO2_blk
        Hb_mm[t0:t1]    = Hb_blk
        StO2_mm[t0:t1]  = StO2_blk
        StO2mean[t0:t1] = StO2_blk.reshape(t1-t0, -1).mean(axis=1).astype(np.float32)

        dHbO2_mm.flush(); dHb_mm.flush(); HbO2_mm.flush(); Hb_mm.flush(); StO2_mm.flush(); StO2mean.flush()

    return {
        "dHbO2_path": os.path.join(out_root, "dHbO2.npy"),
        "dHb_path":   os.path.join(out_root, "dHb.npy"),
        "HbO2_path":  os.path.join(out_root, "HbO2.npy"),
        "Hb_path":    os.path.join(out_root, "Hb.npy"),
        "StO2_path":  os.path.join(out_root, "StO2.npy"),
        "StO2mean_path": os.path.join(out_root, "StO2_mean.npy"),
    }



def _auto_vmin_vmax_joint(rgb, p_low=2, p_high=98, ignore_zeros=True):
    """
    Calcula vmin/vmax conjuntos para un stack RGB (H,W,3) usando percentiles robustos.
    Mantiene la relación entre canales (fidelidad de color).
    """
    a = np.asarray(rgb, dtype=np.float32)
    # aplanar canales juntos
    flat = a.reshape(-1, a.shape[-1])
    flat = flat[np.all(np.isfinite(flat), axis=1)]  # filas sin NaNs/Inf
    if flat.size == 0:
        return 0.0, 1.0
    vals = flat.reshape(-1)  # mezcla R,G,B en un vector
    if ignore_zeros:
        vals = vals[vals > 0]
        if vals.size == 0:
            return 0.0, 1.0
    vmin = float(np.percentile(vals, p_low))
    vmax = float(np.percentile(vals, p_high))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.min(vals)); vmax = float(np.max(vals))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = 0.0, 1.0
        else:
            vmax = vmin + 1e-6
    return vmin, vmax

def _to_uint16_joint(rgb, vmin=None, vmax=None, gamma=0.8, ignore_zeros=True):
    """
    Normaliza RGB -> [0,1] con vmin/vmax conjuntos, aplica gamma común y pasa a uint16.
    rgb: (H, W, 3) float.
    """
    im = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if vmin is None or vmax is None:
        vmin, vmax = _auto_vmin_vmax_joint(im, p_low=2, p_high=98, ignore_zeros=ignore_zeros)
    norm = (im - vmin) / (vmax - vmin + 1e-12)
    norm = np.clip(norm, 0.0, 1.0)
    if abs(gamma - 1.0) > 1e-6:
        norm = np.power(norm, 1.0/gamma)
    return (norm * 65535.0 + 0.5).astype(np.uint16)

def _to_uint16_gray(img, p_low=2, p_high=98, gamma=1.0, ignore_zeros=True):
    """
    Escala robusta por percentiles para una imagen gris (H,W).
    """
    im = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    vals = im[np.isfinite(im)]
    if ignore_zeros:
        vals = vals[vals > 0]
    if vals.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.percentile(vals, p_low))
        vmax = float(np.percentile(vals, p_high))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin = float(np.min(vals)); vmax = float(np.max(vals))
            if vmax <= vmin:
                vmax = vmin + 1e-6
    norm = (im - vmin) / (vmax - vmin + 1e-12)
    norm = np.clip(norm, 0.0, 1.0)
    if abs(gamma - 1.0) > 1e-6:
        norm = np.power(norm, 1.0/gamma)
    return (norm * 65535.0 + 0.5).astype(np.uint16)

# ---------- Exportador con fidelidad (normalización conjunta) ----------

def tiffiles_export(reflectance_stack, out_root="/home/alonso/Desktop"):
    
    if reflectance_stack is None:
        raise ValueError("reflectance_stack es None.")
    T, C, H, W = reflectance_stack.shape
    if C < 3:
        raise ValueError(f"Se requieren ≥3 canales para RGB. C={C}")

    # Configuración fija (ajusta si quieres otros índices)
    GRAY_CH = 0                 # canal para escala de grises
    R_idx, G_idx, B_idx = 15, 10, 0
    GAMMA_GRAY = 0.8
    GAMMA_RGB  = 0.8
    IGNORE_ZEROS = True

    # Carpeta de salida
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(out_root, f"TIFFs_{stamp}")
    out_gray = os.path.join(out_dir, "grayscale")
    out_rgb  = os.path.join(out_dir, "rgb")
    os.makedirs(out_gray, exist_ok=True)
    os.makedirs(out_rgb, exist_ok=True)

    # --- Export GRAYSCALE ---
    for i in range(T):
        img = reflectance_stack[i, GRAY_CH]  # (H,W)
        img_u16 = _to_uint16_gray(img, p_low=2, p_high=98, gamma=GAMMA_GRAY, ignore_zeros=IGNORE_ZEROS)
        imwrite(os.path.join(out_gray, f"frame_{i:04d}.tif"), img_u16, photometric="minisblack")

    # --- Export RGB (normalización conjunta → fidelidad) ---
    for i in range(T):
        r = reflectance_stack[i, R_idx]
        g = reflectance_stack[i, G_idx]
        b = reflectance_stack[i, B_idx]
        rgb = np.stack([r, g, b], axis=-1)  # (H,W,3) float

        rgb_u16 = _to_uint16_joint(rgb, gamma=GAMMA_RGB, ignore_zeros=IGNORE_ZEROS)
        imwrite(os.path.join(out_rgb, f"frame_{i:04d}.tif"), rgb_u16, photometric="rgb")

    return out_dir


#load variables
try:
    reflectance_stack=folder_path_acquisition()
except:
    st.info('Load .npy file')
# Ends load variables
reflectance_visualization(reflectance_stack)

col1,col2,col3=st.columns([1,1,0.5])
with col1:
    λ1, λ2, Hb02_λ1, Hb_λ1, Hb02_λ2, Hb_λ2, E, band1, band2=band_selection()
    run_calculations=st.button('Run calculations and export files')
    if run_calculations and reflectance_stack is not None:
        mbll_2w_delta_pixel(reflectance_stack,band1,band2,Hb02_λ1,Hb_λ1,Hb02_λ2,Hb_λ2)
        tiffiles_export(reflectance_stack)
    elif run_calculations and reflectance_stack is None:
        st.warning("Load reflectance stack")
 
with col3:
    st.subheader('Logs')
    if "logs" not in st.session_state:
        st.session_state.logs = []
    st.write(st.session_state.get("logs"))

    




