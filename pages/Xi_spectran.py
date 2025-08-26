import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import easygui
import os
import cv2
import tifffile as tiff
from bisect import bisect_right
import xml.etree.ElementTree as ET

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

            st.session_state['reflectance_stack'] = reflectance_stack

            if "logs" not in st.session_state:
                st.session_state.logs = []
            st.session_state.logs.append(
                f"✅ Reflectance (memmap) shape: {reflectance_stack.shape}, dtype: {reflectance_stack.dtype}"
            )
    metadata = st.sidebar.file_uploader("📂 Upload CSV with metadata", type=".csv")
    # Si el usuario subió algo, lo guardamos en session_state
    if metadata is not None:
        st.session_state["metadata"] = metadata
    # Retornamos lo que se subió (sirve en este rerun)
    return (
        st.session_state.get('reflectance_stack', None),
        st.session_state.get("metadata",None)
        
    )
#load variables
try:
    reflectance_stack,metadata=folder_path_acquisition()
except:
    st.info('Load both .npz and metadata (.csv)')
# Ends load variables

def show_timestamps_panel(timestamps):
    if timestamps is None:
        st.info("No se han cargado timestamps todavía.")
        return
    metadat_df=pd.read_csv(metadata)
    with st.expander('Metadata'):
        st.dataframe(metadat_df)

    
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
        band1 = st.slider("Select band for HbO₂", min_value=0, max_value=15, value=2, step=1, key="band1_spec")
    with col2:
        band2 = st.slider("Select band for Hb", min_value=0, max_value=15, value=5, step=1, key="band2_spec")

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




def beer_lambert_calculations(
    λ1, λ2,
    Hb02_λ1, Hb_λ1, Hb02_λ2, Hb_λ2,
    E_unused,       # mantenido por compatibilidad, no se usa
    band1, band2,
    reflectance_stack,   # memmap/ndarray: (I, B, H, W) abierto con mmap_mode="r"
    original_stack=None, # no usado aquí
    timestamps=None,     # no usado aquí
    block_size=8,
    out_dir=None,        # si no es None, guarda memmaps: cHbO2, cHb, THb (como .npy)
    mask_range=(1e-5, 3e-4)  # para el preview de máscara
):
    """
    Opción A (segura): no modifica el archivo original y no sobrecarga la RAM.
    - Lee el stack en streaming por bloques.
    - Calcula rmax (1ª pasada).
    - Calcula absorbancia y THb/cHbO2/cHb por bloques (2ª pasada).
    - Si out_dir se especifica, guarda salidas en .npy (memmap) float32: (I,H,W).
    - Devuelve un preview de THb[0] enmascarado para visualizar.
    """
    # ── 0) Streamlit opcional
    try:
        import streamlit as st
        import matplotlib.pyplot as plt
    except Exception:
        class _Dummy:
            def __getattr__(self, name): 
                return lambda *a, **k: None
        st = _Dummy()
        # matplotlib solo si existe entorno gráfico; si no, no pasa nada.

    # ── 1) Shapes y matriz de extinción
    I, B, H, W = reflectance_stack.shape
    E = np.array([[Hb02_λ1, Hb_λ1],
                  [Hb02_λ2, Hb_λ2]], dtype=np.float32)

    det = float(np.linalg.det(E))
    cond = float(np.linalg.cond(E))
    if abs(det) < 0.01:
        st.warning(f"⚠️ Determinante muy bajo ({det:.4f}). Riesgo de inestabilidad.")
    elif abs(det) < 0.05:
        st.info(f"ℹ️ Determinante moderado ({det:.4f}). Úsalo con cautela.")
    else:
        st.success(f"✅ Determinante adecuado: {det:.4f}")
    if cond > 1000:
        st.warning(f"⚠️ Número de condición alto: {cond:.2f}")
    elif cond > 100:
        st.info(f"ℹ️ Condición moderada: {cond:.2f}")
    else:
        st.success(f"✅ Buena condición numérica: {cond:.2f}")

    invE = np.linalg.inv(E).astype(np.float32, copy=False)
    a, b = float(invE[0,0]), float(invE[0,1])
    c, d = float(invE[1,0]), float(invE[1,1])

    # ── 2) PASADA 1: rmax por streaming (barra de progreso)
    p1 = st.progress(0, text="Paso 1/2: Calculando rmax…")
    rmax = None
    for start in range(0, I, block_size):
        end = min(start + block_size, I)
        bmax = np.max(reflectance_stack[start:end])
        rmax = bmax if rmax is None else max(rmax, bmax)
        p1.progress(min(end / I, 1.0), text="Paso 1/2: Calculando rmax…")
    rmax = np.float32(1.0 if rmax is None else rmax)
    p1.empty()
    st.caption(f"rmax (global): {float(rmax):.6g}")

    # ── 3) Salidas como memmap .npy (solo si se solicitó)
    mm_cHbO2 = mm_cHb = mm_THb = None
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        mm_cHbO2 = open_memmap(os.path.join(out_dir, "cHbO2.npy"), mode="w+", dtype=np.float32, shape=(I, H, W))
        mm_cHb   = open_memmap(os.path.join(out_dir, "cHb.npy"),   mode="w+", dtype=np.float32, shape=(I, H, W))
        mm_THb   = open_memmap(os.path.join(out_dir, "THb.npy"),   mode="w+", dtype=np.float32, shape=(I, H, W))

    # ── 4) PASADA 2: procesamiento por bloques (buffer escribible) + barra
    p2 = st.progress(0, text="Paso 2/2: Procesando por bloques…")
    preview_thb0 = None
    lo, hi = mask_range

    for start in range(0, I, block_size):
        end = min(start + block_size, I)

        # Buffer escribible en float32
        src  = reflectance_stack[start:end]               # (n,B,H,W) read-only
        buf  = np.array(src, dtype=np.float32, copy=True) # (n,B,H,W) escribible

        # Normalización + clipping in-place
        buf /= rmax
        np.clip(buf, 1e-2, 1.0, out=buf)

        # Absorbancia in-place: -log(buf)
        np.log(buf, out=buf)
        buf *= -1.0

        # Bandas
        A1 = buf[:, band1, :, :]
        A2 = buf[:, band2, :, :]

        # Combinaciones lineales
        cHbO2_blk = a * A1 + b * A2
        cHb_blk   = c * A1 + d * A2
        THb_blk   = cHbO2_blk + cHb_blk

        # Escribir a disco si procede
        if mm_THb is not None:
            mm_cHbO2[start:end] = cHbO2_blk
            mm_cHb[start:end]   = cHb_blk
            mm_THb[start:end]   = THb_blk
            mm_cHbO2.flush(); mm_cHb.flush(); mm_THb.flush()

        # Primer frame para preview
        if preview_thb0 is None:
            thb0 = THb_blk[0]  # (H,W)
            mask = (thb0 > lo) & (thb0 < hi)
            preview_thb0 = np.where(mask, thb0, np.nan).astype(np.float32, copy=False)

        # liberar refs
        del cHbO2_blk, cHb_blk, THb_blk, A1, A2, buf, src

        p2.progress(min(end / I, 1.0), text="Paso 2/2: Procesando por bloques…")
    p2.empty()

    # ── 5) Visualización: matriz E y preview de THb[0]
    st.caption('Matriz de coeficientes de extinción (E)')
    st.dataframe(E)
    if preview_thb0 is not None:
        fig, ax = plt.subplots()
        cax = ax.imshow(preview_thb0, cmap='inferno', vmin=0, vmax=float(hi))
        fig.colorbar(cax, ax=ax)
        ax.set_title("THb[0] enmascarado")
        st.pyplot(fig)

    return {
        "determinant": det,
        "condition_number": cond,
        "rmax": float(rmax),
        "preview_masked_THb0": preview_thb0,   # (H,W) float32 con NaNs fuera de rango
        "outputs_dir": out_dir,
        "shapes": {"I": I, "B": B, "H": H, "W": W}
    }








col1,col2,col3=st.columns([1,1,0.5])
with col1:
    λ1, λ2, Hb02_λ1, Hb_λ1, Hb02_λ2, Hb_λ2, E, band1, band2=band_selection()
    run_calculations=st.button('Run calculations')
    if run_calculations:
        beer_lambert_calculations(λ1, λ2, Hb02_λ1, Hb_λ1, Hb02_λ2, Hb_λ2, E, band1, band2,reflectance_stack)
        
with col2:
    show_timestamps_panel(
        metadata)
with col3:
    
    st.subheader('Logs')
    if "logs" not in st.session_state:
        st.session_state.logs = []
    st.write(st.session_state.get("logs"))

    




