import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, json, easygui
from tifffile import imwrite
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
import numpy as np, os, json
from math import ceil

st.set_page_config(layout="wide")

default = "/home/alonso/Desktop/"
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
    if st.sidebar.button('Add reflectance .npy file', key=9815):
        path = easygui.fileopenbox(
            msg="Select a .npy file with reflectance stacks",
            default=default, filetypes=["*.npy"]
        )
        if path:
            data = np.load(path, allow_pickle=False, mmap_mode="r")
            st.session_state["reflectance_stack"] = data
            st.session_state["reflectance_path"]  = path

            logs = st.session_state.get("logs", [])
            logs.append(f"✅ Reflectance (memmap) shape: {data.shape}, dtype: {data.dtype}")
            st.session_state["logs"] = logs

            st.sidebar.caption(path)
            st.sidebar.caption(str(data.shape))
    return st.session_state.get("reflectance_stack")

def reflectance_visualization(reflectance_stack):
    if reflectance_stack is None:
        st.info("Load .npy file to visualize.")
        return
    with st.expander('Visualization of reflectance images'):
        n,c,h,w=reflectance_stack.shape
        select_image=st.slider('Select image',0,n-1,step=1)
        select_band=st.slider("Select band",0,c-1)
        image = reflectance_stack[select_image, select_band]
        im_clean = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        vmin = np.percentile(im_clean, 2); vmax = np.percentile(im_clean, 98)
        fig, ax = plt.subplots()
        ax.imshow(im_clean, cmap="gray", vmin=vmin, vmax=vmax)
        ax.axis("off")
        st.pyplot(fig)


def band_selection():
    # 1) λ virtuales desde JSON junto al .npy (mismo nombre, .json)
    lambdas = None
    ref_path = st.session_state.get("reflectance_path")
    if ref_path:
        base = os.path.splitext(os.path.basename(ref_path))[0]
        folder = os.path.dirname(ref_path)
        json_path = os.path.join(folder, f"{base}.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                v = meta.get("virtual_wavelengths_nm")
                if v and len(v) == 16:
                    lambdas = np.array(v, dtype=np.float32)
            except Exception as e:
                st.warning(f"No pude leer {json_path}: {e}")

    # 2) Fallback a XML (primero virtual_bands; si no, bandas físicas)
    if lambdas is None or len(lambdas) != 16:
        try:
            tree = ET.parse("ximea files/CMV2K-SSM4x4-460_600-15.7.20.6.xml")
            root = tree.getroot()
            vbs = root.findall(".//correction_matrix[name='hsi_reflectance']/virtual_bands/virtual_band/wavelength_nm")
            if vbs:
                lambdas = np.array([float(v.text) for v in vbs], dtype=np.float32)
            else:
                bands = sorted(root.findall(".//band"), key=lambda b: int(b.get("index", 0)))
                lambdas = np.array([float(b.find("peaks/peak/wavelength_nm").text) for b in bands], dtype=np.float32)
        except Exception as e:
            st.error(f"No pude obtener longitudes de onda ni de JSON ni de XML: {e}")
            return None, None, None, None, None, None, None, None, None

    # Orden ascendente por λ → índice 0..15 = 460→594 nm
    order = np.argsort(lambdas)
    lambdas = lambdas[order]

    with st.expander("Bands wavelengths (virtual, ascendente)"):
        st.dataframe(pd.DataFrame({"Band": range(16), "Wavelength_nm": lambdas}), hide_index=True)

    # 3) Cargar espectros ε (acepta HbO2 o Hb02)
    try:
        df_spec = pd.read_excel("ximea files/HbO2_Hb_spectrum_full.xlsx")
    except Exception as e:
        st.error(f"No pude leer el Excel de espectros: {e}")
        return None, None, None, None, None, None, None, None, None

    cols = {c.lower().strip(): c for c in df_spec.columns}
    if "lambda" not in cols:
        st.error("El Excel debe contener una columna 'lambda'.")
        return None, None, None, None, None, None, None, None, None
    col_lambda = cols["lambda"]
    col_hbo2 = cols.get("hbo2") or cols.get("hb02")
    col_hb   = cols.get("hb") or "Hb"
    if not col_hbo2:
        st.error("El Excel debe contener columna 'HbO2' (o 'Hb02').")
        return None, None, None, None, None, None, None, None, None

    # 4) Selección de bandas
    band1 = st.slider("Select band for HbO₂", 0, 15, 13, 1, key="band1_spec")
    band2 = st.slider("Select band for Hb",   0, 15, 11, 1, key="band2_spec")
    if band1 == band2:
        st.error("Select different bands")
        return None, None, None, None, None, None, None, band1, band2

    λ1, λ2 = int(lambdas[band1]), int(lambdas[band2])

    # 5) Buscar ε cerca de λ1 y λ2 (450–650 nm)
    df_zoom = df_spec[(df_spec[col_lambda] >= 450) & (df_spec[col_lambda] <= 650)]
    def pick_row(df, wl, tol=2):
        m = df[np.isclose(df[col_lambda], wl, atol=tol)]
        if m.empty:
            idx = (df[col_lambda] - wl).abs().idxmin()
            return df.loc[idx]
        return m.iloc[0]
    row1 = pick_row(df_zoom, λ1); row2 = pick_row(df_zoom, λ2)

    HbO2_λ1 = float(row1[col_hbo2]); Hb_λ1 = float(row1[col_hb])
    HbO2_λ2 = float(row2[col_hbo2]); Hb_λ2 = float(row2[col_hb])

    # 6) Matriz E y diagnóstico de condición
    E = np.array([[HbO2_λ1, Hb_λ1],[HbO2_λ2, Hb_λ2]], dtype=float)
    condE = float(np.linalg.cond(E))
    if condE > 1000:
        st.warning(f"⚠️ Alta inestabilidad numérica. cond(E) = {condE:.2f}")
    elif condE > 100:
        st.info(f"ℹ️ Condición moderada. cond(E) = {condE:.2f}")
    else:
        st.success(f"✅ Buena condición numérica: cond(E) = {condE:.2f}")

    # 7) MÉTRICAS de separación (Δε, relativas, signos, scores)
    delta1 = HbO2_λ1 - Hb_λ1
    delta2 = HbO2_λ2 - Hb_λ2
    scale1 = max(abs(HbO2_λ1), abs(Hb_λ1), 1e-12)
    scale2 = max(abs(HbO2_λ2), abs(Hb_λ2), 1e-12)
    rel1 = abs(delta1) / scale1
    rel2 = abs(delta2) / scale2
    signs_opposite = (np.sign(delta1) != 0) and (np.sign(delta2) != 0) and (np.sign(delta1) != np.sign(delta2))
    score_mean = 0.5 * (rel1 + rel2)
    score_geo  = float(np.sqrt(rel1 * rel2))

    with st.expander("Band selection diagnostics", expanded=True):
        df_E = pd.DataFrame(E, columns=["HbO₂", "Hb"], index=[f"W1 ({λ1} nm)", f"W2 ({λ2} nm)"])
        df_E["Δ(HbO₂−Hb)"] = [delta1, delta2]
        st.dataframe(df_E.style.format("{:.4g}"))

        df_ratio = pd.DataFrame([{
            "λ1 Δ(HbO₂−Hb)": delta1, "λ1 Δ_rel": rel1,
            "λ2 Δ(HbO₂−Hb)": delta2, "λ2 Δ_rel": rel2,
            "Signos opuestos": "Sí" if signs_opposite else "No",
            "Score (media)": score_mean, "Score (geom)": score_geo,
        }])
        st.dataframe(
            df_ratio.style.format({
                "λ1 Δ(HbO₂−Hb)": "{:.4g}", "λ1 Δ_rel": "{:.3f}",
                "λ2 Δ(HbO₂−Hb)": "{:.4g}", "λ2 Δ_rel": "{:.3f}",
                "Score (media)": "{:.3f}", "Score (geom)": "{:.3f}",
            }),
            hide_index=True
        )

        # semáforo de separación
        score = score_geo  # más exigente
        if signs_opposite and score >= 0.30:
            st.success(f"✅ Excelente separación espectral. Score={score:.3f} (signos opuestos).")
        elif score >= 0.15:
            st.info(f"ℹ️ Separación aceptable. Score={score:.3f}{' (signos opuestos)' if signs_opposite else ''}")
        else:
            st.warning(f"⚠️ Separación pobre. Score={score:.3f}. Prueba bandas más alejadas o cruza el signo.")

    # 8) Gráfico espectral (opcional)
    with st.expander('Molar extinction (zoom) + bandas'):
        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(df_zoom[col_lambda], df_zoom[col_hbo2], label="ε HbO₂")
        ax.plot(df_zoom[col_lambda], df_zoom[col_hb],   label="ε Hb")
        ax.axvline(λ1, linestyle="--", lw=2, label=f"HbO₂ ~ {λ1} nm")
        ax.axvline(λ2, linestyle="--", lw=2, label=f"Hb ~ {λ2} nm")
        ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("ε (a.u.)"); ax.grid(True); ax.legend()
        st.pyplot(fig)

    return λ1, λ2, HbO2_λ1, Hb_λ1, HbO2_λ2, Hb_λ2, E, band1, band2

def mbll_2w_delta_pixel(
    R_abs,
    band1, band2,
    eps_HbO2_1, eps_Hb_1,
    eps_HbO2_2, eps_Hb_2,
    baseline_frames=60,
    DPF1=1.0, DPF2=1.0,
    L=1.0,
    reflectance_path=None,
    out_root="/home/alonso/Desktop",
    chunk_size=32,
    mask_min_R=1e-4,
    clip_nonneg=True,
    show_progress=True,          # <-- nuevo
):
    

    # --- resolver carpeta (igual que antes) ---
    if reflectance_path is not None:
        base_dir  = os.path.dirname(os.path.abspath(reflectance_path))
        base_name = os.path.splitext(os.path.basename(reflectance_path))[0]
        run_dir   = os.path.join(base_dir, base_name)
    else:
        base_dir  = os.path.abspath(out_root)
        base_name = "mbll_run"
        run_dir   = os.path.join(base_dir, base_name)
    os.makedirs(run_dir, exist_ok=True)

    eps = 1e-12
    T, C, H, W = R_abs.shape
    n = min(baseline_frames, T)

    # --- E e inversa ---
    a11 = DPF1 * L * float(eps_HbO2_1); a12 = DPF1 * L * float(eps_Hb_1)
    a21 = DPF2 * L * float(eps_HbO2_2); a22 = DPF2 * L * float(eps_Hb_2)
    det = a11*a22 - a12*a21
    if abs(det) < 1e-12:
        raise ValueError("Matriz E mal condicionada (det≈0). Cambia λ1/λ2 o DPF.")
    inv11 =  a22 / det; inv12 = -a12 / det
    inv21 = -a21 / det; inv22 =  a11 / det

    # --- basal ---
    R1_base = np.clip(R_abs[:n, band1].astype(np.float32), mask_min_R, 1.0)
    R2_base = np.clip(R_abs[:n, band2].astype(np.float32), mask_min_R, 1.0)
    OD1_0 = -np.log(R1_base + eps).mean(axis=0)
    OD2_0 = -np.log(R2_base + eps).mean(axis=0)
    HbO2_0 = inv11*OD1_0 + inv12*OD2_0
    Hb_0   = inv21*OD1_0 + inv22*OD2_0
    if clip_nonneg:
        HbO2_0 = np.clip(HbO2_0, 0.0, None)
        Hb_0   = np.clip(Hb_0,   0.0, None)
    tHb_0  = np.clip(HbO2_0 + Hb_0, eps, None)
    good   = (tHb_0 > 10*eps).astype(np.float32)

    # --- memmaps ---
    dHbO2_path   = os.path.join(run_dir, "dHbO2.npy")
    dHb_path     = os.path.join(run_dir, "dHb.npy")
    HbO2_path    = os.path.join(run_dir, "HbO2.npy")
    Hb_path      = os.path.join(run_dir, "Hb.npy")
    StO2_path    = os.path.join(run_dir, "StO2.npy")
    StO2mean_path= os.path.join(run_dir, "StO2_mean.npy")
    dHbO2_mm = np.lib.format.open_memmap(dHbO2_path,   mode='w+', dtype=np.float32, shape=(T,H,W))
    dHb_mm   = np.lib.format.open_memmap(dHb_path,     mode='w+', dtype=np.float32, shape=(T,H,W))
    HbO2_mm  = np.lib.format.open_memmap(HbO2_path,    mode='w+', dtype=np.float32, shape=(T,H,W))
    Hb_mm    = np.lib.format.open_memmap(Hb_path,      mode='w+', dtype=np.float32, shape=(T,H,W))
    StO2_mm  = np.lib.format.open_memmap(StO2_path,    mode='w+', dtype=np.float32, shape=(T,H,W))
    StO2mean = np.lib.format.open_memmap(StO2mean_path,mode='w+', dtype=np.float32, shape=(T,))

    # --- barra de progreso ---
    n_chunks = ceil(T / float(chunk_size))
    bar_ph = st.empty() if show_progress else None
    txt_ph = st.empty() if show_progress else None
    bar = bar_ph.progress(0.0) if show_progress else None

    for k, t0 in enumerate(range(0, T, chunk_size), start=1):
        t1 = min(t0 + chunk_size, T)

        I1 = np.clip(R_abs[t0:t1, band1].astype(np.float32), mask_min_R, 1.0)
        I2 = np.clip(R_abs[t0:t1, band2].astype(np.float32), mask_min_R, 1.0)
        OD1 = -np.log(I1 + eps); OD2 = -np.log(I2 + eps)

        dOD1 = OD1 - OD1_0
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

        HbO2_blk *= good; Hb_blk *= good; StO2_blk *= good
        dHbO2_blk *= good; dHb_blk *= good

        dHbO2_mm[t0:t1] = dHbO2_blk
        dHb_mm[t0:t1]   = dHb_blk
        HbO2_mm[t0:t1]  = HbO2_blk
        Hb_mm[t0:t1]    = Hb_blk
        StO2_mm[t0:t1]  = StO2_blk
        StO2mean[t0:t1] = StO2_blk.reshape(t1-t0, -1).mean(axis=1).astype(np.float32)

        dHbO2_mm.flush(); dHb_mm.flush(); HbO2_mm.flush(); Hb_mm.flush(); StO2_mm.flush(); StO2mean.flush()

        if show_progress:
            frac = k / n_chunks
            bar.progress(min(frac, 1.0))
            txt_ph.text(f"MBLL: procesando bloques {k}/{n_chunks} (frames {t0}:{t1})")

    if show_progress:
        bar_ph.empty(); txt_ph.empty()
        st.success(f"MBLL listo: {run_dir}")

    # --- meta (igual que antes) ---
    meta = {
        "reflectance_path": reflectance_path,
        "out_dir": run_dir,
        "shape_input": [int(T), int(C), int(H), int(W)],
        "bands_used": {"band1": int(band1), "band2": int(band2)},
        "epsilons": {
            "eps_HbO2_1": float(eps_HbO2_1), "eps_Hb_1": float(eps_Hb_1),
            "eps_HbO2_2": float(eps_HbO2_2), "eps_Hb_2": float(eps_Hb_2)
        },
        "DPF": {"DPF1": float(DPF1), "DPF2": float(DPF2)},
        "L": float(L),
        "baseline_frames": int(baseline_frames),
        "chunk_size": int(chunk_size),
        "mask_min_R": float(mask_min_R),
        "clip_nonneg": bool(clip_nonneg),
        "note": "MBLL 2-λ con ancla basal por píxel; ΔOD = OD - OD0; x = E^{-1} y."
    }
    try:
        with open(os.path.join(run_dir, "mbll_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] No pude escribir mbll_meta.json: {e}")

    return {
        "folder": run_dir,
        "dHbO2_path": dHbO2_path,
        "dHb_path":   dHb_path,
        "HbO2_path":  HbO2_path,
        "Hb_path":    Hb_path,
        "StO2_path":  StO2_path,
        "StO2mean_path": StO2mean_path,
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

    




