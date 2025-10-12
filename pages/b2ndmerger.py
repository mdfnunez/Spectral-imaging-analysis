#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, csv
import numpy as np
import easygui

# ─────────────────────── Config ───────────────────────
DEFAULT_DIR = "/home/alonso/Desktop/"   # default para seleccionar .b2nd

# ─────────────────────── Dependencias ───────────────────────
try:
    import blosc2
except Exception as e:
    easygui.msgbox(f"No pude importar blosc2:\n{e}\n\nInstala con:\n  pip install blosc2", "Error")
    sys.exit(1)

# ─────────────────────── Utilidades de lectura ───────────────────────
def _load_b2nd(path):
    """Devuelve np.ndarray desde .b2nd (open[:] o unpack_array2 como fallback)."""
    if hasattr(blosc2, "open"):
        try:
            return blosc2.open(path)[:]   # NDArray -> NumPy
        except Exception:
            pass
    with open(path, "rb") as fh:
        data = fh.read()
    return blosc2.unpack_array2(data)

def _detect_concat_axis(shapes):
    """Devuelve el eje que varía entre archivos. Si ninguno o >1, usa axis=0."""
    dims = np.array(shapes)
    var_axes = [ax for ax in range(dims.shape[1]) if not np.all(dims[:, ax] == dims[0, ax])]
    if len(var_axes) == 0:
        return 0
    if len(var_axes) == 1:
        return var_axes[0]
    easygui.msgbox(f"⚠️ Variación en varios ejes {var_axes}. Usaré axis=0.", "Aviso")
    return 0

# ─────────────────────── Fusión de METADATOS ───────────────────────
def _bytes_key(k):
    return k if isinstance(k, (bytes, bytearray)) else str(k).encode("utf-8", "ignore")

def _str_key(k):
    return k.decode("utf-8", "ignore") if isinstance(k, (bytes, bytearray)) else str(k)

def _to_array(v):
    # normaliza a ndarray (luego convertiremos a list para escribir)
    if isinstance(v, np.ndarray):
        return v
    if isinstance(v, (bytes, bytearray)):
        return np.array([bytes(v)], dtype=object)  # preserva binarios
    try:
        return np.asarray(v)
    except Exception:
        return np.array([v], dtype=object)

def _array_to_list(v):
    # convierte a lista de Python
    if isinstance(v, np.ndarray):
        # aplana (N,1) -> (N,)
        if v.ndim > 1 and v.shape[1] == 1:
            v = v[:, 0]
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]

def _read_all_vlmeta_dict(path):
    """Lee metadatos desde obj.vlmeta y obj.schunk.vlmeta; devuelve {bytes: ndarray}."""
    out = {}
    try:
        obj = blosc2.open(path)
        dicts = []
        if hasattr(obj, "vlmeta") and obj.vlmeta:
            dicts.append(obj.vlmeta)
        if hasattr(obj, "schunk") and getattr(obj.schunk, "vlmeta", None):
            dicts.append(obj.schunk.vlmeta)
        for d in dicts:
            for k, v in d.items():
                bk = _bytes_key(k)
                if bk not in out:   # primera aparición gana
                    out[bk] = _to_array(v)
    except Exception:
        pass
    return out

def _concat_per_frame_series(src_dicts, frames_per_file):
    """
    Devuelve {bytes: ndarray} concatenado por-frame.
    Regla: por archivo len==frames o 1 → per-frame; si no, toma el primero.
    """
    all_keys = set().union(*[set(d.keys()) for d in src_dicts if d])
    merged = {}
    for k in all_keys:
        vals = [d.get(k) for d in src_dicts]
        is_pf = True
        per_file = []
        for v, f in zip(vals, frames_per_file):
            if v is None:
                per_file.append(None); is_pf = False; continue
            v = _to_array(v)
            vlen = (len(v) if v.ndim >= 1 else 1)
            if vlen not in (1, f):
                is_pf = False
            per_file.append(v)

        if is_pf:
            expanded = []
            for v, f in zip(per_file, frames_per_file):
                if v is None:
                    expanded.append(np.array([None]*f, dtype=object))
                else:
                    vlen = (len(v) if v.ndim >= 1 else 1)
                    if vlen == 1:
                        expanded.append(np.repeat(v, f, axis=0))
                    else:
                        if v.ndim > 1 and v.shape[1] == 1:
                            v = v[:, 0]
                        expanded.append(v)
            try:
                merged[k] = np.concatenate(expanded, axis=0)
            except Exception:
                merged[k] = np.concatenate([np.asarray(x, dtype=object) for x in expanded], axis=0)
        else:
            first = next((v for v in per_file if v is not None), None)
            if first is not None:
                merged[k] = _to_array(first)
    return merged

def _xidec_write_merged_vlmeta(paths, out_path, axis_concat, shapes):
    """
    Funde metadatos de todos los 'paths' y los escribe en 'out_path'
    en dst.vlmeta y dst.schunk.vlmeta. Escribe claves en bytes **y** str,
    y valores como listas (máxima compatibilidad con tu app).
    """
    src_dicts = [_read_all_vlmeta_dict(p) for p in paths]
    if not any(src_dicts):
        return
    frames_per_file = [sh[axis_concat] for sh in shapes]

    merged = _concat_per_frame_series(src_dicts, frames_per_file)

    # Alias para claves esperadas por tu app
    alias = {
        b"time_stamp": [b"time_stamp", b"timestamp", b"time", b"TimeStamp"],
        b"exposure_us": [b"exposure_us", b"exposure", b"exposure_time"],
        b"temperature_chip": [b"temperature_chip", b"chip_temperature", b"sensor_temp"],
    }
    for dst_key, candidates in alias.items():
        if dst_key not in merged:
            for c in candidates:
                if c in merged:
                    merged[dst_key] = merged[c]
                    break

    # Preparar dos dicts: uno con claves bytes y otro con claves str; valores como listas
    dict_bytes_lists = {}
    dict_str_lists = {}
    for k, v in merged.items():
        pylist = _array_to_list(v)
        dict_bytes_lists[_bytes_key(k)] = pylist
        dict_str_lists[_str_key(k)] = pylist

    # Escribir en ambos lugares y reabrir (algunas versiones lo requieren)
    try:
        dst = blosc2.open(out_path, mode="a")
        if hasattr(dst, "vlmeta") and dst.vlmeta is not None:
            dst.vlmeta.update(dict_bytes_lists)
            dst.vlmeta.update(dict_str_lists)
        if hasattr(dst, "schunk") and getattr(dst.schunk, "vlmeta", None) is not None:
            dst.schunk.vlmeta.update(dict_bytes_lists)
            dst.schunk.vlmeta.update(dict_str_lists)
    except Exception as e:
        print("WARN(vlmeta write):", e)

    try:
        _ = blosc2.open(out_path, mode="r")  # re-open para “sellar” en algunas builds
    except Exception:
        pass

# ─────────────────────── Merge N archivos ───────────────────────
def merge_b2nd_many(paths, out_path, sort_mode="selection", chunk_hw=256):
    """
    paths: lista de rutas .b2nd (>=2)
    sort_mode: 'selection' (tal cual), 'name', 'mtime'
    """
    paths = list(dict.fromkeys(paths))  # sin duplicados
    assert len(paths) >= 2, "Se requieren ≥2 archivos."

    # Orden
    if sort_mode == "name":
        paths.sort(key=lambda p: os.path.basename(p))
    elif sort_mode == "mtime":
        paths.sort(key=lambda p: os.path.getmtime(p))
    # else: 'selection'

    # Cargar arrays
    arrays, shapes, dtypes = [], [], []
    for p in paths:
        a = _load_b2nd(p)
        arrays.append(a)
        shapes.append(a.shape)
        dtypes.append(a.dtype)

    # Dtype único
    dtypes_set = {str(dt) for dt in dtypes}
    if len(dtypes_set) != 1:
        raise ValueError(f"Dtypes distintos entre archivos: {dtypes_set}")

    # Eje de concatenación
    axis = _detect_concat_axis(shapes)

    # Consistencia en ejes no variables
    ref_shape = arrays[0].shape
    for ax in range(len(ref_shape)):
        if ax == axis:
            continue
        sizes = [a.shape[ax] for a in arrays]
        if not all(s == sizes[0] for s in sizes):
            raise ValueError(f"Dim {ax} no coincide entre archivos: {sizes}")

    # Concatena (RAM)
    merged = np.concatenate(arrays, axis=axis)

    # Chunks N-D (ND real en disco)
    if merged.ndim == 3:        # (T,H,W)
        chunks = (1, min(chunk_hw, merged.shape[1]), min(chunk_hw, merged.shape[2]))
    elif merged.ndim == 4:      # (T,H,W,C)
        chunks = (1, min(chunk_hw, merged.shape[1]), min(chunk_hw, merged.shape[2]), merged.shape[3])
    else:
        raise ValueError(f"ndim={merged.ndim} no soportado")

    # Persistir directo como .b2nd ND válido
    blosc2.asarray(merged, chunks=chunks, urlpath=out_path)

    # Fusionar metadatos (escribe en dst.vlmeta y dst.schunk.vlmeta)
    _xidec_write_merged_vlmeta(paths, out_path, axis, shapes)

    # Auditoría
    audit_csv = os.path.splitext(out_path)[0] + "_audit.csv"
    with open(audit_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "shape"])
        for p, s in zip(paths, shapes):
            w.writerow([os.path.basename(p), list(s)])

    return tuple(merged.shape), merged.dtype, axis, audit_csv

# ─────────────────────── Interfaz EasyGUI ───────────────────────
def main():
    easygui.msgbox("Selecciona ≥2 archivos .b2nd para unir", title="Unir N .b2nd")
    # default de selección en DEFAULT_DIR
    sel = easygui.fileopenbox("Selecciona .b2nd (Ctrl/Shift para múltiple)",
                              default=os.path.join(DEFAULT_DIR, "*.b2nd"),
                              filetypes=["*.b2nd"], multiple=True)
    if not sel:
        return
    if isinstance(sel, str):
        sel = [sel]
    sel = list(dict.fromkeys(sel))

    if len(sel) < 2:
        easygui.msgbox("Selecciona al menos 2 archivos.", "Aviso")
        return

    # Elegir orden
    choice = easygui.buttonbox(
        "¿Cómo ordenar los archivos para la unión?",
        choices=["Respetar selección", "Ordenar por nombre (A→Z)", "Ordenar por fecha (antiguo→nuevo)", "Cancelar"],
        title="Orden"
    )
    if not choice or choice == "Cancelar":
        return
    sort_mode = {"Respetar selección": "selection",
                 "Ordenar por nombre (A→Z)": "name",
                 "Ordenar por fecha (antiguo→nuevo)": "mtime"}[choice]

    # Confirmación visual (previa al orden real)
    listed = sel[:]
    if sort_mode == "name":
        listed = sorted(listed, key=lambda p: os.path.basename(p))
    elif sort_mode == "mtime":
        listed = sorted(listed, key=lambda p: os.path.getmtime(p))
    easygui.codebox("Archivos a unir (en orden):", text="\n".join(listed))

    # Directorio base para guardar = dir del primer archivo seleccionado
    base_dir = os.path.dirname(listed[0]) if listed else DEFAULT_DIR
    default_out = os.path.join(base_dir, "merged.b2nd")

    # Ruta de salida (default en el mismo directorio que los fuentes)
    outp = easygui.filesavebox("Guardar .b2nd unido como:",
                               default=default_out, filetypes=["*.b2nd"])
    if not outp:
        return

    try:
        shape, dtype, axis, audit = merge_b2nd_many(listed, outp, sort_mode=sort_mode)
        msg = [f"✅ Listo: {outp}",
               f"shape: {shape}   dtype: {dtype}",
               f"axis_concat: {axis}",
               f"Auditoría: {audit}"]
        easygui.msgbox("\n".join(map(str, msg)), "Éxito")
    except Exception as e:
        easygui.exceptionbox(f"Ocurrió un error uniendo los archivos:\n{e}", "Error")

if __name__ == "__main__":
    main()
