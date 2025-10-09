import streamlit as st
import pandas as pd
from io import BytesIO

st.title("🔗 Merge Multiple CSV/XLSX by Timestamp into Excel")

# --- Función robusta para leer CSV o XLSX ---
def safe_read(file):
    try:
        if file.name.endswith(".xlsx"):
            df = pd.read_excel(file)
        else:
            df = pd.read_csv(file, sep=None, engine="python", on_bad_lines="skip")
            if df.empty or (len(df.columns) == 1 and df.columns[0].startswith("Unnamed")):
                file.seek(0)
                df = pd.read_csv(file, sep=";", engine="python", on_bad_lines="skip")
        if df.empty or len(df.columns) == 0:
            st.warning(f"⚠️ File '{file.name}' is empty or invalid, skipped.")
            return pd.DataFrame()
        return df
    except pd.errors.EmptyDataError:
        st.warning(f"⚠️ File '{file.name}' is empty, skipped.")
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"⚠️ Could not read '{file.name}': {e}")
        return pd.DataFrame()

# --- Subida de archivos ---
st.subheader("Upload your data files")
files_to_add = st.file_uploader(
    "Upload main files (receivers)", 
    type=["csv", "xlsx"], 
    accept_multiple_files=True
)
file_donor = st.file_uploader(
    "Upload donor file (with info to add)", 
    type=["csv", "xlsx"]
)

if files_to_add and file_donor:
    df_donor = safe_read(file_donor)
    if df_donor.empty:
        st.error("❌ The donor file is empty or unreadable.")
        st.stop()

    # Usa el primer archivo no vacío como ejemplo
    example_df = None
    for f in files_to_add:
        temp_df = safe_read(f)
        if not temp_df.empty:
            example_df = temp_df
            break

    if example_df is None:
        st.error("❌ All receiver files are empty or invalid.")
        st.stop()

    st.subheader("Select timestamp columns")
    left_col = st.selectbox("Timestamp column in receiver files", example_df.columns)
    right_col = st.selectbox("Timestamp column in donor file", df_donor.columns)

    st.caption("Example formats: 20241018_15-14-49-454 or 15:10:50 or 15:10")

    # --- Slider de tolerancia ---
    tolerance_seconds = st.slider(
        "Select time matching tolerance (seconds)", 
        min_value=1.0, 
        max_value=120.0, 
        value=1.0, 
        step=1.0
    )

    # --- Opción de fecha manual ---
    date_str = st.text_input(
        "If donor file has only time (HH:MM or HH:MM:SS), type date YYYYMMDD (optional):", 
        ""
    )

    if st.button("🔄 Merge and Export to Excel"):
        # --- Funciones de parseo ---
        def parse_left(x):
            try:
                return pd.to_datetime(x, format="%Y%m%d_%H-%M-%S-%f", errors="coerce")
            except Exception:
                return pd.to_datetime(x, errors="coerce")

        def parse_right(x, base_date=None):
            # Si el CSV donador solo tiene hora, usa fecha base
            if base_date is not None:
                x = base_date + " " + x.astype(str)
            return pd.to_datetime(x, errors="coerce")

        # --- Procesar archivo donador ---
        base_date_auto = None

        # Procesar el primer receptor para obtener fecha base automática si no se especificó
        df_main_first = safe_read(files_to_add[0])
        if not df_main_first.empty:
            parsed = pd.to_datetime(df_main_first[left_col].astype(str), errors="coerce")
            valid_dates = parsed.dropna()
            if not valid_dates.empty:
                first_date = valid_dates.iloc[0]
                base_date_auto = first_date.strftime("%Y%m%d")
                st.info(f"📅 Using auto-detected base date from receiver: **{base_date_auto}**")

        base_date = date_str if date_str else base_date_auto

        df_donor["_t"] = parse_right(df_donor[right_col].astype(str), base_date)
        df_donor = df_donor.dropna(subset=["_t"]).sort_values("_t")

        # --- Crear Excel en memoria ---
        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            processed_files = 0

            for file in files_to_add:
                df_main = safe_read(file)
                if df_main.empty:
                    continue

                df_main["_t"] = parse_left(df_main[left_col].astype(str))
                df_main = df_main.dropna(subset=["_t"]).sort_values("_t")

                if df_main.empty:
                    st.warning(f"⚠️ No valid timestamps in '{file.name}', skipped.")
                    continue

                # Merge tipo nearest con tolerancia ajustable
                merged = pd.merge_asof(
                    df_main,
                    df_donor,
                    on="_t",
                    direction="nearest",
                    tolerance=pd.Timedelta(seconds=tolerance_seconds)
                )

                sheet_name = file.name.replace(".csv", "").replace(".xlsx", "")[:31]
                merged.to_excel(writer, index=False, sheet_name=sheet_name)
                processed_files += 1

        if processed_files == 0:
            st.error("❌ No valid data merged. Check your timestamps or formats.")
        else:
            st.success(f"✅ Merged {processed_files} files into one Excel file!")

            st.download_button(
                label="📘 Download merged Excel file",
                data=output.getvalue(),
                file_name="merged_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
