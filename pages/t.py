import pandas as pd

file_path = "/home/alonso/Desktop/Sc_pilot_1/mbll_run/sc_pilot_1_StO2.csv"

# Prueba lectura con autodetección de separador
try:
    df = pd.read_csv(file_path, sep=None, engine="python")
    print("✅ Cargado correctamente con shape:", df.shape)
    print("Columnas detectadas:", list(df.columns))
    print(df.head())
except Exception as e:
    print("❌ Error al leer CSV:", e)