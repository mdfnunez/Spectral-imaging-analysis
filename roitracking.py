import streamlit as st 
import numpy as np
import pandas as pd 
import os 
import cv2

# Define la información de los ROIs
roi_info = [
    {'x': 126, 'y': 392, 'ancho': 78, 'alto': 36},  # ROI 1
    {'x': 758, 'y': 116, 'ancho': 78, 'alto': 36},  # ROI 2
    {'x': 358, 'y': 644, 'ancho': 78, 'alto': 36},  # ROI 3
    {'x': 872, 'y': 488, 'ancho': 78, 'alto': 36},  # ROI 4
]

# Directorio donde se encuentran tus imágenes
directorio = '/Volumes/Alonsos SSD/Experimental_data/18.06.24SCIMSI2-raiz.msi.1 18 de jun 2024/18.06.24 SCIMSI2-cervical-collateral-root/2 Normotensive_collateral'

# Crea un DataFrame vacío para almacenar los resultados
resultados = pd.DataFrame(columns=['Archivo', 'ROI', 'Media_Gris'])

# Lista de archivos de imágenes en el directorio
imagenes = [os.path.join(directorio, archivo) for archivo in os.listdir(directorio) if archivo.endswith('.tif')]
imagenes.sort()  # Asegúrate de que las imágenes estén en orden

# Inicializa trackers para cada ROI
trackers = [cv2.TrackerKCF_create() for _ in roi_info]

# Leer la primera imagen para inicializar los trackers
primer_frame = cv2.imread(imagenes[0])

# Inicializar cada tracker con el ROI correspondiente
for i, roi_data in enumerate(roi_info):
    x = roi_data['x']
    y = roi_data['y']
    ancho = roi_data['ancho']
    alto = roi_data['alto']
    roi = (x, y, ancho, alto)
    trackers[i].init(primer_frame, roi)

# Procesar cada imagen y seguir los ROIs
for archivo in imagenes:
    frame = cv2.imread(archivo)

    # Procesa cada ROI
    for i, tracker in enumerate(trackers, start=1):
        success, roi = tracker.update(frame)

        if success:
            x, y, w, h = map(int, roi)
            roi = frame[y:y+h, x:x+w]

            # Calcular la media de la escala de grises
            media_gris = np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))

            # Añade los resultados al DataFrame
            resultados = resultados.append({'Archivo': archivo, 'ROI': f'ROI {i}', 'Media_Gris': media_gris}, ignore_index=True)

            # Dibujar el ROI seguido en la imagen
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2, 1)
        else:
            print(f"Fallo en el seguimiento del ROI {i}")

    # Mostrar el frame con los ROIs seguidos (opcional)
    cv2.imshow("Seguimiento de ROIs", frame)

    # Esperar una tecla para continuar
    if cv2.waitKey(30) & 0xFF == 27:  # Presiona ESC para salir
        break

cv2.destroyAllWindows()

# Guarda los resultados en un archivo CSV
resultados.to_excel('Normot_muscle_180624.xlsx', index=False)


