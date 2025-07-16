import numpy as np
file="/home/alonso/Documents/GitHub/Spectral-imaging-analysis/StO2_data.npz" 
files=np.load(file)
files_reflectance=files["file_names_reflectance"]
print(files_reflectance)