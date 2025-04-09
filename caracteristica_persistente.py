import numpy as np
import mne
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import cm
import tadasets
from barcode2 import barcode2
import time
from persim import PersLandscapeApprox, PersLandscapeExact
from persim.landscapes import plot_landscape_simple
import pickle

inicio = time.perf_counter()
vector = [1,2,4,5,6,7,9,10,14,15,16,17,18,19,20,21,24,25,27,28,30,31,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,57,58,59,60,61,62,64,65]

for i in vector:
    if i<10:
        archivo_set = fr"D:\Mates\Master\TFM\datos_alz\sub-00{i}\eeg\sub-00{i}_task-eyesclosed_eeg.set"
        filename = fr"D:\Mates\Master\TFM\datos_alz\PL_008\landscape00{i}.pk1"
    else:
        archivo_set = fr"D:\Mates\Master\TFM\datos_alz\sub-0{i}\eeg\sub-0{i}_task-eyesclosed_eeg.set"
        filename = fr"D:\Mates\Master\TFM\datos_alz\PL_008\landscape0{i}.pk1"


# Read the file .set ussing MNE
    raw = mne.io.read_raw_eeglab(archivo_set, preload=True)

# The channel from where we extract the data
    idx_canal = 18

    datos_canal, tiempos = raw[idx_canal, :]

# datos_canal is a NumPy array with the data of the channel
# tiempos is an array of time
	
# Save the data of the channel in other array

    vector_datos_canal = datos_canal[0,50000:200000]
    
    N=150000
    fs = 500
    T = 1/fs
    x= tiempos[50000:200000]
    yf = np.fft.fft(vector_datos_canal)
    xf = np.fft.fftfreq(N, T)

    frec_inf = 8  # Frequencies fot the FFT
    frec_sup = 12
    mask_inf = np.abs(xf) >= frec_inf
    mask_sup = np.abs(xf) <= frec_sup

    yf_filtered = np.copy(yf)
    yf_filtered = np.where(mask_inf, yf, 0)
    yf_filtered = np.where(mask_sup, yf_filtered, 0)

    # Build the signal again after the FFT

    vector_datos_canal = np.fft.ifft(yf_filtered).real
    barcode = barcode2(vector_datos_canal)
    barcode = sorted(barcode, key=lambda x: x[1]-x[0], reverse=True)
    barcode = barcode[50:]

    ple = PersLandscapeApprox(dgms=[np.array(barcode)],hom_deg=0)

    with open(filename, "wb") as file:
        pickle.dump(ple, file)

fin = time.perf_counter()
tiempo_total = fin - inicio
print(tiempo_total)
