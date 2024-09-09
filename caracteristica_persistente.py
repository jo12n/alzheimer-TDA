# Leer el archivo .set usando MNE
raw = mne.io.read_raw_eeglab(archivo_set, preload=True)

# Indice del canal
idx_canal = 18

# Extraer los datos del canal
datos_canal, tiempos = raw[idx_canal, :]

# Guardar los datos del canal en un vector

vector_datos_canal = datos_canal[0,50000:200000]

N=150000
fs = 500
T = 1/fs
x= tiempos[50000:200000]
#x = np.linspace(0.0, N*T, N, endpoint=False)
yf = np.fft.fft(vector_datos_canal)
xf = np.fft.fftfreq(N, T)

frec_inf = 8  # Umbral de frecuencia en Hz
frec_sup = 12
mask_inf = np.abs(xf) >= frec_inf
mask_sup = np.abs(xf) <= frec_sup

yf_filtered = np.copy(yf)
yf_filtered = np.where(mask_inf, yf, 0)
yf_filtered = np.where(mask_sup, yf_filtered, 0)

# Reconstruir la senal temporal a partir de la transformada de Fourier filtrada

vector_datos_canal = np.fft.ifft(yf_filtered).real
barcode = barcode2(vector_datos_canal)
barcode = sorted(barcode, key=lambda x: x[1]-x[0], reverse=True)
barcode = barcode[50:]

ple = PersLandscapeApprox(dgms=[np.array(barcode)],hom_deg=0)

	with open(filename, "wb") as file:
pickle.dump(ple, file)
