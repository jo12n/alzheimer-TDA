import numpy as np
import time
import scipy as sp
import matplotlib.pyplot as plt
from persim.landscapes import plot_landscape_simple
from matplotlib import cm
import tadasets
from barcode_ST import barcode_st
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from persim import PersLandscapeApprox, PersLandscapeExact
from persim.landscapes import plot_landscape_simple
from persim.landscapes import snap_pl
import pickle
import os
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

inicio = time.perf_counter()

def load_PL(folder_path, label):
    data = []
    labels = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'rb') as file:
            loaded_data = pickle.load(file)
            data.append(loaded_data)
            labels.append(label)
    return data, labels

data_alz, label_alz = load_PL(fr"D:\Mates\Master\TFM\datos_alz\PL_006\PL_alz", label=0)
data_control, label_control = load_PL(fr"D:\Mates\Master\TFM\datos_alz\PL_006\PL_control", label=1)

data_alz2, label_alz2 = load_PL(fr"D:\Mates\Master\TFM\datos_alz\PL_007\PL_alz", label=0)
data_control2, label_control2 = load_PL(fr"D:\Mates\Master\TFM\datos_alz\PL_007\PL_control", label=1)

data_alz3, label_alz3 = load_PL(fr"D:\Mates\Master\TFM\datos_alz\PL_008\PL_alz", label=0)
data_control3, label_control3 = load_PL(fr"D:\Mates\Master\TFM\datos_alz\PL_008\PL_control", label=1)

datos2 = data_alz2 + data_control2
datos3 = data_alz3 + data_control3
datos = data_alz + data_control
labels = label_alz + label_control
datos = np.array(datos)
datos2 = np.array(datos2)
datos3 = np.array(datos3)
labels = np.array(labels)

max_dim = max([len(pl.values) for pl in datos])*500
max_dim2 = max([len(pl.values) for pl in datos2])*500
max_dim3 = max([len(pl.values) for pl in datos3])*500
max_dim = max(max_dim,max_dim2,max_dim3)
print(max_dim)

for i in range(0,len(datos)):
    datos[i] = datos[i].values
    datos2[i] = datos2[i].values
    datos3[i] = datos3[i].values
    """
    datos[i] = np.vstack(datos[i].T)
    scaler = StandardScaler()
    datos[i] = scaler.fit_transform(datos[i])
    pca = PCA(n_components=500)
    datos[i] = pca.fit_transform(datos[i])
    """
    datos[i] = datos[i].flatten()
    datos2[i] = datos2[i].flatten()
    datos3[i] = datos3[i].flatten()
    size = max_dim - len(datos[i])
    size2 = max_dim - len(datos2[i])
    size3 = max_dim - len(datos3[i])
    datos[i] = np.concatenate((datos[i], np.zeros(size)))
    datos2[i] = np.concatenate((datos2[i], np.zeros(size2)))
    datos3[i] = np.concatenate((datos3[i], np.zeros(size3)))
    datos[i] = 3*datos[i]+3*datos2[i]+2*datos3[i]

datos = np.array(np.vstack(datos))
scaler = StandardScaler()
datos = scaler.fit_transform(datos)

skf = StratifiedKFold(n_splits=5)
prediccion = labels.copy()

for train_index, test_index in skf.split(datos, labels):
    clf = RandomForestClassifier(n_estimators=100000, random_state=14)
    clf.fit(datos[train_index], labels[train_index])
    y_pred = clf.predict(datos[test_index])
    for i in range(0,len(test_index)):
        prediccion[test_index[i]] = y_pred[i]

print(labels)
print(prediccion)
cm = confusion_matrix(labels, prediccion)

fin = time.perf_counter()
tiempo_total = fin - inicio
print(tiempo_total) 

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Alzheimer', 'Control'], yticklabels=['Alzheimer', 'Control'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()