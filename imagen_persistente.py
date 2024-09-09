def imagen_pers(barcode,resolucion,guardado):
	for i in range(len(barcode)):
		barcode[i][1] = barcode[i][1] - barcode[i][0]
	max_nac = max([v[0] for v in barcode])
	min_nac = min([v[0] for v in barcode])
	max_TV = max([v[1] for v in barcode])
	min_TV = min([v[1] for v in barcode])
	rango_foto = max(max_nac-min_nac,max_TV-min_TV)
	imagen = Image.new('RGB', (resolucion,resolucion), 'white')
	pixeles = imagen.load()
	matriz = np.zeros((resolucion, resolucion))
	k=0
	l=0
	paso = (1.2*rango_foto)/(resolucion-1)
	for i in np.arange(min_nac-0.1*rango_foto, min_nac+1.1*rango_foto, paso):
		for j in np.arange(min_TV-0.1*rango_foto, min_TV+1.1*rango_foto, paso):
			matriz[k,l] = crear_imagen(barcode,i,i+paso,j,j+paso,rango_foto)
			k=k+1
		k=0
		l=l+1
	maximal = np.amax(matriz)
	for i in range(resolucion):
		for j in range(resolucion):
			pixeles[j,i] = (round(255*matriz[i,j]/maximal),0,100)
	imagen = imagen.transpose(Image.FLIP_TOP_BOTTOM)
	imagen.save(guardado)
	return
