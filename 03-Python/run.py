#!/bin/usr/python3

import scipy.io
import numpy as np
from random import randint
import os
import matplotlib.pyplot as plt
from PIL import ImageDraw, Image
from shutil import copyfile, rmtree
import time
import urllib
import zipfile
import ipdb

if not(os.path.isdir("./00")):

	url = 'https://www.dropbox.com/s/l5rq8s0baedu44f/wiki.zip?dl=1'
	u = urllib.request.urlopen(url)
	data = u.read()
	u.close()
	with open(os.getcwd() + '/' + 'wiki.zip', "wb") as f :
	    f.write(data)
	f.close()

	zip_ref = zipfile.ZipFile(os.getcwd() + '/' + 'wiki.zip', 'r')
	zip_ref.extractall(os.getcwd())
	zip_ref.close()

start = time.time()
os.mkdir("./temp")
mat = scipy.io.loadmat('wiki.mat')
data = mat['wiki'][0,0].tolist()

numFotos = randint(7,17)
elegido = []
fotos = 0
path_00 = os.listdir("./00")
while fotos < numFotos:

	rint = randint(0,600)
	path = path_00[rint]

	im = plt.imread(f"./00/{path}")
	if im.shape == (1,1,4):
		pass
	else:
		copyfile(f"./00/{path}", f"./temp/pic_{fotos}.jpg")
		fotos += 1
		elegido.append(rint)

empezo = True
for i in elegido:
	path = path_00[i]
	fila, num= np.where(data[2]==f"00/{path}")
	if data[3][0,num] == 1:
		gender = "Male"
	else:
		gender = "Female"

	split = path.split("_")
	age = int(data[1][0,num])-int(split[1][0:4])

	im = Image.open(f"./00/{path}").convert("L")
	imResize = im.resize((256,256))

	dwp = ImageDraw.Draw(imResize)
	dwp.rectangle(((110,110),(160,150)),fill=(140))
	dwp.text((118,118),f"{gender}\n{age}",fill=(255))

	if empezo:
		empezo = False
		rta = imResize
	else:
		rta = np.hstack([rta,imResize])

imagenFinal = rta[:,0:256*4]
restantes = numFotos-4
for i in range(5,numFotos+1,4):
	if restantes > 3:
		imagenFinal = np.vstack((imagenFinal,rta[:,256*(i-1):256*(i+3)]))
		restantes -= 4
	else:
		imagenes = rta[:,256*(i-1):]
		vacio = np.zeros([256,256*(4-restantes)])
		ad = np.hstack((imagenes,vacio))
		ad = ad.astype(np.uint8)
		imagenFinal = np.vstack((imagenFinal,ad))

finish = time.time()
print(f"El algoritmo tardó {finish-start} seconds")
rmtree("./temp")
print(f"Se presentaron {numFotos} fotos aleatorias de la base de datos Wiki mostrando etiquetas de género y edad")	

plt.figure(figsize=(15,15))
plt.imshow(imagenFinal, cmap='gray')
plt.show()



