#!/bin/usr/python3

import scipy.io
import numpy as np
from random import randint
import os
import matplotlib.pyplot as plt
import cv2
from PIL import ImageDraw, Image
from shutil import copyfile, rmtree
import time

if not(os.path.isdir("./wiki")):
	os.system("wget https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki.tar.gz")
	os.system("tar -xvf wiki.tar.gz")

start = time.time()
os.chdir("wiki")
os.mkdir("./temp")
mat = scipy.io.loadmat('wiki.mat')
data = mat['wiki'][0,0].tolist()

numFotos = randint(7,17)
elegido = []
fotos = 0
while fotos < numFotos:

	rint = randint(0,len(data[0][0])+1)
	path = data[2][0,rint]

	im = plt.imread(f"{path[0]}")
	if im.shape == (1,1,4):
		pass
	else:
		copyfile(str(path[0]), f"./temp/pic_{fotos}.jpg")
		fotos += 1
		elegido.append(rint)

empezo = True
for i in elegido:
	
	if data[3][0,i] == 1:
		gender = "Male"
	else:
		gender = "Female"

	path = data[2][0,i]
	init = str(path[0])
	split = init.split("_")
	age = int(data[1][0,i])-int(split[1][0:4])

	im = Image.open(path[0]).convert("L")
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



