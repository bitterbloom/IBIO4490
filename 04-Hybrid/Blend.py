#!/bin/usr/python3

import skimage
import matplotlib.pyplot as plt
import numpy as np  
import skimage
import pdb
from skimage.transform import rescale
import warnings

warnings.filterwarnings("ignore")

Im_papa = skimage.io.imread("./imgs/papa.jpg")
Im_yo = skimage.io.imread("./imgs/yo.jpg")

Im_papa = skimage.transform.resize(Im_papa,(1024,1024),mode = "reflect", preserve_range = True)
Im_yo = skimage.transform.resize(Im_yo,(1024,1024),mode = "reflect", preserve_range = True)

Gauss_papa = {"1024":Im_papa}
Gauss_yo = {"1024":Im_yo}

for G in range(1,6):
	Im_papa = skimage.transform.rescale(Im_papa,0.5,mode = "reflect",multichannel = True)
	Gauss_papa[str(Im_papa.shape[0])] = Im_papa
	Im_yo = skimage.transform.rescale(Im_yo,0.5,mode = "reflect",multichannel = True)
	Gauss_yo[str(Im_yo.shape[0])] = Im_yo

Lapl_papa = {}
Lapl_yo = {}
# pdb.set_trace()
for L in Gauss_papa.keys():
	if L == "32":
		continue
	else:
		temp = int(int(L)/2)
		LT_papa = skimage.transform.rescale(Gauss_papa[str(temp)],2,mode = "reflect",
															    multichannel = True)
		LT_yo = skimage.transform.rescale(Gauss_yo[str(temp)],2,mode = "reflect",
															    multichannel = True)
		# pdb.set_trace()
		L_papa = Gauss_papa[L] - LT_papa
		L_yo = Gauss_yo[L] - LT_yo
		L_papa[L_papa < 0] = 0
		L_yo[L_yo < 0] = 0

		Lapl_papa[L] = L_papa
		Lapl_yo[L] = L_yo

Scales = [2**x for x in range(6,11)]
Blend = np.hstack((Gauss_papa["32"][:,0:16,:],Gauss_yo["32"][:,16:,:]))
for scale in Scales:
	Blend = skimage.transform.rescale(Blend,2,mode = "reflect",multichannel = True)
	
	Frec = np.hstack((Lapl_papa[str(scale)][:,0:int(int(scale)/2),:],Lapl_yo[str(scale)][:,int(int(scale)/2):,:]))
	Blend = Blend + Frec

plt.imshow(np.uint8(Blend))
plt.show()

skimage.io.imsave("./imgs/blend.png",np.uint8(Blend))