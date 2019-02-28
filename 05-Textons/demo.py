#!/usr/bin/env python3
# -*- coding: utf-8 -*-
def histc(X, bins):
    import numpy as np
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i-1] += 1
    return np.array(r)

import pickle
model = open('RF','rb')
RF = pickle.load(model)
model.close()
text = open("text","rb")
textons = pickle.load(text)
text.close()

import sys
sys.path.append('python')

import numpy as np
import cifar10
data = cifar10.load_cifar10(mode = "test")
images_test, labels_test = cifar10.get_data(data)

from fbCreate import fbCreate
fb = fbCreate(support=2, startSigma=0.6)
k = 16*32

from assignTextons import assignTextons
from fbRun import fbRun
import matplotlib.pyplot as plt
import random

plt.figure()
for i in range(1,8,2):
    
    rand = random.randint(0,10000)
    im = images_test[rand,:,:]
    lab = labels_test[rand]
    
    txtmap = assignTextons(fbRun(fb,im),textons.transpose())
    hist = []
    hist.append(histc(np.array(txtmap).flatten(),np.arange(k))/(32**2))
    prediction = RF.predict(np.array(hist))
    
    if lab == 0:
        labe = "airplane"
    elif lab == 1:
        labe = "automobile"
    elif lab == 2:
        labe = "bird"
    elif lab == 3:
        labe = "cat"
    elif lab == 4:
        labe = "deer"
    elif lab == 5:
        labe = "dog"
    elif lab == 6:
        labe = "frog"
    elif lab == 7:
        labe = "horse"
    elif lab == 8:
        labe = "ship"
    elif lab == 9:
        labe = "truck"
    
    if prediction == 0:
        label = "airplane"
    elif prediction == 1:
        label = "automobile"
    elif prediction == 2:
        label = "bird"
    elif prediction == 3:
        label = "cat"
    elif prediction == 4:
        label = "deer"
    elif prediction == 5:
        label = "dog"
    elif prediction == 6:
        label = "frog"
    elif prediction == 7:
        label = "horse"
    elif prediction == 8:
        label = "ship"
    elif prediction == 9:
        label = "truck"
    
    plt.subplot(int(420+i))
    plt.imshow(images_test[rand,:,:],cmap="gray")
    plt.title(f"{labe.title()} classified as {label}")
    plt.subplot(int(420+(i+1)))
    plt.imshow(txtmap)
    plt.title("Texton map")

plt.subplots_adjust(wspace=0.1,hspace=0.6)
plt.show()
