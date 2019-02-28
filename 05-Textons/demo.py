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
for i in range(1,11,2):
    
    rand = random.randint(0,10000)
    im = images_test[rand,:,:]
    
    txtmap = assignTextons(fbRun(fb,im),textons.transpose())
    hist = histc(np.array(txtmap).flatten(),np.arange(k))/(32**2)
    prediction = RF.predict(np.array(hist))
    
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
    
    plt.subplot(520+i)
    plt.imshow(images_test[rand,:,:],cmap="gray")
    plt.title(f"Classified as: {label}")
    plt.subplot(520+i+1)
    plt.imshow(txtmap)
    plt.title("Texton map")
    
plt.show()