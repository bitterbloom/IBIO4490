#!/usr/bin/env python3
# -*- coding: utf-8 -*-
def histc(X, bins):
    import numpy as np
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i-1] += 1
    return np.array(r)

import matplotlib.pyplot as plt
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


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

texton_maps = []
for im in images_test:
    texton_maps.append(assignTextons(fbRun(fb,im),textons.transpose()))
    
hists = []
for txtmap in texton_maps:
    hists.append(histc(np.array(txtmap).flatten(),np.arange(k))/(32**2))

predictions = RF.predict(np.array(hists))

from sklearn.metrics import confusion_matrix,accuracy_score
confusionmat = confusion_matrix(labels_test,predictions)
ACA = accuracy_score(labels_test,predictions)
print(f"ACA in test = {ACA}")

plt.figure()
plot_confusion_matrix(confusionmat, classes=range(0,10), normalize=True,
                      title='Normalized confusion matrix for test')

plt.show()
<<<<<<< HEAD
plt.savefig("ConfMatTest.png")
=======
>>>>>>> 65ad25cf8ff9da1d467677266a45be5e2cf53ed4
