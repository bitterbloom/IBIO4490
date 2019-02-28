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


import sys
sys.path.append('python')

import numpy as np
import cifar10
data = cifar10.load_cifar10(mode = 5)
images_train, labels_train = cifar10.get_data(data)

from fbCreate import fbCreate
fb = fbCreate(support=2, startSigma=0.6)

k = 16*32

from fbRun import fbRun
num = 200
sample = np.hstack(images_train[0:num,:,:])
filterResponses = fbRun(fb,sample)

from computeTextons import computeTextons
map, textons = computeTextons(filterResponses, k)

from assignTextons import assignTextons
texton_maps = []
for im in images_train:
    texton_maps.append(assignTextons(fbRun(fb,im),textons.transpose()))

from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators = 350)

hists = []
for txtmap in texton_maps:
    hists.append(histc(np.array(txtmap).flatten(),np.arange(k))/(32**2))
    
RF.fit(np.array(hists),labels_train)

predictions = RF.predict(np.array(hists))

from sklearn.metrics import confusion_matrix,accuracy_score
confusionmat = confusion_matrix(labels_train,predictions)
ACA = accuracy_score(labels_train,predictions)
print(f"ACA in train = {ACA}")

import pickle
model = open('RF','wb')
pickle.dump(RF,model)
model.close()
text = open('text','wb')
pickle.dump(textons,text)
text.close()


plt.figure()
plot_confusion_matrix(confusionmat, classes=range(1,11), normalize=True,
                      title='Normalized confusion matrix for train')

plt.show()

