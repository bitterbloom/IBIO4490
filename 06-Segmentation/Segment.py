#!/home/asvolcinschi/anaconda3/bin/ipython

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from skimage import io,color
import math

def segmentByClustering(rgbImage, colorSpace, clusteringMethod, numberOfClusters):
    if colorSpace.lower() in ['rgb', 'lab', 'hsv', 'rgb+xy', 'lab+xy', 'hsv+xy']:
        if len(colorSpace) > 3:
            colorSpaceTemp = colorSpace.split("+")
            colorSpace = colorSpaceTemp[0]+colorSpaceTemp[1]
        imageAverage = eval(colorSpace)(rgbImage)
    else:
        raise Exception("Invalid colorSpace argument")
    
    if clusteringMethod.lower() in ['kmeans', 'gmm', 'hierarchical', 'watershed']:
        temp = eval(clusteringMethod)(imageAverage,numberOfClusters)
        seg = temp.reshape(rgbImage.shape[0],rgbImage.shape[1])
    else:
        raise Exception("Invalid clusteringMethod argument")


    return seg

def rgb(image):
    return(norm(image))

def lab(image):
    image = color.rgb2lab(image)
    return(norm(image))

def hsv(image):
    image = color.rgb2hsv(image)
    return(norm(image))

def rgbxy(image):
    return(norm(addxy(image)))

def labxy(image):
    image = color.rgb2lab(image)
    return(norm(addxy(image)))

def hsvxy(image):
    image = color.rgb2hsv(image)
    return(norm(addxy(image)))

def addxy(image):
    shape = image.shape
    height = shape[0]
    width = shape[1]
    x,y = np.mgrid[0:height,0:width]
    image = np.dstack((image,x,y))
    return image

def norm(image):
    for channel in range(0,image.shape[2]):
        maximum = np.max(image[:,:,channel])
        minimum = np.min(image[:,:,channel])
        newMaximum = 255
        newMinimum = 0
        for i in range(0,image.shape[0]):
            for j in range(0,image.shape[1]):
                image[i,j,channel] = math.floor(((newMaximum - newMinimum)*\
                    (image[i,j,channel]-minimum))/(maximum-minimum))+newMinimum
    
    newImage = np.zeros((image.shape[0],image.shape[1]))
    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
            newImage[i,j] = np.average(image[i,j,:])
    
    return newImage

def kmeans(data,k):
    from sklearn.cluster import KMeans
    data = data.flatten().reshape(-1,1)
    kmean = KMeans(n_clusters=k)
    kmean.fit(data)
    labels = kmean.labels_
    
    return labels

def gmm(data,k):
    from sklearn import mixture
    data = data.flatten().reshape(-1,1)
    gauss = mixture.GaussianMixture(n_components=k)
    gauss.fit(data)
    labels = gauss.predict(data)
    labels = labels.astype(np.int64)
    return labels

def hierarchical(data,k):
    from sklearn.cluster import AgglomerativeClustering
    from scipy.misc import imresize
    data = imresize(data,(160,240))
    data = data.flatten().reshape(-1,1)
    hier = AgglomerativeClustering(n_clusters=k,linkage='ward')  
    hier.fit_predict(data)
    labels = hier.labels_
    labels = np.reshape(labels,(160,240))
    labels = imresize(labels,(321,481),interp="nearest")
    labels = labels/50
    labels = labels.astype(np.int64)
    return labels

def watershed(data,k):
    from skimage.feature import peak_local_max
    temp = 255*np.ones(data.shape)
    noK = True
    count = 0
    while noK:
        local_max = peak_local_max(temp-data,num_peaks=k+count,indices=False)
        from skimage.morphology import watershed
        import scipy.ndimage as ndi
        markers = ndi.label(local_max)[0]
        labels = watershed(data,markers)
        labels = labels - np.ones(data.shape)
        labels = labels.astype(np.int64)
        if (len(np.unique(labels))) == k:
            noK = False
        count += 1
    return labels