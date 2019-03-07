#!/home/asvolcinschi/anaconda3/bin/ipython

def histc(X, bins):
    import numpy as np
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i-1] += 1
    return np.array(r)

def hist_intersection(hist1, hist2):
    count = 0
    for i in range(len(hist1)):
        count += min(hist1[i], hist2[i])
    return count

def metric(imageFile,colorSpace,method,k):
    import imageio
    import numpy as np
    from Segment import segmentByClustering
    im = imageio.imread(imageFile)
    seg = segmentByClustering(im,colorSpace,method,k)
    import scipy.io as sio
    gt=sio.loadmat(imageFile.replace('jpg', 'mat'))
    segm=gt['groundTruth'][0,2][0][0]['Segmentation']
    clustersTest = len(np.unique(seg))
    clustersLabel = len(np.unique(segm))
    if clustersLabel > clustersTest:
        histTest = np.histogram(seg,bins=np.arange(0,clustersTest))
        histLabel = np.histogram(segm,bins=np.linspace(0,clustersLabel+1,clustersTest))
    elif clustersTest > clustersLabel:
        histTest = np.histogram(seg,bins=np.linspace(0,clustersTest+1,clustersLabel))
        histLabel = np.histogram(segm,bins=np.arange(0,clustersLabel))
    else:
        histTest = np.histogram(seg,bins=np.arange(clustersTest))
        histLabel = np.histogram(segm,bins=np.arange(clustersLabel))

    histTest = np.sort(histTest[0])
    histLabel = np.sort(histLabel[0])
    intersect = round(hist_intersection(histTest,histLabel)/(im.shape[0]*im.shape[1]),4)
    return intersect