import math
import numpy as np
import pylab
import utils
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import img_as_float
from skimage.measure import compare_psnr
from sklearn.cluster import KMeans


def header():
    return 'WEEK 6: Clusterization and Visualization'

def run():

    clusterization()

    return

def clusterization():

    image = imread(utils.PATH.MATERIALS_FILE('parrots.jpg'))
    #pylab.imshow(image)
    #plt.show()

    image = img_as_float(image)
    n_height   = image.shape[0]
    n_width    = image.shape[1]
    n_channels = image.shape[2]

    X = image.reshape([n_height*n_width, n_channels])
    print(X.shape)

    for n_clusters in range(2, 21):
        km = KMeans(n_clusters=n_clusters, init='k-means++', random_state=241, n_jobs=-1)
        km.fit(X)

        X_mean = np.zeros_like(X)
        X_med  = np.zeros_like(X)

        for i in range(km.n_clusters):
            cluster = km.labels_==i
            cluster_i = X[cluster]
            X_mean[cluster] = np.mean(cluster_i, axis=0)
            X_med[cluster]  = np.median(cluster_i, axis=0)

        image_mean = X_mean.reshape([n_height, n_width, n_channels])
        image_med  = X_med.reshape([n_height, n_width, n_channels])

        psnr_mean = psnr_metric(image, image_mean)
        psnr_med = psnr_metric(image, image_med)

        print(n_clusters)
        print(psnr_mean)
        print(psnr_med)

        # check
        #psnr_mean = compare_psnr(image, image_mean)
        #psnr_med = compare_psnr(image, image_med)
        #
        #print(n_clusters)
        #print(psnr_mean)
        #print(psnr_med)


    pylab.imshow(image_mean)
    plt.show()

    pylab.imshow(image_med)
    plt.show()

    return

def psnr_metric(X1, X2):
    X = X1 - X2
    n, m, c = X.shape[0], X.shape[1], X.shape[2]

    mse = np.sum(X**2)/(n*m*c)
    maxi = 1

    return 20*math.log10(maxi) - 10*math.log10(mse)
