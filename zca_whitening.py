from PIL import Image
from pylab import *
import numpy as np
import scipy as sp
from matplotlib import pylab as plt
import glob
import matplotlib.cm as cm
 
 
def flatten_matrix(matrix):
    vector = matrix.flatten(0)
    vector = vector.reshape(1, len(vector))
    return vector
 
def zca_whitening(inputs):
    sigma = np.dot(inputs.T, inputs)/float(inputs.shape[0])
    U,S,V = np.linalg.svd(sigma)
    epsilon = 0.1
    ZCAMatrix = np.dot(np.dot(U, np.diag(1.0/np.sqrt(S + epsilon))), U.T)
    np.savetxt( 'zca_white.txt', ZCAMatrix )
    return np.dot(ZCAMatrix, inputs.T)
 
fileNames = glob.glob('test/*')
image_size = 50
count = 0
for fileName in fileNames:
    im =  Image.open(fileName) 
    im1 = im.resize((image_size,image_size), Image.ANTIALIAS)
    img = np.array(im1)
    mean = np.mean(img)
    var = np.var(img)
    img = (img-mean)/float(sqrt(var))
    if count == 0:
        imgMat = flatten_matrix(img)
        count = count + 1
    else:
        imgMat = np.r_[imgMat, flatten_matrix(img)]
whitenImg = zca_whitening( imgMat )
 
 
print "load zca data"
ZCAMatrix = np.loadtxt( 'zca_white.txt' )
print "load image"
fileNames = glob.glob('test/*')
print "calc zca & save image"
count = 0
for fileName in fileNames:
    im =  Image.open(fileName) 
    im1 = im.resize((image_size,image_size), Image.ANTIALIAS)
    img = np.array(im1)
    w,h,c = img.shape
 
    #global contrast normalization?
    mean = np.mean(img)
    var = np.var(img)
    img = (img-mean)/float(sqrt(var))
 
    imgMat = flatten_matrix(img)
 
    whitenImg = np.dot(ZCAMatrix, imgMat.T)
    img = whitenImg.reshape([h,w,c])
 
 
    maxv = np.max(img)
    minv = np.min(img)
    img = (img-minv)/np.float(maxv-minv)*255
    img = img.astype(np.uint8)
    im2 = Image.fromarray(img, 'RGB')
 
    fn = "test_dataset2/"+fileNames[count].split("/")[1]
    count = count + 1
    im2.save(fn)