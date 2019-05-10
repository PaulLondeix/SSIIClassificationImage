import glob
import cv2
from sys import argv
from python_speech_features import logfbank
import shutil
from sklearn.cluster import KMeans
import numpy as np

verbose = True
k1 = int(argv[1])
k2 = int(argv[2])

image = cv2.imread('../resources/base_samples/citron1.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)
print('#########################')
print(keypoints)
print('#########################')
print(descriptors)
print('#########################')
print(sift)
print('#########################')
print('nb. of keypoints: ', len(keypoints))

lesSIFT = []


dimImg = []  # nb of mfcc per file
listImg = glob.glob("../resources/base_samples/*.jpg")

'''for img in listImg:
    if verbose:
        print("###", img, "###")
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if verbose:
        print("MFCC: ", mfcc_feat.shape)
    dimImg.append(len(keypoints))
    lesSIFT = np.append(lesSIFT, mfcc_feat, axis=0)

try:
    with open("../resources/kmeans1.txt", "rb") as input:
        kmeans1 = pickle.load(input)
except EOFError:
    exit(-1)

bows = np.empty(shape=(0, k1), dtype=int)

kmeans1new = kmeans1.predict(lesMfcc)

i = 0
for nb in dimSons:  # for each sound (file)
    tmpBow = [0] * k1
    j = 0
    while j < nb:  # for each MFCC of this sound (file)
        tmpBow[kmeans1new[i]] += 1
        j += 1
        i += 1
    copyBow = tmpBow.copy()
    bows = np.append(bows, [copyBow], 0)

print(bows)

try:
    with open("../resources/kmeans2.txt", "rb") as input:
        kmeans2 = pickle.load(input)
except EOFError:
    exit(-1)


print(kmeans2.predict(bows))'''
