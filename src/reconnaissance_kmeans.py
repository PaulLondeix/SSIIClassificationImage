import glob
import cv2
from sys import argv
from python_speech_features import logfbank
import shutil
from sklearn.cluster import KMeans
import numpy as np
import pickle

verbose = True

k1 = int(argv[1])

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

for img in listImg:
    if verbose:
        print("###", img, "###")
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if verbose:
        print("SIFT: ", descriptors)
    dimImg.append(len(keypoints))
    lesSIFT = np.append(lesSIFT, descriptors, axis=0)

try:
    with open("../resources/kmeans1.txt", "rb") as input:
        kmeans1 = pickle.load(input)
except EOFError:
    kmeans1 = KMeans(n_clusters=k1, random_state=0).fit(lesSIFT)
    with open("../resources/kmeans1.txt", "wb") as output:
        pickle.dump(kmeans1, output, pickle.HIGHEST_PROTOCOL)

bows = np.empty(shape=(0, k1), dtype=int)

i = 0
for nb in dimImg:  # for each sound (file)
    tmpBow = [0] * k1
    j = 0
    while j < nb:  # for each MFCC of this sound (file)
        tmpBow[kmeans1[i]] += 1
        j += 1
        i += 1
    copyBow = tmpBow.copy()
    bows = np.append(bows, [copyBow], 0)

print(bows)
