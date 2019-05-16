import glob
import cv2
from sys import argv
from sklearn.cluster import KMeans
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

verbose = False

k1 = int(argv[1])

lesSIFT = np.empty(shape=(0, 128), dtype=float)

labels = []
dimImg = []  # nb of mfcc per file
listImg = glob.glob("../resources/test_samples/*.jpg")

for img in listImg:
    labels.append(0) if img[26] == 'c' else labels.append(1)
    if verbose:
        print("###", img, "###")
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if verbose:
        print("SIFT: ", descriptors)
    dimImg.append(len(descriptors))
    lesSIFT = np.append(lesSIFT, descriptors, axis=0)

if verbose:
    print(labels)

try:
    with open("../resources/kmeans1.txt", "rb") as input:
        kmeans1 = pickle.load(input)
except (EOFError, FileNotFoundError):
    exit(-1)

bows = np.empty(shape=(0, k1), dtype=int)

i = 0
for nb in dimImg:  # for each sound (file)
    tmpBow = [0] * k1
    j = 0
    while j < nb:  # for each MFCC of this sound (file)
        tmpBow[kmeans1.labels_[i]] += 1
        j += 1
        i += 1
    copyBow = tmpBow.copy()
    bows = np.append(bows, [copyBow], 0)

try:
    with open("../resources/sauvegarde.logr", "rb") as input:
        logisticRegr: LogisticRegression = pickle.load(input)
except EOFError:
    exit(-1)

labelsPredicted = logisticRegr.predict(bows)

score = logisticRegr.score(bows, labels)
print("train score = ", score)
print(labels)
print(labelsPredicted)
