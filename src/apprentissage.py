import glob
import cv2
from sys import argv
from sklearn.cluster import KMeans
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

verbose = False

k1 = int(argv[1])

lesSIFT = np.empty(shape=(0, 128), dtype=float)

labels = []
dimImg = []  # nb of mfcc per file
listImg = glob.glob("../resources/base_samples/*.jpg")

for img in listImg:
    labels.append(0) if img[26] == 'b' else labels.append(1)
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

# try:
#    with open("../resources/kmeans1.txt", "rb") as input:
#        kmeans1 = pickle.load(input)
# except (EOFError, FileNotFoundError):
kmeans1 = KMeans(n_clusters=k1, random_state=0).fit(lesSIFT)
with open("../resources/kmeans1.txt", "wb") as output:
    pickle.dump(kmeans1, output, pickle.HIGHEST_PROTOCOL)

bows = np.empty(shape=(0, k1), dtype=int)

i = 0
for nb in dimImg:  # for each img
    tmpBow = [0] * k1
    j = 0
    while j < nb:  # for each SIFT of this img
        tmpBow[kmeans1.labels_[i]] += 1
        j += 1
        i += 1
    copyBow = tmpBow.copy()
    bows = np.append(bows, [copyBow], 0)

# print(bows)

logisticRegr = LogisticRegression()

logisticRegr.fit(bows, labels)

labelsPredicted = logisticRegr.predict(bows)

score = logisticRegr.score(bows, labels)

# sauvegarde de l'objet
with open('../resources/sauvegarde.logr', 'wb') as output:
    pickle.dump(logisticRegr, output, pickle.HIGHEST_PROTOCOL)

print("train score = ", score)

print(confusion_matrix(labels, labelsPredicted))
