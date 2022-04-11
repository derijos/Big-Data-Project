import cv2
from skimage import feature
from imutils import paths
import progressbar
import time
from sklearn.svm import SVC
import pickle

healthy = list(paths.list_images('Corn Disease detection\Healthy corn'))
infected = list(paths.list_images('Corn Disease detection\infected'))

features = []
label = []

widgets = ["Extracting Feature Vectors From Positive Samples", " ", progressbar.Bar(), " ", progressbar.Percentage(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(healthy), widgets=widgets)

pbar.start()

for i, imagepath in enumerate(healthy):
    try:
        image = cv2.imread(imagepath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = cv2.resize(gray, (100, 100))
        H = feature.hog(edged, orientations=9, pixels_per_cell=(10, 10),
                        cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
        features.append(H)
        label.append("Healthy Crop")
        pbar.update(i)

    except:
        continue

pbar.finish()



widgets = ["Extracting Feature Vectors From Negative Samples", " ", progressbar.Bar(), " ", progressbar.Percentage(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(infected), widgets=widgets)

pbar.start()

for i, imagepath in enumerate(infected):
    try:
        image = cv2.imread(imagepath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = cv2.resize(gray, (100, 100))
        H = feature.hog(edged, orientations=9, pixels_per_cell=(10, 10),
                        cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
        features.append(H)
        label.append("Infected Crop")
        pbar.update(i)

    except:
        continue
pbar.finish()


pickle.dump(features, open('feature_vectors.pkl', "wb"))
pickle.dump(features, open('label.pkl', "wb"))

#Training Model
svm = SVC()
svm.fit(features, label)
pickle.dump(svm, open("model.pkl", "wb"))