import cv2
from skimage import feature
from imutils import paths
import pickle

model = pickle.load(open("model.pkl", 'rb'))
imagePath = list(paths.list_images("testing_images"))


for i in imagePath:
    image = cv2.imread(i)
    img1 = cv2.resize(image, (480, 480))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.resize(gray, (100, 100))
    H = feature.hog(edged, orientations=9, pixels_per_cell=(10, 10),
                    cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")

    pred = model.predict([H])[0]

    if pred == "Healthy Crop":
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)

    cv2.putText(img1, pred, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imshow("Testing Image", img1)
    cv2.waitKey(0)

