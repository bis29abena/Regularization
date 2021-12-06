# Usage
# python main.py --dataset "dataset"

# import the necessary packages
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from bis29_.preprocessing.preprocessor import SimplePreprocessor
from bis29_.datasets.datasetloader import SimpleDatasetLoader
from imutils import paths
import argparse

# construct an argument parser to parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input datasets")
args = vars(ap.parse_args())

# grab the list of image paths
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the image processor, load the dataset from disk and reshape the data matrix
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths=imagePaths, verbose=500)
print(f"[INFO] when data wasn't reshaped {data[0]}\n")
data = data.reshape((data.shape[0], 3072))
print(f"[INFO] when data was reshaped {data[0]}")

# encode the labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# partition the data to training and testing split 75% and 25% respectively
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# loop over our set of regularizers
for i in (None, "l1", "l2"):
    # train and SGD classifier using a softmax loss function and the
    # specified regularization function for 10 epochs
    print(f"[INFO] training model with {i} penalty")
    model = SGDClassifier(loss="log", penalty=i, max_iter=10, learning_rate="constant", tol=1e-3, eta0=0.01
                          , random_state=12)
    model.fit(trainX, trainY)

    # evaluate the classifier
    acc = model.score(testX, testY)
    print(f"[INFO] {i} penalty accuracy: {round(acc * 100, 2)}")