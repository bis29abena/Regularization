# import the necessary packages
import cv2 as cv


class SimplePreprocessor:
    def __init__(self, width, height, inter=cv.INTER_AREA):
        # store the target width, height and interpolation method
        # used when resizing
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        # resize the image to a fixed size, ignoring the aspect ratio
        return cv.resize(image, (self.width, self.height), interpolation=self.inter)