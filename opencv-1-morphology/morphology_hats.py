import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default="bts.jpg", help="path to image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

if (len(image.shape) == 3):
    # convert colored image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
else:
    # input image is already in gray scale
    print("ALREADY GRAY SCALE")
    gray = image

cv2.imshow("original", image)
cv2.imshow("gray scale", gray)
cv2.waitKey(0)

# the black hat operation is good for detecting dark regions on a bright background

rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (39, 15))
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
cv2.imshow("blackhat", blackhat)
cv2.waitKey(0)

# The top hat (white hat) operation is good for detecting bright regions on a dark background
# it is the diffference between the original input image and the "Opening" (erosion then dilation) of that image
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (200, 200))
whitehat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
cv2.imshow("white hat", whitehat)
cv2.waitKey(0)