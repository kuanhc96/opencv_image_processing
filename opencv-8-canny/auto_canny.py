# https://pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/?_ga=2.24063933.680671124.1701662763-1842902230.1698424416
import numpy as np
import argparse
import glob
import cv2

# inputs:
# image: a grayscale image
# sigma: a deviation value used to calculate default hyperparameters for hysteresis. Ranges for 0 - 1.0
# Low sigma results in a tighter boundary for hysteresis; high sigma results in a wider boundary for hysteresis
def auto_canny(image, sigma=0.33):
    # compute the median pixel intensity and use that as a basis for calculating the hyperparameters for hysteresis
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1 - sigma) * v)) # use max with 0 as one of the inputs in case (1 - sigma) * v < 0
    upper = int(min(255, (1 + sigma) * v)) # use max with 0 as one of the inputs in case (1 - sigma) * v < 0
    edged = cv2.Canny(image, lower, upper)
    print(v)
    print(lower)
    print(upper)

    return edged


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="path to input dataset of images")
args = vars(ap.parse_args())


for image_path in glob.glob(args["images"] + "/*.jpg"):
    # load images, convert to grayscale, and blur
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    wide = cv2.Canny(blurred, 10, 200)
    tight = cv2.Canny(blurred, 225, 250)
    auto = auto_canny(blurred, 0.1)
    
    cv2.imshow("Original", image)
    cv2.imshow("edges", np.hstack([wide, tight, auto]))
    cv2.waitKey(0)