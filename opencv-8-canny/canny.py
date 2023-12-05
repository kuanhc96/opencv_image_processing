# https://pyimagesearch.com/2021/05/12/opencv-edge-detection-cv2-canny/?_ga=2.14732209.680671124.1701662763-1842902230.1698424416
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True, help="path to input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

cv2.imshow("Original", image)
cv2.imshow("blurred", blurred)


# after blurring, Canny calculates gradient magnitude and orientation for each pixel in the input image. It the applies "non-maxima supression" to remove noise
# lastly, hysteresis thresholding is applied to filter out "weak edges"
# Hystereis thresholding:
# Two hyperparameters are used to search for "strong edges" in a graph; edges that do not meet the criteria of a "strong edge" are filtered out
# T_upper: the gradient that defines a "strong edge"
# T_lower: The gradient that defines a "weak edge", which is filtered out by Canny
# If an edge's intensity > T_upper, than it is a strong edge and preserved
# If an edge's intensity < T_lower, it is a weak edge and is filtered out
# If an edge's intensity < T_upper but > T_lower, then:
#   If this edge is connected to a strong edge, it is preserved;
#   otherwise, the edge will be filtered out
# first input: the image that Canny will be performed on
# second and third input: the bounds for Hysteresis. Tuning these hyperparameters is not a trivial task
wide = cv2.Canny(blurred, 10, 200)
mid = cv2.Canny(blurred, 30, 150)
tight = cv2.Canny(blurred, 220, 240)
cv2.imshow("wide hysteresis boundaries", wide)
cv2.imshow("mid hysteresis boundaries", mid)
cv2.imshow("tight hysteresis boundaries", tight)
cv2.waitKey(0)