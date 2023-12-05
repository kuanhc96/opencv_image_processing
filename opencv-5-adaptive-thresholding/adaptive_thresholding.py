# https://pyimagesearch.com/2021/05/12/adaptive-thresholding-with-opencv-cv2-adaptivethreshold/?_ga=2.110568453.1862556207.1701016519-1842902230.1698424416
import argparse
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True, help="path to input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Image", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#blurred = cv2.GaussianBlur(gray, (3, 3), 0)
blurred = gray

# Otsu's thresholding:
# otsu's thresholding assumes that there is a bimodal distribution of pixel intensities and tries to
# provide a thresholding scheme such that the variance between background and foreground pixels is minimal
# when using Otsu's thresholding, the thresholding value no longer matters -- it will be automaticallyc computed, so 0 is fine
# After Otsu is applied, since cv2.THRESH_BINARY_INV is used, the values less than the threshold will be set to 255
(T, threshInv) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
masked = cv2.bitwise_and(image, image, mask=threshInv)
cv2.imshow("mask", threshInv)
cv2.imshow(f"Threshold: {T}", masked)
cv2.waitKey(0)


# Adaptive thresholding:
# Adaptive thresholding,  as opposed to simple or otsu's thresholding, calculates local thresholds for different regions of an input image
# This is done by using a sliding window that will encompass a range of pixels. Assuming that each localized region will have a similar lighting scheme,
# adaptive thresholding calculates local thresholds using the formula: T = mean(I_local) - C
# The "mean" here can either be the arithmatic mean or the gaussian mean (mean of a Gaussian distribution)
# The constant, C, is used to avoid thresholding out too many pixels.
# Think about the constant C this way: if the mean roughly divides the pixel values into two equal halves, that would mean half of the pixels don't meet the threshold
# C is a hyperparameter used to provide fine tuning, depending on the situation. Luckily, in most cases, there will be a range of tolerable C's that can still achieve good results.
# the window size is also a hyperparameter used to provide fine tuning. The smaller the window, the more calculations will be done
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)
thresh = cv2.dilate(thresh, None, iterations=3)
cv2.imshow("Mean adaptive thresholding", thresh)
masked = cv2.bitwise_and(image, image, mask=thresh)
cv2.imshow("Masked with mean adaptive thresholding", masked)
kernel = np.ones((3, 3), dtype=np.uint8)
masked = cv2.erode(masked, kernel, iterations=2)
cv2.imshow("(eroded) Masked with mean adaptive thresholding", masked)
cv2.waitKey(0)

thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
thresh = cv2.dilate(thresh, None, iterations=3)
cv2.imshow("Gaussian adaptive thresholding", thresh)
masked = cv2.bitwise_and(image, image, mask=thresh)
cv2.imshow("(eroded) Masked with gaussian adaptive thresholding", masked)
masked = cv2.erode(masked, kernel, iterations=2)
cv2.waitKey(0)

cv2.copyMakeBorder()