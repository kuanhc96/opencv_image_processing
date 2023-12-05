import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True, help="path to input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Image", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

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
