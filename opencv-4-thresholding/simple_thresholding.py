# https://pyimagesearch.com/2021/04/28/opencv-thresholding-cv2-threshold/?_ga=2.117304454.1862556207.1701016519-1842902230.1698424416
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,help="Path to input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Image", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray scale", gray)

kernel_sizes = [(3, 3), (5, 5), (7, 7), (9, 9),]
threshold_values = [100, 150, 200]
for kernel_size in kernel_sizes:
    blurred = cv2.GaussianBlur(gray.copy(), kernel_size, 0) # The last argument is the standard deviation; 0 menas the algorithm will automatically calculate it
    cv2.imshow(f"kernel size: {kernel_size}", blurred)
    # apply thresholding: Thresholding allows us to convert a grayscale image to a binary image based on hyperparameters that we set
    # arg 1: the image to be thresholded
    # arg 2: the threshold value. If a pixel's value is beyond or below this value, the method defined in arg 4 is applied
    # arg 3: The "otherwise" value
    # arg 4: THRESH_BINARY: if a pixel's value is above the threshold, set it to be 255; otherwise, set it to be arg 3
    #        THRESH_BINARY_INV: if a pixel's value is above the threshold, set it to be 0; otherwise, set it to be arg 3 (255, in the following case)
    # the results of THRESH_BINARY and THRESH_BINARY_INVERSE are inverse to each other, so taking the bitwise_not of each other will get the other output
    # The issue with simple thresholding is that the threshold value has to be determined manually through experimentation
    for threshold in threshold_values:
        (T, threshInv) = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)
        cv2.imshow(f"threshold value: {threshold} | blurring kernel size: {kernel_size} | Threshold binary inverse", threshInv)
        cv2.waitKey(0)
        (T, thresh) = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
        cv2.imshow(f"threshold value: {threshold} | blurring kernel size: {kernel_size} | Threshold binary", thresh)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
cv2.destroyAllWindows()


cv2.imshow("original", image)
blurred = cv2.GaussianBlur(gray.copy(), (7, 7), 0) # The last argument is the standard deviation; 0 menas the algorithm will automatically calculate it
(T, threshInv) = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("mask", threshInv)
threshInv = cv2.dilate(threshInv, None ,iterations=3)
cv2.imshow("dilated mask", threshInv)
masked = cv2.bitwise_and(image, image, mask=threshInv)
cv2.imshow("output", masked)
cv2.waitKey(0)