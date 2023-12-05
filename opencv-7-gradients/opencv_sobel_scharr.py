# https://pyimagesearch.com/2021/05/12/image-gradients-with-opencv-sobel-and-scharr/?_ga=2.224447389.680671124.1701662763-1842902230.1698424416
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True, help="path to input image")
ap.add_argument("-s", "--scharr", type=int, default=0, help="use Scharr method or not. Default is Sobel method")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)

kernel_size = -1 if args["scharr"] > 0 else 3 # if using the scharr method, kernel size = -1; if using the Sobel method, the kernel size is 3
gX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=kernel_size) # dx = 1: calculate gradient in X direction
gY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=kernel_size) # dy = 1: calculate gradient in Y direction

# convert the 32-bit unsigned matrices to 8-bit unsigned matrices for visualization
gX = cv2.convertScaleAbs(gX)
gY = cv2.convertScaleAbs(gY)

# The addWeighted function combines two images to form a third based on weighted coefficients assigned to the two images
# combined = alpha * image1 + beta * image2 + gamma
# alpha and beta are weighted coefficients, and gamma is a scalar added to the combined image to adjust brightness
combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)

cv2.imshow("Sobel/Scharr X", gX)
cv2.imshow("Sobel/Scharr Y", gY)
cv2.imshow("Sobel/Scharr Combined", combined)
cv2.waitKey(0)