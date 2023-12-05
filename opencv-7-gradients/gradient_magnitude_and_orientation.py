# https://pyimagesearch.com/2021/05/12/image-gradients-with-opencv-sobel-and-scharr/?_ga=2.224447389.680671124.1701662763-1842902230.1698424416
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True, help="path to image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)

gX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ) # dx = 1: calculate gradient in X direction
gY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ) # dy = 1: calculate gradient in Y direction

# compute gradient magnitude and orientation
magnitude = np.sqrt((gX ** 2) + (gY ** 2))
orientation = np.arctan2(gY, gX) * (180/np.pi) % 180

(fig, axs) = plt.subplots(nrows=1, ncols=3, figsize=(8, 4))

axs[0].imshow(gray, cmap="gray")
axs[1].imshow(magnitude, cmap="jet")
axs[2].imshow(orientation, cmap="jet")
axs[0].set_title("grayscale")
axs[1].set_title("gradient magnitude")
axs[2].set_title("gradient orientation [0, 180]")

for i in range(0, 3):
    axs[i].get_xaxis().set_ticks([])
    axs[i].get_yaxis().set_ticks([])

plt.tight_layout()
plt.show()