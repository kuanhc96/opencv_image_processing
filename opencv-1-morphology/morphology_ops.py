import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="taiwan_original.jpg", help="path to image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
if (image.shape[2] == 3):
    # convert colored image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
else:
    # input image is already in gray scale
    print("ALREADY GRAY SCALE")
    gray = image

cv2.imshow("original", image)
cv2.imshow("gray scale", gray)

# apply erosion:
# Erosion: A foreground pixel in the input image will be kept only if ALL pixels inside the structuring element (kernel) are > 0
# That means, the larger the kernel, the more pixels are likely to be eroded away on a given iteration
gray_copy = gray.copy()
kernel = np.ones((3, 3), dtype=np.uint8)
for i in range(1, 7):
    # erosion is not applied inplace to the input
    eroded = cv2.erode(gray_copy, kernel, iterations=i)
    cv2.imshow(f"Eroded {i} times", eroded)
    cv2.waitKey(0)

cv2.destroyAllWindows()
cv2.imshow("gray scale", gray_copy)
# apply dilation:
# Dilation: The opposite of erosion. a center pixel of the kernel is set to white if ANY pixel in the kernel is > 0
for i in range(1, 7):
    dilated = cv2.dilate(gray_copy, None, iterations=i)
    cv2.imshow(f"Dilated {i} times", dilated)
    cv2.waitKey(0)

cv2.destroyAllWindows()
# The ideal type of kernel depends on the shape of the objects that need to be preserved
# rectangular kernels will preserve rectangular shapes more easily
# cross kernels will preserve horizontal and vertical protrusions
# elipse kernels will preserve circular objects
# Opening: To open up "breaches" while preserving dominant features. This is an erosion followed by a dilation
kernel_types = [cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE] 
kernel_sizes = [(3, 3), (5, 5), (7, 7), (9, 9)] # The larger the kernel, the greater erasing effect
for kernel_type in kernel_types:
    for kernel_size in kernel_sizes:
        kernel = cv2.getStructuringElement(kernel_type, kernel_size)
        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        cv2.imshow(f"Opening: kernel type: {kernel_type}, size: {kernel_size}", opening)
        cv2.waitKey(0)

# Closing: To close up "breaches" while preserving dominant features. This is a dilation followed by an erosion
# rectangular kernels will grow rectangular shapes more easily
# cross kernels will grow pixels that  round out edges, forming circular shapes
# ellipse kernels will grow pixels that are circular to each other?
for kernel_type in kernel_types:
    for kernel_size in kernel_sizes:
        kernel = cv2.getStructuringElement(kernel_type, kernel_size)
        closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        cv2.imshow(f"closing: kernel type: {kernel_type}, size: {kernel_size}", closing)
        cv2.waitKey(0)

# gradient: this is the difference between the dilation and erosion. 
# This takes what would be "grown" (dilated) and subtracts what would be "removed" (eroded)
# This is good for boundary detection
# The larger the kernel, the wider the boundaries will become, which makes sense because
# The larger the kernel, the more dilation will happen; the larger the kernel, the more erosion will happen
# The difference between which will become greater with kernel size
for kernel_type in kernel_types:
    for kernel_size in kernel_sizes:
        kernel = cv2.getStructuringElement(kernel_type, kernel_size)
        closing = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        cv2.imshow(f"gradient: kernel type: {kernel_type}, size: {kernel_size}", closing)
        cv2.waitKey(0)

cv2.destroyAllWindows()
