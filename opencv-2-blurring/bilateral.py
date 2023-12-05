import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="alice_with_rifle2.jpg", help="path to image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Original", image)
params = [(11, 21, 7), (11, 41, 21), (11, 61, 39), (11, 100, 39), (100, 61, 500)]

for (diameter, sigmaColor, sigmaSpace) in params:
    # Bilateral blurring:
    # Bilateral blurring is able to blur images while preserving edges
    # The algorithm takes into account the spactial neighbors of a central pixel in a kernel,
    # as well as the intensity (color) of neighboring pixels in a kernel
    # The idea is that, if two pixels have a stark difference in intensity, then they are likely different objects,
    # and should preserve the boundary dividing them
    # This method is considerably slower than the simple average, gaussian, and median blurring methods
    blurred = cv2.bilateralFilter(image, diameter, sigmaColor, sigmaSpace)
    # There are three input variable to this function -- diameter, sigmaColor, and sigmaSpace
    # Diameter: the larger the diameter, the more pixels will be included in blurring. Similar to kernel size.
    # sigmaColor: color standard deviation. The larger the sigmaColor, the more colors will be included. If it gets too large, then edges may be blurred out
    # sigmaSpace: spatial standard deviation. The larger the sigmaSpace, the farther out from the centraol pixel diameter will influence blurring

    cv2.imshow(f"blurred d={diameter}, sc={sigmaColor}, ss={sigmaSpace}", blurred)
    cv2.waitKey(0)