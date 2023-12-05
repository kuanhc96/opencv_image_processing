# https://pyimagesearch.com/2021/04/28/opencv-smoothing-and-blurring/?_ga=2.76874773.1862556207.1701016519-1842902230.1698424416
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="alice_with_rifle2.jpg", help="path to image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Original", image)
# The larger the kernel, the more blurred the image will become, since the central pixel of the kernel is being replaced by averages that involve more pixels
kernel_sizes = [(3, 3), (9, 9), (15, 15)] 

for (kX, kY) in kernel_sizes:
    # (simple) average blurring:
    # take the average value of the pixels surrounding the central pixel in a kernel and replace the central pixel with that value
    # Each pixel in the kernel is weighted equally, which can easily over-blur an image
    blurred = cv2.blur(image, (kX, kY))
    cv2.imshow(f"Average ({kX}, {kY})", blurred)
    cv2.waitKey(0)
# cv2.destroyAllWindows()


cv2.imshow("Original", image)
for (kX, kY) in kernel_sizes:
    # gaussian blurring:
    # Documentation: https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#gaussianblur
    # The gaussian blur gives a more natural blur compared to the simple blurring technique, albeit at a slightly slower speed
    # take the average value of the pixels in a kernel and replace the central pixel using this average value
    # This average is computed using the gaussian equation, which gives more weight to the pixels that are closer to the central pixel
    # This will provide better blurring
    # G(x, y) = 1/(2*pi*sigma)*e^(-(x^2 + y^2)/2(sigma^2))
    # In the Gaussian, x, y are input variables, and sigma is a hyperparameter that can be fine tuned. It represents the standard deviation of the Gaussian distribution
    blurred = cv2.GaussianBlur(image, (kX, kY), 0) 
    # the third variable is the aforementioned sigma value in the gaussian equation. 0 means OpenCV will automatically calculate one
    # This value can be set to a specific value if the same results need to be reproduced
    cv2.imshow(f"Gaussian ({kX}, {kY})", blurred)
    cv2.waitKey(0)

cv2.destroyAllWindows()

cv2.imshow("Original", image)
for k in (3, 9, 15):
    # Median blurring:
    # Median blurring is most suitable for blurring out "salt and pepper noise" -- pixels that are randomly 0 (black) or 255 (white)
    # Median blurring replaces the center pixel in the kernel with the median value of the pixels found in the kernel, thereby 
    # using a pixel that actually exists in the kernel to replace the center pixel
    # As such, extremeties can be effectively blurred out, since the median, by definition, is robust agains extremeties
    # Note that, the kernel used in Median blurring MUST by a square, which is why the input does not take a tupe of (kX, kY)
    # Additionally, the median blurring technique causes pixels of similar colors to be blurred together, since they likely share similar medians
    # Notice how lines, folds, ruffles, gradually are blurred out from the image
    blurred = cv2.medianBlur(image, k)
    cv2.imshow(f"Median {k}", blurred)
    cv2.waitKey(0)

cv2.destroyAllWindows()