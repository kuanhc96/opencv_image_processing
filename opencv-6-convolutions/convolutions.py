from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2

def convolve(image, kernel):
    (image_height, image_width) = image.shape[:2]
    (kernel_height, kernel_width) = kernel.shape[:2]

    # calculate the number of pixels that should be padded along the x and y axes of the image
    # padding is applied so that the spatial dimensions of the output image after convolution does not shrink
    num_pad_width = ( kernel_width - 1  ) // 2
    num_pad_height = ( kernel_height - 1  ) // 2

    # cv2.copyMakeBorder:
    # this method is used to add padding to an image
    # There are various ways to provide padding to an image, documented on p.123 of Szeliski's book
    # For instance:
    # cv2.BORDER_REPLICATE: replicate the border pixels and use them to add padding
    # cv2.BORDER_WRAP: the padded value is determined by the pixels on the opposite end of a given border
    # The 1st input is the image that needs to be padded
    # The 2, 3, 4, and 5th input variable to this method all denote the number of pixels that should be padded along the top, bottom, left, and right borders, respectively
    image = cv2.copyMakeBorder(image, num_pad_height, num_pad_height, num_pad_width, num_pad_width, cv2.BORDER_REPLICATE)
    output = np.zeros((image_height, image_width), dtype="float32")

    # iterate over the padded image, performing convolutions as the kernel slides from top to bottom, left to right
    # draw out a kernel and an image matrix and these for loop values will make sense
    # The x, y values represent the central pixel of the kernel
    for y in np.arange(num_pad_height, image_height + num_pad_height): 
        for x in np.arange(num_pad_width, image_width + num_pad_width):
            convolved_region = image[y - num_pad_height:y + num_pad_height + 1, x - num_pad_width: x + num_pad_height + 1]
            # perform element wise multiplication and sum the results to get the output of the convolution at (x, y)
            convolved_value = (convolved_region * kernel).sum()

            # store output of convolved value into the output matrix:
            # Remember that the output image is NOT padded, so the coordinates will be off by on compared to the padded image
            output[y - num_pad_height, x - num_pad_width] = convolved_value

    # Final processing steps after convolution:
    # rescale the image so that the intensities fall in between the legal borders of an image
    # rescale_intensity will convert the values into percentages of the range given.
    # For instance, if one of the values in the array is 41 and the in_range=(0, 255), then the value will be converted to 41/256
    output = rescale_intensity(output, in_range=(0, 255)) 
    output = (output * 255).astype("uint8")
    return output

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to image")
args = vars(ap.parse_args())

smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
bigBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))

# The sharpen kernel
sharpen = np.array(
    (
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ), dtype="int"
)

# The laplacian kernel
# Used for edge detection
laplacian = np.array(
    (
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ), dtype="int"
)

sobelX = np.array(
    (
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ), dtype="int"
)

sobelY = np.array(
    (
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ), dtype="int"
)

kernelBank = (
    ("small_blur", smallBlur),
    ("big_blur", bigBlur),
    ("sharpen", sharpen),
    ("laplacian", laplacian),
    ("sobelX", sobelX),
    ("sobelY", sobelY),
)

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
for (kernel_name, kernel) in kernelBank:
    print(f"[INFO] applying {kernel_name} kernel")
    convolved_image = convolve(gray, kernel)
    cv2.imshow(f"{kernel_name}", convolved_image)
    cv2.waitKey(0)