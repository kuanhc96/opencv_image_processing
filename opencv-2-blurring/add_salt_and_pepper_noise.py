import random
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True, help="path to image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("original", image)

(h, w) = image.shape[:2]
num_pixels = h*w

num_salt = random.randint(0.1 * num_pixels, 0.4*num_pixels)
for i in range(num_salt):
    y = random.randint(0, h - 1)
    x = random.randint(0, w - 1)
    image[y][x] = 255

num_pepper = random.randint(0.1 * num_pixels, 0.4*num_pixels)
for i in range(num_pepper):
    y = random.randint(0, h - 1)
    x = random.randint(0, w - 1)
    image[y][x] = 0

cv2.imshow("salt and pepper", image)
cv2.waitKey(0)
cv2.imwrite(f"salt_and_pepper_{args['image']}", image)