import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="alice_with_rifle2.jpg")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("RGB", image)

for (name, channel) in zip(("B", "G", "R"), cv2.split(image)):
    cv2.imshow(name, channel)

cv2.waitKey(0)


# HSV color space:
# Hue: "pure" color. The colors exist on a spectrum from 0 degrees to 360 degrees, each representing a different color
#       In openCV, since a pixel can only have values between [0, 255], the Hue space ranges from [0, 179], representing 180 distinct colors
#       That means, colors are distinguished by each other every 2 degrees
# Saturation: How "Pure" this a given color is. ranges falls between [0, 255]. 0 represents white for any Hue; 255 represents a pure color
# Value: defines a "shade". Ranges between [0, 255]. 0 represents black (total shade), whereas 255 represents no shade
# The HSV color space is used in computer vision applications when some specific color is being tracked
# However, the HSV color space is not good for representing the way humans understand colors, specifically, the "difference" in color
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV", hsv)

for (name, channel) in zip(("H", "S", "V"), cv2.split(hsv)):
    cv2.imshow(name, channel)

cv2.waitKey(0)
cv2.destroyAllWindows()

# LAB color space:
# RGB and HSV values cannot capture the "differences" between color. the Euclidean distance in the RGB and HSV have no meaning,
# so one cannot say that red is "closer" to purple than green based on these spaces.
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
cv2.imshow("L*a*b", lab)

for (name, channel) in zip(( "L*", "a*", "b*" ), cv2.split(lab)):
    cv2.imshow(name, channel)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Grayscale
# The grayscale of an image is a weighted average of the RGB values in an image.
# Since the human eye perceives RGB colors to varying degrees, these values are not weighted equally
# The conversion used is grayscale = 0.299*R + 0.587*G, 0.114*B
# Grayscale images are used when color is unimportant, such as detecting faces, or using object classifiers
# having two fewer color channels to store in memory also improves speed
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original", image)
cv2.imshow("grayscale", gray)
cv2.waitKey(0)