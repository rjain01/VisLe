import cv2 as cv
import numpy as np


img = cv.imread(
    r'Photos\apple3.jpg')

cv.imshow("Original", img)

# Convert image to HSV format
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

lower = np.array([15, 0, 0])
upper = np.array([116, 255, 255])

# Creating a Mask

mask = cv.inRange(hsv, lower, upper)
mask = cv.bitwise_not(mask)
final = cv.bitwise_and(img, img, mask=mask)

cv.imshow("Final", final)
cv.imwrite("apple3_withoutBG.jpg", final)

cv.waitKey(0)
cv.destroyAllWindows
