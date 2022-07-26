import cv2 as cv
import numpy as np


img = cv.imread(
    r'D:\VisLe\Photos\apple3_withoutBG.jpg')

image = img.copy()
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = cv.GaussianBlur(img, (11, 11), 0)

cont, her = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

i = 0
for cont in cont:
    area = cv.contourArea(cont)
    if area > 1000:
        x, y, h, w = cv.boundingRect(cont)
        cropped_img = image[y:y+w, x:x+h]
        name = "apple_obj" + str(i) + ".jpg"
        cv.imwrite(name, cropped_img)
        i += 1

cv.waitKey(0)
cv.destroyAllWindows
