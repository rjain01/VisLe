import cv2 as cv
from matplotlib import image
import numpy as np
import os


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


def allimg(folder):
    image = load_images_from_folder(folder)
    imlist = []
    i = 0
    for img in image:
        name = "img" + str(i)
        imlist.append(img)

    return imlist


def convertHSV(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    return hsv


def createMaskRotten(hsv):
    lower_rot_light = np.array([6, 100, 145])
    upper_rot_light = np.array([23, 190, 210])
    mask_rot_light = cv.inRange(hsv, lower_rot_light, upper_rot_light)

    lower_rot_dark = np.array([6, 120, 16])
    upper_rot_dark = np.array([23, 190, 120])
    mask_rot_dark = cv.inRange(hsv, lower_rot_dark, upper_rot_dark)

    mask_rot = mask_rot_light + mask_rot_dark

    return mask_rot


def findRottenArea(img, mask_rot):

    masked_rotten = cv.bitwise_and(img, img, mask=mask_rot)

    masked_rotten = cv.cvtColor(masked_rotten, cv.COLOR_BGR2GRAY)
    masked_rotten = cv.GaussianBlur(masked_rotten, (11, 11), 0)
    cont_rotten, her_rotten = cv.findContours(
        masked_rotten, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    cont_rotten = sorted(cont_rotten, key=cv.contourArea, reverse=True)
    area_rotten = 0
    for cont in cont_rotten:
        area_rotten += cv.contourArea(cont)

    return area_rotten, cont_rotten


def createMaskBg(hsv):

    lower_bg = np.array([0, 0, 204])
    upper_bg = np.array([179, 65, 255])

    mask_bg = cv.inRange(hsv, lower_bg, upper_bg)

    return mask_bg


def findBgArea(img, mask_bg):

    masked_bg = cv.bitwise_and(img, img, mask=mask_bg)

    masked_bg = cv.cvtColor(masked_bg, cv.COLOR_BGR2GRAY)
    area_bg = cv.countNonZero(masked_bg)

    return area_bg


def calculations(img, area_rotten, area_bg, cont_rotten):
    height = int(img.shape[0])
    width = int(img.shape[1])
    total_area = height*width
    area_fruit = total_area - area_bg

    percentage_rotten = round((area_rotten/area_fruit)*100, 2)
    # print(f"Percentage of rotten fruit is {percentage_rotten} %")

    image = img
    org = (50, 100)
    font = cv.FONT_HERSHEY_SCRIPT_SIMPLEX
    fontScale = 1

    if (percentage_rotten > 5):
        # print("The fruit is rotten!")
        img = cv.putText(image, 'Rotten', org, font, fontScale,
                         color=(0, 0, 255), thickness=2)
        cv.drawContours(img, cont_rotten, -1, (255, 0, 0), 2)

    else:
        # print("The fruit looks good!")
        img = cv.putText(image, 'Fresh', org, font, fontScale,
                         color=(0, 255, 0), thickness=2)
