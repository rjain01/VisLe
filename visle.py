import tools
import cv2 as cv

imgList = tools.allimg(
    r"D:\VisLe\Photos")

for img in imgList:

    # cv.imshow("Original", img)

    img = cv.resize(img, (500, 500))

    hsv = tools.convertHSV(img)
    mask_rot = tools.createMaskRotten(hsv)
    area_rotten, cont_rotten = tools.findRottenArea(img, mask_rot)
    mask_bg = tools.createMaskBg(hsv)
    area_bg = tools.findBgArea(img, mask_bg)
    tools.calculations(img, area_rotten, area_bg, cont_rotten)

    cv.imshow("Result", img)

    cv.waitKey(0)
    cv.destroyAllWindows()
