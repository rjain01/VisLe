import cv2 as cv

img = cv.imread('Photos/dog.jpg')
cv.imshow("Dog", img)


def resizeFrame(frame, scale=100):
    height = int(frame.shape[0] + scale)
    width = int(frame.shape[1] + scale)
    dimension = (width, height)

    return cv.resize(frame, dimension, interpolation=cv.INTER_AREA)


def half(frame):
    height = int(frame.shape[0])
    width = int(frame.shape[1])
    dimension = (int(width/2), int(height/2))

    return cv.resize(frame, dimension)


def double_(frame):
    height = int(frame.shape[0])
    width = int(frame.shape[1])
    dimension = (width*2, height*2)

    return cv.resize(frame, dimension)


resized = resizeFrame(img)
half_img = half(img)
double_img = double_(img)

cv.imshow("Resized", resized)
cv.imshow("Half", half_img)
cv.imshow("Double", double_img)
cv.waitKey(0)
