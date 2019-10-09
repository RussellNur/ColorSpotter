import cv2
import numpy as np
#from PIL import Image

def getRGB(self):
    # opencv tutorial is from here: https://www.learnopencv.com/how-to-select-a-bounding-box-roi-in-opencv-cpp-python/
    # Read image
    im = cv2.imread("image_1.jpg")
    # Read image using PIL
    #imPIL = Image.open("image_1.jpg", "r")

    # Select ROI
    r = cv2.selectROI(im)

    # Crop image
    imCrop = im[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

    height, width, channels = imCrop.shape

    imageRGB = []

    for x in range(0, height, 1):
        for y in range(0, width, 1):
            color = imCrop[x, y]
            currentColor = []
            # print(color[0], color[1], color[2])
            currentColor.append(color[2])
            currentColor.append(color[1])
            currentColor.append(color[0])
            imageRGB.append(currentColor)

    print(imageRGB)
    return imageRGB

    # pix_val = list(imCrop.getdata())

    # print(pix_val)

    # Display cropped image
    #cv2.imshow("Image", imCrop)
    #cv2.waitKey(0)
