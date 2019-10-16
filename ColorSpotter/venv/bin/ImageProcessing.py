import cv2
# Method to select the area of the picture
# Pass image name as a string (file_path)
def getRGB(imageName):
    """

    :type imageName: str
    """
    # opencv tutorial is from here: https://www.learnopencv.com/how-to-select-a-bounding-box-roi-in-opencv-cpp-python/
    # Read image
    im = cv2.imread(imageName)

    # Select ROI
    r = cv2.selectROI("ColorSpotter - select a rectangle", im)

    # Crop image
    imCrop = im[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

    # Height and width of the selected area
    height, width, channels = imCrop.shape

    # Array of arrays of RGB values of every single pixel in the selected area (to be returned)
    imageRGB = []

    # Iterate through each pixel
    for x in range(0, height, 1):
        for y in range(0, width, 1):
            color = imCrop[x, y]
            # currentColor array to store value as RGB (not BGR as it is in the color variable above)
            currentColor = []
            # Append RGB values in the corect order: R-G-B into the array currentColor (pixel color)
            currentColor.append(color[2])
            currentColor.append(color[1])
            currentColor.append(color[0])
            # Append pixel's RGB color (array) into the main array of RGB values of the selected area (into imageRGB array).
            imageRGB.append(currentColor)
    # Close all the cv2 windows
    cv2.destroyAllWindows()
    # Return the array of arrays of the RGB values of all pixels in the selected area
    return imageRGB