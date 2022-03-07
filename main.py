import cv2, time, math
from scipy.spatial.distance import cdist
import numpy as np
from functools import partial

WINDOW_NAME = "VexVision by Ansel"
PREVIEW_NAME = "Preview"
TRACKBARS = "Customize"

BLACK = np.array([0,0,0], dtype = np.uint8)
RED = np.array([0,0,255], dtype = np.uint8)
GREEN = np.array([0,255,0], dtype = np.uint8)
BLUE = np.array([255,0,0], dtype = np.uint8)
YELLOW = np.array([0,255,255], dtype = np.uint8)
PURPLE = np.array([255,0,255], dtype = np.uint8)
AQUA = np.array([255,255,0], dtype = np.uint8)
COLORS = [RED, GREEN, BLUE, YELLOW, PURPLE, AQUA]

BLOB_COLOR = GREEN


def nothing(x):
    pass

def  applyCanny(frame):

    cv2.Canny(frame, 100, 200)

def applyBlur(frame):
    size = 1 + 2*cv2.getTrackbarPos("blur", TRACKBARS) # Kernel size must be odd
    return cv2.GaussianBlur(frame, [size, size] ,0) # apply a blur to smooth out noise.


# labels is a 2d array with each pixel specifying the label of the connected component it belongs to
def drawCCRects(frame, numLabels, labels, stats, centroids):

    pass
    

# Set image to be distance from color
def applyColorFilter(frame, targetColor):

    SCALAR = 255.0 / math.sqrt(255**2+255**2+255**2)

    tr, tg, tb = targetColor

    # [num pixels x 3]
    h,w,c = frame.shape
    flattened = frame.reshape(h*w, c)

    result = 255 - cdist([[tb, tg, tr]], flattened) * SCALAR # now scaled between 0 - 255 in terms of distance to color. 255 = color
    result = result.astype(np.uint8) # convert float to 0-255

    threshold = 140 + cv2.getTrackbarPos("threshold", TRACKBARS)
    
    # Get connected components
    binary = cv2.threshold(result, threshold, 255, cv2.THRESH_BINARY)[1] # binary is a binary array
    binary = binary.reshape(h,w) # reshape as 2d array in preparation for convolution

    # "Pad" elements close-by to ones through convolution in order to connect closely-related components
    padding = 1 + 2*cv2.getTrackbarPos("connectivity", TRACKBARS)
    binary = cv2.filter2D(src=binary, ddepth=-1, kernel=np.ones([padding, padding], dtype = np.uint8))
    binary[binary > 1] = 1 # restrict back to 1
    output = cv2.connectedComponentsWithStats(binary, 8, cv2.CV_32S)

    # Convert label to color
    def toColor(x, i):
        return np.array(0, dtype = np.uint8) if (x == 0) else COLORS[x % len(COLORS)][i]

    # Convert label for each pixel into corresponding color. Very expensive operation
    labels = output[1]
    b = (labels*100%256).reshape(h,w)
    g = (labels*200%256).reshape(h,w)
    r = (labels*300%256).reshape(h,w)

    print(labels.shape, labels.dtype)

    coloredResult = np.dstack([b,g,r]).astype(np.uint8, copy = False)
    return coloredResult


def main():

    # Create filtered and preview windows
    cv2.namedWindow(PREVIEW_NAME)
    cv2.moveWindow(PREVIEW_NAME, 700, 0)
    cv2.namedWindow(WINDOW_NAME)
    cv2.namedWindow(TRACKBARS)
    cv2.moveWindow(TRACKBARS, 0, 400)
    
    # Create threshold and gaussian blur trackbars
    cv2.imshow(TRACKBARS, np.zeros([1, 500]))
    cv2.createTrackbar("threshold", TRACKBARS , 30, 80, nothing)
    cv2.createTrackbar("blur", TRACKBARS, 10, 20, nothing)
    cv2.createTrackbar("connectivity", TRACKBARS, 15, 30, nothing)

    # init blob detection
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 5
    params.blobColor = 255
    blobDetector = cv2.SimpleBlobDetector_create(params)

    
    vc = cv2.VideoCapture(0)
    print("Capture started")


    while True:
        rval, frame = vc.read()
        frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5) # reduce image by half
        frame = applyBlur(frame) # gaussian blur
        if not rval:
            break

        filteredFrame = applyColorFilter(frame, [255, 0, 0]) # Greyscale image with distance to desired color

        # Display filtered and preview windows
        cv2.imshow(WINDOW_NAME, filteredFrame)
        cv2.imshow(PREVIEW_NAME, frame)
        key = cv2.waitKey(10) # wait 10 ms while at the same time listen for keyboard input. necessary for opencv loop to work
        if key == 27: # exit on ESC
            break

    vc.release()
    cv2.destroyWindow(WINDOW_NAME)

if __name__ == "__main__":
    main()
