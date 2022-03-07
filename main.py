import cv2, time, math
from scipy.spatial.distance import cdist
import numpy as np
from functools import partial

WINDOW_NAME = "VexVision by Ansel"
PREVIEW_NAME = "Preview"

RED = [0,0,255]
GREEN = [0,255,0]
BLUE = [255,0,0]

BLOB_COLOR = GREEN


def nothing(x):
    pass

def  applyCanny(frame):

    cv2.Canny(frame, 100, 200)

def applyBlur(frame):
    size = 1 + 2*cv2.getTrackbarPos("blur", WINDOW_NAME) # Kernel size must be odd
    return cv2.GaussianBlur(frame, [size, size] ,0) # apply a blur to smooth out noise.

# Set image to be distance from color
def applyColorFilter(frame, targetColor):

    SCALAR = 255.0 / math.sqrt(255**2+255**2+255**2)

    tr, tg, tb = targetColor

    # [num pixels x 3]
    h,w,c = frame.shape
    flattened = frame.reshape(h*w, c)

    result = 255 - cdist([[tb, tg, tr]], flattened) * SCALAR # now scaled between 0 - 255 in terms of distance to color. 255 = color

    threshold = 140 + cv2.getTrackbarPos("threshold", WINDOW_NAME)
    result[result <= threshold] = 0 # Set pixels far enough from desired color to 0

    result = result.astype(np.uint8) # convert float to 0-255
    result = np.tile(np.array([result]).transpose(), (1, 3)) # convert greyscale into rgb
    result = result.reshape(h,w, 3) # reshape into height x width x rgb

    return result


def main():

    # Create filtered and preview windows
    cv2.namedWindow(PREVIEW_NAME)
    cv2.moveWindow(PREVIEW_NAME, 700, 0)
    cv2.namedWindow(WINDOW_NAME)
    
    # Create threshold and gaussian blur trackbars
    cv2.createTrackbar("threshold", WINDOW_NAME , 30, 80, nothing)
    cv2.createTrackbar("blur", WINDOW_NAME, 0, 20, nothing)

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
        keypoints = blobDetector.detect(filteredFrame) # detect blobs from greyscale

        # Outline detected blobs
        blank = np.zeros((1, 1))
        filteredFrame = cv2.drawKeypoints(filteredFrame, keypoints, blank, BLOB_COLOR, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        frame = cv2.drawKeypoints(frame, keypoints, blank, BLOB_COLOR, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

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
