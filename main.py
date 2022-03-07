import cv2, time, math
from scipy.spatial.distance import cdist
import numpy as np
from functools import partial

WINDOW_NAME = "VexVision by Ansel"
PREVIEW_NAME = "Preview"
TRACKBARS = "Customize"

BLACK = [0,0,0]
RED = [0,0,255]
GREEN = [0,255,0]
BLUE = [255,0,0]
YELLOW = [0,255,255]
PURPLE = [255,0,255]
AQUA = [255,255,0]
COLORS = [RED, YELLOW, PURPLE, AQUA]

BLOB_COLOR = GREEN
FONT = cv2.FONT_HERSHEY_SIMPLEX

def distance(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

# A label is a single connected component for a single frame
class Label:
    def __init__(self, x, y, area):
        self.x = x
        self.y = y
        self.area = area
        self.owner = None # object that owns label

# An object for a frame that is defined usually by one, but sometimes more, labels
class State:
    def __init__(self, obj):
        self.labels = [] # at least one label that belongs to the object and defines its location
        self.object = obj

    def assign(self, label):
        self.labels.append(label)
        label.owner = self.object

    # use a weighted average of coordinates of labels to find object's coordinates
    def getPosition(self):
        sumX = 0
        sumY = 0
        sumArea = 0
        for label in self.labels:
            sumArea += label.area
            sumX += label.x * label.area
            sumY += label.y * label.area
        return [round(sumX / sumArea), round(sumY / sumArea)]

    def getArea(self):
        sumArea = 0
        for label in self.labels:
            sumArea += label.area
        return sumArea

# An persistent object consisting of a current and previous state
class Object:

    def __init__(self, ID, label):
        self.id = ID
        self.prevState = None
        self.currState = State(self)
        self.currState.assign(label)

    def resetTick(self):
        self.prevState = self.currState
        self.currState = State(self)

# Storing a list of objects currently on the screen. Track based on labels in each frame. Objects may contains multiple labels close together if there was
# originally one object there
class Objects:

    def __init__(self):
        self.objects = [] # a list of persistent objects that get modified at each frame
        self.labels = [] # the connected components for this specific frame

    def existsID(self, ID):
        for obj in self.objects:
            if obj.id == ID:
                return True
        return False

    def createObject(self, label):
        # Find next available ID
        freeID = 0
        while self.existsID(freeID):
            freeID += 1

        # Create object and append to self.objects
        self.objects.append(Object(freeID, label))
        

    # Call at beginning of each tick to reset labels
    def resetStartOfTick(self):
        self.labels = []
        for obj in self.objects:
            obj.resetTick()

    # numLabels is the number of labels (objects)
    # labels is a 2d array with each pixel specifying the label of the connected component it belongs to
    # stats[label, COLUMN] returns an array of information for that object
    # Call this function after resetStartOfTick() to store labels but before updateObjects(), which uses those labels, for each tick
    def drawDetectedBoxes(self, frame, numLabels, labels, stats, centroids):
        
        for i in range(1, numLabels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i,cv2.CC_STAT_TOP]
            x2 = x + stats[i, cv2.CC_STAT_WIDTH]
            y2 = y  + stats[i, cv2.CC_STAT_HEIGHT]
            
            cv2.rectangle(frame, [x, y], [x2, y2], GREEN, 2)
            center = np.around(centroids[i]).astype(int).tolist()
            cv2.circle(frame, center, 5, BLUE, -1)
            cv2.putText(frame, "({}, {})".format(*center), [center[0] - 40, center[1] - 13], FONT, 0.5, BLUE, 2, cv2.LINE_AA)

            self.labels.append(Label(*center, stats[i, cv2.CC_STAT_AREA]))

    # Given the current and previous set of labels, calculate the new positions of objects, and possibly add/delete current objects
    def updateObjects(self):

        MAX_DISTANCE_IDENTITY = 50

        # Sort objects by area from largest to smallest
        self.objects.sort(key = lambda obj : obj.prevState.getArea(), reverse = True)

        # Find a label that is close to each object
        for obj in self.objects:

            if obj.prevState is None:
                    continue

            closestLabel = None
            closestDistance = 10000
            
            for label in self.labels:
                if label.owner is not None: # only consider labels that have not been assigned to an object yet
                    continue
                dist = distance(label.x, label.y, *obj.prevState.getPosition())
                if dist < closestDistance:
                    closestDistance = dist
                    closestLabel = label

            print("Closest:", closestDistance)
            # If the closest label is sufficiently close, assign it to the object
            if closestDistance < MAX_DISTANCE_IDENTITY:
                obj.currState.assign(label)
                #print("object assigned")

        # For unassigned labels, assign to closest object or create new object if no object sufficiently close
        for label in self.labels:
            if label.owner is not None: # only consider labels that have not been assigned to an object yet
                continue

            closestDistance = 10000

            for obj in self.objects:
                if obj.prevState is None:
                    continue
                dist = distance(label.x, label.y, *obj.prevState.getPosition())
                if dist < closestDistance:
                    closestDistance = dist
                    closestLabel = label

            if closestDistance < MAX_DISTANCE_IDENTITY: # If the closest object is sufficiently close, have it be assigned to object
                obj.currState.assign(label)
                #print("additional label assigned to object")
            else: # otherwise, create new object for label
                self.createObject(label)
                #print("object created")

        # Destroy objects that did not get assigned any label
        i = 0
        while i < len(self.objects):
            if len(self.objects[i].currState.labels) == 0:
                #print("object deleted")
                del self.objects[i]
            else:
                i += 1
            

    def drawObjects(self, frame):
        print(len(self.objects))
        for obj in self.objects:
            pos = obj.currState.getPosition()
            print(pos, type(pos))
            radius = round(math.sqrt(obj.currState.getArea() / math.pi))
            color = COLORS[obj.id % len(COLORS)]
            cv2.circle(frame, pos, radius, color, 3)
            

def nothing(x):
    pass

def  applyCanny(frame):

    cv2.Canny(frame, 100, 200)

def applyBlur(frame):
    size = 1 + 2*cv2.getTrackbarPos("blur", TRACKBARS) # Kernel size must be odd
    return cv2.GaussianBlur(frame, [size, size] ,0) # apply a blur to smooth out noise.
    

# Set image to be distance from color
def applyColorFilter(frame, objects, targetColor):

    SCALAR = 255.0 / math.sqrt(255**2+255**2+255**2)

    tr, tg, tb = targetColor

    # reshape into a [num pixels x color] array for easier data manipulation
    h,w,c = frame.shape
    flattened = frame.reshape(h*w, c)

    result = 255 - cdist([[tb, tg, tr]], flattened) * SCALAR # now scaled between 0 - 255 in terms of distance to color. 255 = color
    result = result.astype(np.uint8) # convert float to 0-255

    threshold = 140 + cv2.getTrackbarPos("threshold", TRACKBARS) # slightly arbitrary limit to sliders between 140 and 220 for detection threshold
    
    # Binary step function to convert analog into [detected] [not detected] pixels
    binary = cv2.threshold(result, threshold, 255, cv2.THRESH_BINARY)[1] # binary is a binary array
    binary = binary.reshape(h,w) # reshape as 2d array in preparation for convolution

    # "Pad" elements close-by to ones through convolution in order to connect closely-related components
    padding = 1 + 2*cv2.getTrackbarPos("connectivity", TRACKBARS)
    binary = cv2.filter2D(src=binary, ddepth=-1, kernel=np.ones([padding, padding], dtype = np.uint8)) # one-matrix kernel with parametrized dimensions
    binary[binary > 1] = 1 # restrict back to 1
    output = cv2.connectedComponentsWithStats(binary, 8, cv2.CV_32S)

    # Convert label to color
    def toColor(x, i):
        return np.array(0, dtype = np.uint8) if (x == 0) else COLORS[x % len(COLORS)][i]

    # Convert label for each pixel into corresponding color. Using ufuncs here for extremely fast computation but sacrificing control over colors (who cares?)
    labels = output[1]
    b = (labels*100%256).reshape(h,w)
    g = (labels*200%256).reshape(h,w)
    r = (labels*300%256).reshape(h,w)

    # Merge the three rgb arrays back into a single [num pixels x 3] array
    coloredResult = np.dstack([b,g,r]).astype(np.uint8, copy = False)

    # Draw bounding boxes and centroids
    objects.drawDetectedBoxes(coloredResult, *output)
    
    return coloredResult


def main():

    # Create filtered and preview windows
    cv2.namedWindow(PREVIEW_NAME)
    cv2.moveWindow(PREVIEW_NAME, 700, 0)
    cv2.namedWindow(WINDOW_NAME)
    cv2.namedWindow(TRACKBARS)
    cv2.moveWindow(TRACKBARS, 0, 400)
    
    # Create threshold and gaussian blur trackbars
    cv2.imshow(TRACKBARS, np.zeros([1, 800]))
    cv2.createTrackbar("threshold", TRACKBARS , 35, 80, nothing)
    cv2.createTrackbar("blur", TRACKBARS, 10, 20, nothing)
    cv2.createTrackbar("connectivity", TRACKBARS, 30, 50, nothing)

    # init blob detection
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 5
    params.blobColor = 255
    blobDetector = cv2.SimpleBlobDetector_create(params)

    objects = Objects()
    
    vc = cv2.VideoCapture(0)
    print("Capture started")


    while True:
        objects.resetStartOfTick()
        
        rval, frame = vc.read()
        frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5) # reduce image by half
        frame = applyBlur(frame) # gaussian blur
        if not rval:
            break

        filteredFrame = applyColorFilter(frame, objects, [255, 0, 0]) # Greyscale image with distance to desired color
        objects.updateObjects()
        objects.drawObjects(filteredFrame)

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
