import cv2, time, math
from scipy.spatial.distance import cdist
import numpy as np
from functools import partial

# The color to detect for. In HSV
TARGET_COLOR = [0, 128, 128]


WINDOW_NAME = "VexVision by Ansel"
PREVIEW_NAME = "Preview"
TRACKBARS = "Customize"
VISUALIZER = "Color Visualizer"
VISUALIZER2 = "Brightness Visualizer"

BLACK = [0,0,0]
RED = [0,0,255]
GREEN = [0,255,0]
BLUE = [255,0,0]
YELLOW = [0,255,255]
PURPLE = [179, 0, 255]
PINK = [255,0,255]
AQUA = [255,255,0]
ORANGE = [255, 128, 0]
DGREEN = [48, 132, 73]
COLORS = [RED, YELLOW, PURPLE, AQUA, ORANGE, DGREEN, PINK]

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

        MAX_DISTANCE_IDENTITY = 20

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

            #print("Closest:", closestDistance)
            # If the closest label is sufficiently close, assign it to the object
            if closestLabel is not None and closestDistance < MAX_DISTANCE_IDENTITY:
                obj.currState.assign(closestLabel)
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
        #print(len(self.objects))
        s = "["
        for obj in self.objects:
            s += str(obj.id) + ", "
            pos = obj.currState.getPosition()
            #print(pos, type(pos))
            radius = round(math.sqrt(obj.currState.getArea() / math.pi))
            color = COLORS[obj.id % len(COLORS)]
            cv2.circle(frame, pos, radius, color, 3)
        #print(s, "]")
            

def nothing(x):
    pass

def  applyCanny(frame):

    cv2.Canny(frame, 100, 200)

def applyBlur(frame):
    size = 1 + 2*cv2.getTrackbarPos("blur", TRACKBARS) # Kernel size must be odd
    return cv2.GaussianBlur(frame, [size, size] ,0) # apply a blur to smooth out noise.
    

# Set image to be distance from color
# targetColor in hsv
# For HSV, hue range is [0,179], saturation range is [0,255], and value range is [0,255].
def applyColorFilter(hsvFrame, objects, targetColor):

    # height, width, color = 3
    h,w,c = hsvFrame.shape

    # Convert to hsv and flatten
    hsvFrame = hsvFrame.reshape(h*w, c)

    # colorDist is the modular distance to the targetColor in terms of hue
    colorDist = (180 + hsvFrame[:, 0] - targetColor[0]) % 180 # extract hue column
    colorDist = np.minimum(colorDist, 180 - colorDist) # modular distance
    colorDist = colorDist.reshape(h, w)

    #image = np.tile(np.array([colorDist]).transpose(), (1, 3))
    #print(image.shape)
    #cv2.imshow(VISUALIZER, colorDist.reshape(h,w))

    # targetArr is the euclidean distance to targetColor in terms of brightness
    targetArr = np.array([targetColor[1], targetColor[2]])
    darkArr = hsvFrame[:, 1:] # extract saturation and value columns
    darkDist = cdist([targetArr], darkArr)
    darkDist = darkDist.reshape(h,w)

    # Get user-defined thresholds
    cThresh = cv2.getTrackbarPos("color threshold", TRACKBARS)
    dThresh = cv2.getTrackbarPos("dark threshold", TRACKBARS)
    print(cThresh, dThresh)

    # Pixel is considered detected if it is close enough to targetColor both in terms of hue and brightness based on thresholds
    colorMask = (colorDist <= cThresh)
    darkMask = (darkDist <= dThresh)
    binary = (colorMask & darkMask).astype(np.uint8) # 1 if detected, 0 if not detected
    cv2.imshow(VISUALIZER, colorMask.astype(np.uint8)*255)
    cv2.imshow(VISUALIZER2, darkMask.astype(np.uint8)*255)

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

class Mouse:
    def __init__(self):
        self.x = -1
        self.y = -1
        self.active = False
        self.time = 0
        self.TIME_SHOWN = 1 # seconds
        self.radius = 5

    def press(self, x, y):
        self.x = x
        self.y = y
        self.active = True
        self.time = time.time()

    def isActive(self):
        if self.active and time.time() - self.time > self.TIME_SHOWN:
            self.active = False
        return self.active

    def draw(self, frame):
        if self.isActive():
            cv2.rectangle(frame, [self.x - self.radius, self.y - self.radius], [self.x + self.radius, self.y + self.radius], RED, 3)

mouse = Mouse()

def handleMouse(event, x, y, flags, param):

    global TARGET_COLOR

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse.press(x, y)

        # Find the average color of all the points around the clicked point, and set target color to this
        sumB = 0
        sumG = 0
        sumR = 0
        n = 0
        for i in range(max(0, x - mouse.radius), min(x + mouse.radius, cameraFrame.shape[1] - 1)):
            for j in range(max(0, y - mouse.radius), min(y + mouse.radius, cameraFrame.shape[0] - 1)):
                n  += 1
                sumB += cameraFrame[j][i][0]
                sumG += cameraFrame[j][i][1]
                sumR += cameraFrame[j][i][2]

        arr = np.array([[[round(sumB / n), round(sumG / n), round(sumR / n)]]], dtype = np.uint8)
        print(arr.shape)
        TARGET_COLOR = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)[0][0]
        print("HSV color:", TARGET_COLOR)


cameraFrame = None
def main():
    global cameraFrame

    y2 = 350
    y3 = 700

    # Create filtered and preview windows
    cv2.namedWindow(PREVIEW_NAME)
    cv2.moveWindow(PREVIEW_NAME, 650, 0)
    cv2.setMouseCallback(PREVIEW_NAME, handleMouse)
    cv2.namedWindow(WINDOW_NAME)
    
    cv2.namedWindow(VISUALIZER)
    cv2.moveWindow(VISUALIZER, 0, y2)
    cv2.namedWindow(VISUALIZER2)
    cv2.moveWindow(VISUALIZER2, 650, y2)

    cv2.namedWindow(TRACKBARS)
    cv2.moveWindow(TRACKBARS, 200, y3)
    
    # Create threshold and gaussian blur trackbars
    cv2.imshow(TRACKBARS, np.zeros([1, 800]))
    cv2.createTrackbar("color threshold", TRACKBARS , 10, 50, nothing)
    cv2.createTrackbar("dark threshold", TRACKBARS , 50, 200, nothing)
    cv2.createTrackbar("blur", TRACKBARS, 5, 20, nothing)
    cv2.createTrackbar("connectivity", TRACKBARS, 20, 50, nothing)
    

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
        
        rval, cameraFrame = vc.read()

        # add contrast to image
        alpha = 1 + cv2.getTrackbarPos("contrast", TRACKBARS) / 30
        brightness = cv2.getTrackbarPos("brightness", TRACKBARS) - 50
       # print(alpha, brightness)
        if alpha != 1:
            cameraFrame = np.clip(cameraFrame * alpha + brightness, 0, 255).astype(np.uint8)
        
        cameraFrame = cv2.resize(cameraFrame, (0,0), fx=0.5, fy=0.5) # reduce image by half
        cameraFrame = applyBlur(cameraFrame) # gaussian blur
        hsvFrame = cv2.cvtColor(cameraFrame, cv2.COLOR_BGR2HSV)
        if not rval:
            break

        filteredFrame = applyColorFilter(hsvFrame, objects, TARGET_COLOR) # Greyscale image with distance to desired color
        objects.updateObjects()
        objects.drawObjects(cameraFrame)

        # Display filtered and preview windows
        cv2.imshow(WINDOW_NAME, filteredFrame)
        mouse.draw(cameraFrame)
        cv2.imshow(PREVIEW_NAME, cameraFrame)
        key = cv2.waitKey(10) # wait 10 ms while at the same time listen for keyboard input. necessary for opencv loop to work
        if key == 27: # exit on ESC
            break

    vc.release()
    cv2.destroyWindow(WINDOW_NAME)

if __name__ == "__main__":
    main()
