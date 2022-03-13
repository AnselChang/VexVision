import cv2, time, math
import numpy as np


WINDOW_NAME = "A"
WINDOW_NAME2 = "B"

def main():
   
    cv2.namedWindow(WINDOW_NAME)
    
    vc = cv2.VideoCapture(0)
    print("Capture started")

    while True:
        
        rval, cameraFrame = vc.read()
        if not rval:
            break

        
        cameraFrame = cv2.resize(cameraFrame, (0,0), fx=0.5, fy=0.5) # reduce image by half
        cameraFrame = cv2.GaussianBlur(cameraFrame, (5, 5), 0)
        img_gray = cv2.cvtColor(cameraFrame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(img_gray,100,200)
        gray_BGR = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        lines = cv2.HoughLinesP(edges, 1, math.pi / 180, 70, maxLineGap  = 100)
        print(lines)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(cameraFrame, (x1,y1), (x2,y2), [0,255,0], 3)
        

        # Display filtered and preview windows
        cv2.imshow(WINDOW_NAME, cameraFrame)
        cv2.imshow(WINDOW_NAME2, gray_BGR)
        key = cv2.waitKey(10) # wait 10 ms while at the same time listen for keyboard input. necessary for opencv loop to work
        if key == 27: # exit on ESC
            break

    vc.release()
    cv2.destroyWindow(WINDOW_NAME)


if __name__ == "__main__":
    main()
