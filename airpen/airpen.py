# import the necessary packages
from collections import deque
import numpy as np
import argparse
import imutils
import cv2
import sys
import uinput

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
    help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=32,
    help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "green"
# ball in the HSV color space
greenLower = (105, 70, 70)
greenUpper = (125, 255, 255)
# red = np.uint8([[[60,62,93]]])
# print(cv2.cvtColor(red,cv2.COLOR_BGR2HSV))

# initialize the list of tracked points, the frame counter,
# and the coordinate deltas
pts = deque(maxlen=args["buffer"])
counter = 0
(dX, dY) = (0, 0)
direction = ""
SAFE_ZONE = 64
angles = ((np.pi/4) * np.array((1,3,5,7)))
t1 = np.pi/4
t2 = 3*t1
pathString = []
lastQ = None
quadrant = None

device = uinput.Device([
    uinput.KEY_A,
    uinput.KEY_B,
    uinput.KEY_C,
    uinput.KEY_D,
    uinput.KEY_E,
    uinput.KEY_F,
    uinput.KEY_G,
    uinput.KEY_H,
    uinput.KEY_I,
    uinput.KEY_J,
    uinput.KEY_K,
    uinput.KEY_L,
    uinput.KEY_M,
    uinput.KEY_N,
    uinput.KEY_O,
    uinput.KEY_P,
    uinput.KEY_Q,
    uinput.KEY_R,
    uinput.KEY_S,
    uinput.KEY_T,
    uinput.KEY_U,
    uinput.KEY_V,
    uinput.KEY_W,
    uinput.KEY_X,
    uinput.KEY_Y,
    uinput.KEY_Z,
    uinput.KEY_BACKSPACE,
    uinput.KEY_SPACE,
    uinput.KEY_MINUS,
    uinput.KEY_DOT,
    uinput.KEY_COMMA,
    uinput.KEY_QUESTION,
])

with open("words.txt") as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content]

substring = ''
possibleWord = ''

masterList = {
'1': [' '], '4':['COMPLETE'], '3':['BACKSPACE'],'2':['COMPLETE'],
'14': ['a','r','x','-'], '12': ['n','m','f','?'] ,
'23': ['e','l','k','#'], '21':['o','u','v','w'],
'34': ['t','c','z','.'], '32':['i','h','j',','],
'41': ['y','b','p','q'], '43':['s','d','g','/']
}

masterList2 = {
'1': [uinput.KEY_SPACE], '4':[''], '3':[uinput.KEY_BACKSPACE],'2':[''],
'14': [uinput.KEY_A, uinput.KEY_R, uinput.KEY_X, uinput.KEY_MINUS], '12': [ uinput.KEY_N, uinput.KEY_M, uinput.KEY_F, uinput.KEY_QUESTION] ,
'23': [ uinput.KEY_E, uinput.KEY_L, uinput.KEY_K, '#'], '21':[ uinput.KEY_O, uinput.KEY_U, uinput.KEY_V, uinput.KEY_W],
'34': [ uinput.KEY_T, uinput.KEY_C, uinput.KEY_Z, uinput.KEY_DOT], '32':[ uinput.KEY_I, uinput.KEY_H, uinput.KEY_J,uinput.KEY_COMMA],
'41': [ uinput.KEY_Y, uinput.KEY_B, uinput.KEY_P, uinput.KEY_Q], '43':[ uinput.KEY_S, uinput.KEY_D, uinput.KEY_G,'/']
}

started = False
nonprint = False
# if a video path was not supplied, grab the reference
# to the webcamd
if not args.get("video", False):
    camera = cv2.VideoCapture(1)

# otherwise, grab a reference to the video file
else:
    camera = cv2.VideoCapture(args["video"])

def printLetters():
    for k,v in masterList.items():
        if k == '14':
            for i,j in enumerate(v):
                point = (int(middle[0] + 16 + (np.cos(-t1)*(64+48*i) )), \
                        int(middle[1] + (np.sin(-t1)*(64+48*i))))
                cv2.putText(frame, j, point, cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 3)
                cv2.putText(frame, j, point, cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
        elif k == '21':
            for i,j in enumerate(v):
                point = (int(middle[0] - 16 + (np.cos(t1)*(64+48*i) )), \
                        int(middle[1] +32+  (np.sin(t1)*(64+48*i))))
                cv2.putText(frame, j, point, cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 3)
                cv2.putText(frame, j, point, cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
        elif k == '41':
            for i,j in enumerate(v):
                point = (int(middle[0] + (np.cos(-t1)*(64+48*i) )), \
                        int(middle[1] - 32 + (np.sin(-t1)*(64+48*i))))
                cv2.putText(frame, j, point, cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 3)
                cv2.putText(frame, j, point, cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
        elif k == '12':
            for i,j in enumerate(v):
                point = (int(middle[0] + 32 + (np.cos(t1)*(64+48*i) )), \
                        int(middle[1] +  (np.sin(t1)*(64+48*i))))
                cv2.putText(frame, j, point, cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 3)
                cv2.putText(frame, j, point, cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
        elif k == '32':
            for i,j in enumerate(v):
                point = (int(middle[0] - 32 + (np.cos(t2)*(64+48*i) )), \
                        int(middle[1] + 16+ (np.sin(t2)*(64+48*i))))
                cv2.putText(frame, j, point, cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 3)
                cv2.putText(frame, j, point, cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
        elif k == '23':
            for i,j in enumerate(v):
                point = (int(middle[0] + (np.cos(t2)*(64+48*i) )), \
                        int(middle[1] + 32 +  (np.sin(t2)*(64+48*i))))
                cv2.putText(frame, j, point, cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 3)
                cv2.putText(frame, j, point, cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
        elif k == '43':
            for i,j in enumerate(v):
                point = (int(middle[0] + (np.cos(-t2)*(64+48*i) )), \
                        int(middle[1] - 16 + (np.sin(-t2)*(64+48*i))))
                cv2.putText(frame, j, point, cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 3)
                cv2.putText(frame, j, point, cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
        elif k == '34':
            for i,j in enumerate(v):
                point = (int(middle[0] - 32 + (np.cos(-t2)*(64+48*i) )), \
                        int(middle[1] + 16 +  (np.sin(-t2)*(64+48*i))))
                cv2.putText(frame, j, point, cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 3)
                cv2.putText(frame, j, point, cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)

def getPoint(radians, radius):
    return ((int(middle[0] + (np.cos(radians)*radius)), \
            int(middle[1] + (np.sin(radians)*radius))))

def getAngle(a,b):
    return np.arctan2(b[1]-a[1],b[0]-a[0])

def getQuad():
    if center is None:
        return None
    else:
        if (np.sqrt(np.sum((center-middle)**2))) <= SAFE_ZONE:
            return '0'
        else:
            theta = getAngle(middle, center)
            if theta > t2 or theta < -(t2):
                return '1'
            elif theta > t1 and theta < t2:
                return '2'
            elif theta > 0 or theta > -(t1):
                return '3'
            elif theta < -(t1) and theta > -(t2):
                return '4'

def follow():
    global substring
    global lastQ
    global pathString
    if pathString != []:
        lastQ = pathString[-1]

    if quadrant == '0' and lastQ != '0' and pathString != []:
        writeChar()

    if (lastQ != quadrant) and (quadrant is not None) and (quadrant != '0'):
        if len(pathString) >= 2 and  quadrant == pathString[-2]:
            pathString = pathString[:-2]
        pathString.append(quadrant)


def writeChar():
    global pathString
    global masterList2
    global masterList
    global lastQ
    global device
    global substring
    global nonprint
    global possibleWord
    # print('path: ',''.join(pathString))
    i = ''.join(pathString[0:2])
    j = len(pathString[2:])
    # print("ij :", i, j)
    try:
        if masterList[i][j] == " ":
            substring = ''
            possibleWord = ''
        elif masterList[i][j] =="BACKSPACE":
            substring = substring[:-1]
        elif masterList[i][j] =="COMPLETE":
            completeWord()
            nonprint = True
        else:
            substring += masterList[i][j]

        if not nonprint:
            device.emit_click(masterList2[i][j])
        nonprint = False

    except Exception as e:
        print("error:",e)

    try:
        possibleWord = next(s for s in content if s.startswith(substring))
    except Exception as e:
        possibleWord = ''

    lastQ = '0'
    pathString = []

def completeWord():
    global substring
    global possibleWord
    global masterList2
    global masterList
    comp = possibleWord[len(substring):]
    for letter in comp:
        spot = [k for k,v in masterList.items() if letter in v][0]
        index = masterList[spot].index(letter)
        device.emit_click(masterList2[spot][index])
    device.emit_click(uinput.KEY_SPACE)
    substring = ''
    possibleWord = ''

# keep looping
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()

    if not started:
        width = np.size(frame, 1) #here is why you need numpy!  (remember to "import numpy as np")
        height = np.size(frame, 0)
        middle = np.array((int(width/2), int(height/2)))
        started = True

    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if args.get("video") and not grabbed:
        break

    # resize the frame, blur it, and convert it to the HSV
    # color space
    # frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None


    # put center circle
    GRAY = (64,64,64)
    cv2.circle(frame, tuple(middle), SAFE_ZONE, GRAY, 2)

    # add cross
    for r in angles:
        cv2.line(frame, getPoint(r,SAFE_ZONE), getPoint(r, 3 * SAFE_ZONE), GRAY, 2)


    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = np.array((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))

        # only proceed if the radius meets a minimum size
        if radius > 10:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),
                (0, 255, 255), 2)
            cv2.circle(frame, tuple(center), 5, (0, 0, 255), -1)
            pts.appendleft(center)


    # loop over the set of tracked points
    for i in np.arange(1, len(pts)):
        # if either of the tracked points are None, ignore
        # them
        if pts[i - 1] is None or pts[i] is None:
            continue

        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, tuple(pts[i - 1]), tuple(pts[i]), (0, 0, 255), thickness)

    # mirror video effect
    frame = cv2.flip(frame,1)

    quadrant = getQuad()
    follow()
    # if quadrant == '0':
        # writeChar()
    # print("direction", direction)
    # show the movement deltas and the direction of movement on
    # the frame

    cv2.putText(frame, possibleWord, (middle[0], height - 30), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (0, 0, 255), 3)

    printLetters()


    # show the frame to our screen and increment the frame counter
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    counter += 1

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()

