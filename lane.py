import cv2
import numpy as np 
import matplotlib.pyplot as plt

def canny(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    canny = cv2.Canny(blur,50,150)
    return canny

def roi(image):
    height = image.shape[0]
    polygons = np.array([
        [(200,height),(1100,height),(550,250)]
        ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,polygons,255)
    masked = cv2.bitwise_and(image,mask)
    return masked

def display_lines(image , lines ):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1,y1,x2,y2  in lines:
            #x1,y1,x2,y2 = line.reshape(4)
            cv2.line(line_image , (x1,y1),(x2,y2),(255,0,0),10)
    
    return line_image


def make_cord(image , line_parameters):
    slope ,intercept  = line_parameters
    y1  = image.shape[0]
    y2  = int(y1*(3/5))
    x1  = int((y1 - intercept)/slope)
    x2  = int((y2 - intercept)/slope)

    return np.array([x1,y1,x2,y2])


def averageslp(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameter = np.polyfit((x1,x2),(y1,y2),1)
        slope  =  parameter[0]
        intercept = parameter[1]
        if slope <0:
            left_fit.append((slope,intercept))
        else :
            right_fit.append((slope,intercept))
    left_fit_avg  = np.average(left_fit,axis  = 0)
    right_fit_avg  = np.average(right_fit,axis  = 0)
    left_line = make_cord(image , left_fit_avg)
    right_line = make_cord(image , right_fit_avg)
    return np.array([left_line,right_line])


image = cv2.imread('PATH TO IMAGE')
lane_image = np.copy(image)
canny = canny(lane_image)
masked = roi(canny)
lines = cv2.HoughLinesP(masked,2,(np.pi/180),100,np.array([]),minLineLength = 40,maxLineGap = 5)

average_line =  averageslp(lane_image,lines)

line_image = display_lines(lane_image , average_line)
crop  = cv2.addWeighted(lane_image,0.8, line_image,1,1)

cv2.imshow('result',crop)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''
cap = cv2.VideoCapture('/home/satya/Downloads/test2.mp4')

while True:
    _  , frame = cap.read()
    lane_image = np.copy(frame)
    canny1 = canny(lane_image)
    masked = roi(canny1)
    lines = cv2.HoughLinesP(masked,2,(np.pi/180),100,np.array([]),minLineLength = 40,maxLineGap = 5)

    average_line =  averageslp(lane_image,lines)

    line_image = display_lines(lane_image , average_line)
    crop  = cv2.addWeighted(lane_image,0.8, line_image,1,1)

    cv2.imshow('result',crop)
    if cv2.waitKey(1) & 0xFF  == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

'''




