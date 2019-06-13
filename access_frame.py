import numpy as np
from PIL import ImageGrab
from numpy import ones,vstack
from numpy.linalg import lstsq
import matplotlib.pyplot as plt
import cv2
import time
from directkeys import PressKey, W, A, S, D
from statistics import mean


 
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = 5
    #blur = cv2.GaussianBlur(gray,(5,5),0)
    canny = cv2.Canny(gray, 200, 300)
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
        for line  in lines:
            if line is not None:
                x1,y1,x2,y2 = line.reshape(4)
                cv2.line(line_image , (x1,y1),(x2,y2),(255,0,0),10)
    
    return line_image


def make_cord(image , line_parameters):
    try:
       slope, intercept = line_parameters
    except TypeError:
       slope, intercept = 0,0
    
    if slope != 0:
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
        if slope < 0: # y is reversed in image
           left_fit.append((slope, intercept))
        else:
           right_fit.append((slope, intercept))
    
    left_fit_average  = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line  = make_cord(image, left_fit_average)
    right_line = make_cord(image, right_fit_average)
    averaged_lines = [left_line, right_line]
    return averaged_lines



def main():
    #for i in list(range(4))[::-1]:
       # print(i+1)
       # time.sleep(1)
        
    #while True:
        #PressKey(W)
##        printscreen_pil =  ImageGrab.grab(bbox=(0,40,640,480))
        img = cv2.imread('C:\Users\SATYA\Downloads\lane1.')
        frame =np.asarray(printscreen_pil)
        canny = preprocess(frame)
        
        #open_im = cv2.morphologyEx(canny,cv2.MORPH_OPEN,(5,5))
        masked = roi(canny)
        lines = cv2.HoughLinesP(masked,2,(np.pi/180),100,np.array([]),minLineLength = 40 , maxLineGap = 5)
        average_line = averageslp(frame , lines)
        line_img = display_lines(frame,average_line)
        crop = cv2.bitwise_or(frame,line_img)
##        crop  = cv2.addweighted(lane_image, 0.8,line_image, 1,1)

        cv2.imshow("result", crop)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
main()
