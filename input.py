# import the modules
import os
from os import listdir
import cv2
import numpy as np

def crop_roi(img, roi, imgContour, ref_area):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blur the grayscale image
    gray_blurred = cv2.medianBlur(gray, 5)
    # We will use cv2.HoughCircles() function to detect circles in a grayscale image (It does so by using Hough Transform)

    circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 30, 50 , 30, minRadius=0, maxRadius=0)  
    detected_circles = np.uint16(np.around(circles))
    list_r = []
    for i in detected_circles[0, :]:
        list_r.append(i[2])
    cv2.imwrite("roi.jpg",roi)
    X = 0
    Y = 0
    R = 0
    for i in detected_circles[0, :]:
        r = i[2]
        if r == max(list_r):
            X = i[0]
            Y = i[1]
            R = r
            roi = img[Y-R:Y+R, X-R:X+R]

    cv2.imwrite("roi.jpg",roi)

    imgBlur = cv2.GaussianBlur(roi, (7,7), 1) # Step 1: Blur the image
    imgGray = cv2.cvtColor(imgBlur,cv2.COLOR_BGR2GRAY) # Step 2: Convert the blurred image into Black&White image

    threshold1 = 255
    threshold2 = 255
    imgCanny = cv2.Canny(imgGray,threshold1,threshold2)
    kernel = np.ones((5,5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations = 1)

    contours, hierarchy = cv2.findContours(imgDil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        for j in range(len(contours[i])):
            for k in range(len(contours[i][j])):
                contours[i][j][k][0]+=(X-R)
                contours[i][j][k][1]+=(Y-R)

    list_area = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        list_area.append(area)

    max_area = max(list_area)
    scaling_factor = ref_area / max_area

    for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:
                cv2.drawContours(imgContour, cnt, -1, (0,0,255),1)
            print(area*scaling_factor)
    cv2.imwrite("contour.jpg",imgContour)

# get the path to the folder
path = input("Path: ")
ref_area = int(input("Reference Area: "))
# change the directory to the given path
os.chdir(path)
list=[]
for images in os.listdir(path):

    # check if the image ends with png or jpg or jpeg
    if (images.endswith(".png") or images.endswith(".jpg") or images.endswith(".jpeg")):
        list.append(images)

for img_name in range(len(list)):
    if list[img_name] == "ref.jpg":

        orig_img = cv2.imread("ref.jpg")
        roi = orig_img.copy()
        imgContour = orig_img.copy()

        crop_roi(orig_img,roi, imgContour, ref_area)






