import cv2
import imutils
import numpy as np
from astar import path_planning
def top_view(points):
    # print("entered tpv")
    # initialize image
    blank_image = np.zeros((500,500,3), np.uint8)
    background = blank_image
    background_height, background_width, channels = background.shape
    car = cv2.imread('car.jpeg')
    car = imutils.resize(car, width=20)

    '''cv2.imshow("car", car)
    cv2.waitKey(1)'''


    # is point pe image aana chahiye
    # external_points_set = [(400,150 ), (350, 300), (70, 200), (210, 110)]
    external_points_set = [(250,250 ),(350,100 ), (70, 200),(200,400)]
    # external_points_set = points

    # print(f"external_points_set: {external_points_set}")

    for external_point in external_points_set.copy():
        height, width, channel = car.shape
        # print(f"height {height}, width {width}")
        # external point should be center but due to opencv its top left corner

        x, y = external_point
        # opencv point is shift of origin
        opencv_point = (x-int(height/2), y-int(width/2))
        x, y = opencv_point

        # error checking for out of region points
        if x <= 0 or y <= 0:
            # print(f"point {external_point} is not displayed on top view beacause its out of region")
            external_points_set.remove(external_point)
            continue
        # print(f"opencv points {x} and  {y}")

        # I want to put logo on opencv point, So I create a ROI
        rows,cols,channels = car.shape
        # print(f"rows {rows}, columns {cols}")

        # error checking for out of region points
        if rows+x >= background_height or cols+y >= background_width:
            # print(f"point {external_point} is not displayed on top view beacause its out of region")
            external_points_set.remove(external_point)
            continue

        roi = background[x:rows+x, y:cols+y ]

        # Now create a mask of logo and create its inverse mask also
        img2gray = cv2.cvtColor(car,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # Now black-out the area of logo in ROI
        img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

        # Take only region of logo from logo image.
        img2_fg = cv2.bitwise_and(car,car,mask = mask)


        # Put logo in ROI and modify the main image
        dst = cv2.add(img1_bg,img2_fg)
        background[x:rows+x, y:cols+y ] = dst



    path=path_planning(external_points_set)
    if path!=None:        
        for i in path:
            background= cv2.circle(background,(i[1],i[0]), radius=0, color=(188, 145, 42), thickness=-1)
    else:
        pass

    
    cv2.imshow('top_view',background)
    cv2.waitKey(1)
    # print(path)
    return