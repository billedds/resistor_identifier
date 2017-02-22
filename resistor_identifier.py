#!/usr/bin/env python
import numpy as np
import math
import cv2
from time import sleep

# global variables
non_zero_pix = []
top_vals = []
color = []
top_colors_array = []
cx = []
cy = []





def process_image(input_image):

    res = cv2.imread(input_image)

    res = cv2.resize(res, (640, 480))
    # structuring element, similar to se = ones(5, 5) in MatLab
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    # filter out noise (e.g. white glare/reflection)
    # res = cv2.GaussianBlur(res, (3, 3), 0)
    cv2.waitKey(0)
    gray_res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    canny_res = cv2.Canny(gray_res, 100, 200)

    # dilate edged image
    dilated_res = cv2.dilate(canny_res, kernel)
    # closing dilated image
    closed_res = cv2.morphologyEx(dilated_res, cv2.MORPH_CLOSE, kernel)
    # eroding image to get back to normal size
    eroded_res = cv2.erode(closed_res, kernel, iterations = 3)
    # binary filtered image
    filtered_res = cv2.dilate(eroded_res, kernel)
    # original image with the filtered image as a binary mask
    mask = filtered_res
    final_res = cv2.bitwise_and(res, res, mask = mask) 

##    cv2.imshow('Original Resistor, 0', res)
##    cv2.waitKey(0)
##    cv2.imshow('Grayscale, 1', gray_res)
##    cv2.waitKey(0)
##    cv2.imshow('Canny, 2', canny_res)
##    cv2.waitKey(0)
##    cv2.imshow('Dilated, 3', dilated_res)
##    cv2.waitKey(0)
##    cv2.imshow('Closed, 4', closed_res)
##    cv2.waitKey(0)
##    cv2.imshow('Eroded, 5', eroded_res)
##    cv2.waitKey(0)
##    cv2.imshow('Filtered, 6', filtered_res)
##    cv2.waitKey(0)
##    cv2.imshow('Final, 7', final_res)
##    cv2.waitKey(0)
    
    cv2.imwrite('final_filtered_res.jpg', final_res)
    
    identify_colors(final_res)


    

def identify_colors(processed_image):

    # openCV represents images as NumPy arrays in reverse-color-order
    # each entry in the list below is a tuple of two values, lower and upper limits
    # e.g. red: 100 <= R <= 200, 15 <= G <= 56, 17 <= B <= 50        
    color_bounds = [
        ([0, 0, 0], [32, 32, 32]),               # black
        ([50, 70, 80], [80, 100, 120]),          # brown 
        ([17, 15, 100], [50, 56, 200]),          # red
        ([0, 75, 200], [51, 153, 255]),          # orange
        ([0, 146, 190], [62, 210, 250]),         # yellow
        ([110, 90, 70], [140, 125, 120]),        # green
        ([86, 31, 4], [220, 88, 50]),            # blue
        ([51, 30, 100], [255, 90, 170]),         # violet
        ([120, 120, 120], [160, 160, 160]),      # gray
        ([161, 161, 161], [255, 255, 255]),      # white
        ([30, 70, 70], [50, 110, 130]),         # gold
        ([180, 180, 180], [195, 195, 195])]      # silver
        
    global non_zero_pix
    global top_vals
    global color
    global top_colors_array
    global cx, cy
    count = 0
    current_num_pix = 0
    past_num_pix = 0
         
    for (lower, upper) in color_bounds:
            					            
        if count == 0: 
            color.append('black')
        elif count == 1:
            color.append('brown')
        elif count == 2:
            color.append('red')
        elif count == 3:
            color.append('orange')
        elif count == 4:
            color.append('yellow')
        elif count == 5:
            color.append('green')
        elif count == 6:
            color.append('blue')
        elif count == 7:
            color.append('violet')
        elif count == 8:
            color.append('gray')
        elif count == 9:
            color.append('white')
        elif count == 10:
            color.append('gold')
        elif count == 11:
            color.append('silver')
        
        # create NumPy arrays from bounds
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")

        # find colors within specified bounds and apply the mask
        mask = cv2.inRange(processed_image, lower, upper) 
        identified_colors = cv2.bitwise_and(processed_image, processed_image, mask = mask)       

        # count non-zero pixels in each colored image
        identified_colors_gray = cv2.cvtColor(identified_colors, cv2.COLOR_BGR2GRAY)
        non_zero_pix.append(cv2.countNonZero(identified_colors_gray))

        # compute center of mass
        if non_zero_pix[count] > 0:
            ret, identified_colors_binary = cv2.threshold(identified_colors_gray, 0, 255, cv2.THRESH_BINARY)
            moments = cv2.moments(identified_colors_binary)
            cx.append(int(moments['m10'] / moments['m00']))
            cy.append(int(moments['m01'] / moments['m00']))
        else:
            cx.append(0)
            cy.append(0)

        # show the images
        print('current color: '), color[count]
        print('pixel count '), non_zero_pix[count]
        print('center column of pixels: '), cx[count]
        print
        cv2.imshow('Identified Colors Images', np.hstack([processed_image, identified_colors]))
        cv2.waitKey(0)
    	count += 1
	
    
    # ignore black, gray, & white
    # TO-DO: filter out white glare in original image
    non_zero_pix[0] = 0
    non_zero_pix[8] = 0
    non_zero_pix[9] = 0
    
    # sort out top 3 colors found
    top_colors_array = sorted(zip(non_zero_pix, color, cx), reverse = True)[:4]
    print('top colors found: '), top_colors_array
    print
    
    calculate_value(top_colors_array)




def calculate_value(color_array):

    # this function will calculate the value of the resistor in the image
    # the value calculated will be based on the input array, which has [(# of non-zero pixels, associated color)]
    # Note: gold represents 0.1 Ohm multiplier, +/- 5%; silver is 0.01 Ohm, +/-10%; neither hold an actual value
    # ans = sorted(zip(color_array[:][:]), reverse = False)[:4]
    cx_ordered = []
    color_ordered = []
    non_zero_pix_ordered = []
    ordered_array = []
    band_color = []
    band_val = [0.0, 0.0, 0.0, 0.0]
    resistor_val = 0.0
    
    for k in range(0, 4):
        non_zero_pix_ordered.append(color_array[k][0])
        color_ordered.append(color_array[k][1])
        cx_ordered.append(color_array[k][2])

    ordered_array = sorted(zip(cx_ordered, color_ordered, non_zero_pix_ordered), reverse = False)

    print color_ordered

    for l in range(0, 4):
        
        if color_ordered[l] == 'black':
            if l == 2:
                band_val[l] = 1
            else:
                band_val[l] = 0
            
        elif color_ordered[l] == 'brown':
            if l == 2:
                band_val[l] = 10
            else:
                band_val[l] = 1
            
        elif color_ordered[l] == 'red':
            if l == 2:
                band_val[l] = 100
            else:
                band_val[l] = 2
            
        elif color_ordered[l] == 'orange':
            if l == 2:
                band_val[l] = 1000
            else:
                band_val[l] = 3
            
        elif color_ordered[l] == 'yellow':
            if l == 2:
                band_val[l] = 10000
            else:
                band_val[l] = 4
            
        elif color_ordered[l] == 'green':
            if l == 2:
                band_val[l] = 100000
            else:
                band_val[l] = 5
            
        elif color_ordered[l] == 'blue':
            if l == 2:
                band_val[l] = 1000000
            else:
                band_val[l] = 6
            
        elif color_ordered[l] == 'violet':
            if l == 2:
                band_val[l] = 10000000
            else:
                band_val[l] = 7
            
        elif color_ordered[l] == 'gray':
            band_val[l] = 8
            
        elif color_ordered[l] == 'white':
            band_val[l] = 9
            
        elif color_ordered[l] == 'gold':
            if l == 2:
                band_val[l] = 0.1
            elif l == 3:
                band_val[l] = 5.0
                
        elif color_ordered[l] == 'silver':
            if l == 2:
                band_val[l] = 0.01
            elif l == 3:
                band_val[l] = 10.0                

    multiplier = band_val[2]
    tolerance = band_val[3]
    resistor_val = ((10.0 * band_val[0]) + band_val[1]) * multiplier

    print('ordered left-to-right: '), ordered_array
    print
    print
    print('This is a  '), resistor_val, (' Ohm resistor')
    print('The tolerance is +/- '), tolerance, (' %')

            
                



    
process_image('resistor_1.jpg')


