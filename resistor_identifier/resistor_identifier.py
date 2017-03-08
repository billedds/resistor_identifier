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


def gamma_correction(res):

    def nothing(x):
        pass

    cv2.namedWindow('Set Gamma correction')
    cv2.createTrackbar('Gamma', 'Set Gamma correction', 0, 40, nothing)

    original_res = res

    while(True):
        gamma = cv2.getTrackbarPos('Gamma', 'Set Gamma correction')
        # need to scale gamma back down to [0, 4] with a step of 0.1
        gamma = gamma / 10.0
        gamma = 0.1 if gamma == 0 else gamma
        inverse_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inverse_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        res = cv2.LUT(original_res, table)
        cv2.imshow('Set Gamma correction', res)
        k = cv2.waitKey(1) & 0xFF

        if k == 27:
            break

    return res

def set_median(res):

    def nothing(x):
        pass

    cv2.namedWindow('Set noise-filtering parameter')
    cv2.createTrackbar('Filter level', 'Set noise-filtering parameter', 1, 6, nothing)

    while(True):
        median_kernel = cv2.getTrackbarPos('Filter level', 'Set noise-filtering parameter')

        # need to make sure the kernel size is odd
        if median_kernel == 0:
            median_kernel = 1
        
        median_kernel =  (median_kernel * 2) - 1
        median_res = cv2.medianBlur(res, median_kernel)
        cv2.imshow('Set noise-filtering parameter', median_res)
        k = cv2.waitKey(1) & 0xFF

        if k == 27:
            break

    return median_res



def set_canny(gray_res):

    def nothing(x):
        pass

    cv2.namedWindow('Set thresholds for Canny edge detection')

    cv2.createTrackbar('Lower threshold', 'Set thresholds for Canny edge detection', 127, 255, nothing)
    cv2.createTrackbar('Upper threshold', 'Set thresholds for Canny edge detection', 127, 255, nothing)

    while(True):
        canny_lower = cv2.getTrackbarPos('Lower threshold', 'Set thresholds for Canny edge detection') 
        canny_upper = cv2.getTrackbarPos('Upper threshold', 'Set thresholds for Canny edge detection')
        canny_res = cv2.Canny(gray_res, canny_lower, canny_upper)
        cv2.imshow('Set thresholds for Canny edge detection', canny_res)
        k = cv2.waitKey(1) & 0xFF

        if k == 27:
            break

    return canny_res




def set_dilate(canny_res):

    def nothing(x):
        pass

    cv2.namedWindow('Set dilate level')

    cv2.createTrackbar('Horizontal level', 'Set dilate level', 1, 21, nothing)
    cv2.createTrackbar('Vertical level', 'Set dilate level', 1, 21, nothing)
    
    while(True):
        horiz_val = cv2.getTrackbarPos('Horizontal level', 'Set dilate level')
        vert_val = cv2.getTrackbarPos('Vertical level', 'Set dilate level')

        # have to make sure kernel values != 0
        if horiz_val == 0:
            horiz_val = 1

        if vert_val == 0:
            vert_val = 1
            
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_val, vert_val))
        dilated_res = cv2.dilate(canny_res, kernel)
        cv2.imshow('Set dilate level', dilated_res)
        k = cv2.waitKey(1) & 0xFF

        if k == 27:
            break

    return dilated_res




def set_close(dilated_res):

    def nothing(x):
        pass

    cv2.namedWindow('Set close level')

    cv2.createTrackbar('Horizontal level', 'Set close level', 1, 25, nothing)
    cv2.createTrackbar('Vertical level', 'Set close level', 1, 25, nothing)
    
    while(True):
        horiz_val = cv2.getTrackbarPos('Horizontal level', 'Set close level')
        vert_val = cv2.getTrackbarPos('Vertical level', 'Set close level')

        # have to make sure kernel values != 0
        if horiz_val == 0:
            horiz_val = 1

        if vert_val == 0:
            vert_val = 1
            
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_val, vert_val))
        closed_res = cv2.morphologyEx(dilated_res, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('Set close level', closed_res)
        k = cv2.waitKey(1) & 0xFF

        if k == 27:
            break

    return closed_res



def set_erode(closed_res):

    def nothing(x):
        pass

    cv2.namedWindow('Set erode level')

    cv2.createTrackbar('Horizontal level', 'Set erode level', 1, 21, nothing)
    cv2.createTrackbar('Vertical level', 'Set erode level', 1, 21, nothing)   
    cv2.createTrackbar('Num. of iterations', 'Set erode level', 1, 5, nothing)
    
    while(True):
        horiz_val = cv2.getTrackbarPos('Horizontal level', 'Set erode level')
        vert_val = cv2.getTrackbarPos('Vertical level', 'Set erode level')
        iterations = cv2.getTrackbarPos('Num. of iterations', 'Set erode level')

        # have to make sure kernel values != 0
        if horiz_val == 0:
            horiz_val = 1

        if vert_val == 0:
            vert_val = 1

        if iterations == 0:
            iterations = 1
                    
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_val, vert_val))
        eroded_res = cv2.erode(closed_res, kernel, iterations)
        cv2.imshow('Set erode level', eroded_res)
        k = cv2.waitKey(1) & 0xFF

        if k == 27:
            break

    return eroded_res



def process_image(input_image):

    res = cv2.imread(input_image)
    rows, columns, dim = res.shape
    # reduce size down to 20%
    scale = 0.2
    
    # 1) resize image
    res = cv2.resize(res, (int(columns*scale), int(rows*scale)))
    rows, columns, dim = res.shape

    # 1.1) gamma correction
    # cv2.imshow('Set Gamma correction', res)
    res = gamma_correction(res)
    
    # 2) filter out noise (e.g. white glare/reflection)
    median_res = set_median(res)

    # 3) convert to grayscale
    gray_res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    # 4) run Canny edge detection
    canny_res = set_canny(gray_res)
  
    # 5) set the structuring element (kernel) separately for each of the following operations
    
    # 6) dilate edged image
    dilated_res = set_dilate(canny_res)
 
    # 7) closing dilated image
    closed_res = set_close(dilated_res)

    # 8) eroding image to get back to normal size
    eroded_res = set_erode(closed_res)
    
    # 9) dilate one more time to get binary filtered image
    filtered_res = set_dilate(eroded_res)

    # 10) original image with the filtered image as a binary mask
    mask = filtered_res
    final_res = cv2.bitwise_and(median_res, median_res, mask = mask) 

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
        ([30, 70, 70], [50, 110, 130]),          # gold
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
	
    
##    # ignore black, gray, & white
##    # TO-DO: filter out white glare in original image
##    non_zero_pix[0] = 0
##    non_zero_pix[8] = 0
##    non_zero_pix[9] = 0
    
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

            
                



    
process_image('resistor_black_background.jpg')


