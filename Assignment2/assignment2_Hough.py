from multiprocessing.sharedctypes import Value
from re import I
from cv2 import line
import numpy as np
import cv2

def sobelx(img):

    sobel = np.zeros([width, height], np.uint8)
    
    sobel_x_kernel =[[1, 0, -1], 
                     [2, 0, -2], 
                     [1, 0, -1]
                    ]
    
    #apply kernel to the image
    for i in range(width):
        for j in range(height):
            sobel[i][j] = applyKernel_bw(sobel_x_kernel, neighborhood_bw(img,3, i, j))
    
    return sobel

def sobely(img):
    sobel = np.zeros([width, height], np.uint8)
    
    #Sobel filter for y axis
    # 1 2 1
    # 0 0 0
    # -1 -2 -1
    sobel_y_kernel =[[1, 2, 1], 
                     [0, 0, 0], 
                     [-1, -2, -1]
                    ]
    
    #apply kernel to the image
    for i in range(width):
        for j in range(height):
            sobel[i][j] = applyKernel_bw(sobel_y_kernel, neighborhood_bw(img,3,i,j))
    
    return sobel

# how to apply kernel for NxN neighbourhood
def applyKernel(kernel, target):
    if (len(kernel) != len(target)) and (len(kernel[0]) != len(target[0])) :
        raise Exception("kernel matrix and target matrix are not the same size")

    res = [[multRgb(kernel[i][j], target[i][j]) for i in range(len(kernel))] for j in range(len(kernel[0]))]
    sum = [0,0,0]
    for i in range(len(res)):
        for j in range(len(res[0])):
            sum = addRgb(res[i][j], sum)

    #floor and ceiling of r,g,b values before returning sum
    for k in range(len(sum)):
        if sum[k] > 255:
            sum[k] = 255
        else:
            if sum[k] < 0:
                sum[k] = 0
        #round to an integer before returning
        sum[k] = int(round(sum[k]))
    return sum

# how to apply kernel for NxN neighbourhood
def applyKernel_bw(kernel, target):

    res = [[kernel[i][j] * target[i][j] for i in range(len(kernel))] for j in range(len(kernel[0]))]
    sum = 0
    for i in range(len(res)):
        for j in range(len(res[0])):
            sum += res[i][j]

    #floor and ceiling of values before returning sum
    if sum > 255:
        sum = 255
    else:
        if sum < 0:
            sum = 0

    #round to an integer before returning
    return int(round(sum))

# simple helper method for multiplying two rgb vectors:
# a1 * b1
# a2 * b2
# a3 * b3
def multRgb(vec1, vec2):
    if(len(vec1) == len(vec2)):
        return [vec1[i]*vec2[i] for i in range(len(vec1))]
    else:
        raise Exception("vectors are not same length")

# simple helper method for adding two rgb vectors:
# a1 + b1
# a2 + b2
# a3 + b3
def addRgb(vec1, vec2):

        return [vec1[i]+vec2[i] for i in range(len(vec1))]

#function for creating a gaussian kernel for part 2B and 2C
# source: https://numpy.org/doc/stable/reference/random/generated/numpy.random.multivariate_normal.html
def createGaussKernel(size=3, sigma=1):
    centre = [int((size-1)/2), int((size-1)/2)] #since gauss kernel is an square, centre is just the middle.
    gauss = np.zeros([size,size,3])

    for i in range(size):
        for j in range(size):
            gauss[i][j] = [ np.e**((-1)*(np.square(i-centre[0]) + np.square(j-centre[1])) / (2*np.square(sigma))) / (sigma*np.sqrt(2*np.pi)),
                            np.e**((-1)*(np.square(i-centre[0]) + np.square(j-centre[1])) / (2*np.square(sigma))) / (sigma*np.sqrt(2*np.pi)) ,
                            np.e**((-1)*(np.square(i-centre[0]) + np.square(j-centre[1])) / (2*np.square(sigma))) / (sigma*np.sqrt(2*np.pi)) ]

    return gauss

def createGaussKernel_bw(size = 3, sigma = 1):
    centre = [int((size-1)/2), int((size-1)/2)] #since gauss kernel is an square, centre is just the middle.
    gauss = np.zeros([size,size])

    for i in range(size):
        for j in range(size):
            gauss[i][j] = np.e**((-1)*(np.square(i-centre[0]) + np.square(j-centre[1])) / (2*np.square(sigma))) / (sigma*np.sqrt(2*np.pi))

    return gauss

# returns a neighbourhood of length size, centerred around index (row, col) 
# pixels outside border of image are set to 0
# only odd number sizes are accepted because they have a center -- 1, 2, 3, 5, 
# source: https://stackoverflow.com/questions/22550302/find-neighbors-in-a-matrix/22550933
def neighborhood(size, row, col):
    radius = int((size-1)/2)

    return [[image[i][j] if  i >= 0 and i < width and j >= 0 and j < height else [0,0,0]
                for j in range(col-1-radius, col+radius)]
                    for i in range(row-1-radius, row+radius)]

def neighborhood_bw(src, size, row, col):
    radius = int((size-1)/2)

    return [[src[i][j] if  i >= 0 and i < width and j >= 0 and j < height else 0
                for j in range(col-1-radius, col+radius)]
                    for i in range(row-1-radius, row+radius)]

def part1a_HoughTransform(edge_map):

    #calculate the gradient images using Sobel
    bw_blur = cv2.GaussianBlur(bw_image, (3,3), 0, 0, cv2.BORDER_DEFAULT)
    sobel_x = sobelx(bw_blur)
    sobel_y = sobely(bw_blur)

    # initialize the hough space using boundaries 
    # theta is between [-90, 90] because range of arctan
    # largest d value is just the entire width and height of image

    #hough = np.zeros([int(np.linalg.norm(np.amax(sobel_x) - np.amin(sobel_y))), 181])
    hough = np.zeros([sum(edge_map.shape), 181])

    for i, value in np.ndenumerate(edge_map): 
   
        x = sobel_x[i[0]][i[1]]
        y = sobel_y[i[0]][i[1]]

        theta = int(np.degrees(np.arctan2(y, x)))
        hough [int(i[0]*np.cos(theta) + i[1]*np.sin(theta))][theta] += 1

    # find all maximum values of hough space and save their indices
    maximums = []
    arrayMax = np.amax(hough)
    for index, value in np.ndenumerate(hough):
        if value >= arrayMax:
            maximums.append([index[0], index[1]])

    return hough, maximums

def part1c_DrawDetectedLines(lines):

    result = image

    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(result,(x1,y1),(x2,y2),(0,0,255),1)

    return result 

#global variables
imagepath= "hough\\hough1.png"
image = cv2.imread(imagepath)
width, height, channels = image.shape
bw_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#variables part 1a
lowerThresh = 120
upperThresh = 180
edge_map = cv2.Canny(image, lowerThresh,upperThresh)

print("Hello welcome to COMP-6341 Assignment 2 by Iymen Abdella, Student ID: 40218280. March 1st 2022!", end="\n")

print("\n --------------------- Part 1 A ------------------------- \n", end="\n")
#hough, maximums = part1a_HoughTransform(edge_map)

print("\n --------------------- Part 1 B ------------------------- \n", end="\n")
#cv2.imshow("Part 1B: Hough transform", hough)
cv2.waitKey(0)

print("\n --------------------- Part 1 C ------------------------- \n", end="\n")
#cv2.imshow("Part 1C: Detect lines on hough transform", part1c_DrawDetectedLines(maximums))
# Desired result: 
#cv2.imshow("Part 1C: Detect lines on hough transform", part1c_DrawDetectedLines(cv2.HoughLines(edge_map, 1, np.pi / 180, 150, None, 0, 0)))
cv2.waitKey(0)
