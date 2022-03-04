#import necessary libraries
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

def part2a_HarrisCornerDetection(thresh, k=0.05):

    res = np.zeros([width, height])
    
    #compute gradients after smooothing
    bw_blur = cv2.GaussianBlur(bw_image, (3,3), 0, 0, cv2.BORDER_DEFAULT)
    sobel_x = sobelx(bw_blur)
    sobel_y = sobely(bw_blur)

    #compute multiplication and then smooth with gaussian
    sobel_xx = cv2.GaussianBlur(sobel_x*sobel_x, (3,3), 0, 0, cv2.BORDER_DEFAULT) 
    sobel_yy = cv2.GaussianBlur(sobel_y*sobel_y, (3,3), 0, 0, cv2.BORDER_DEFAULT) 
    sobel_xy = cv2.GaussianBlur(sobel_x*sobel_y, (3,3), 0, 0, cv2.BORDER_DEFAULT) 

    #compute Harris matrix for 3x3 neighbourhood
    for index, value in np.ndenumerate(image):
        row = index[0]
        col = index[1]

        #compute sums of components using list comprehension 
        axx = neighborhood_bw(sobel_xx,3,row,col)
        axx = sum(sum(axx,[]))

        axy = neighborhood_bw(sobel_xy,3,row,col)
        axy = sum(sum(axy,[]))

        ayy = neighborhood_bw(sobel_yy,3,row,col)     
        ayy = sum(sum(ayy,[]))

        #combine to form the Harris matrix and compute the corner response. 
        harris = np.array([[axx, axy], [axy, ayy]])
        corner_response = np.linalg.det(harris) - k*np.square((np.matrix.trace(harris)))

        #threshold and save
        if (corner_response > thresh):
            res[row][col] = corner_response
    
    #normalize before returning result:
    
    res = cv2.normalize(res, )
    return sobel_x, sobel_y, sobel_xy, res

def part2d_cv_DrawKeyPoints():

    return 

def part3a_MatchDescriptors():
    return 

def part3b_DisplayKeypoints():
    return 

def part3c_DisplayMatchedKeypoints():
    return 

def part4_ApplySobel():
    return 

def part5a_DescribeFeatures_ContrastInvariant(): 
    return 

def part5b_AdaptiveNon_MaxSuppression(): 
    return 

#global variables
imagepath= "test-field.jpg"
image = cv2.imread(imagepath)
width, height, channels = image.shape
bw_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#variables part 1a
lowerThresh = 120
upperThresh = 180
edge_map = cv2.Canny(image, lowerThresh,upperThresh)

print("Hello welcome to COMP-6341 Assignment 2 by Iymen Abdella, Student ID: 40218280. March 1st 2022!", end="\n")

print("\n --------------------- Part 2 A ------------------------- \n", end="\n")
Ix, Iy, Ixy, res =  part2a_HarrisCornerDetection(thresh=1000)
cv2.waitKey(0)

print("\n --------------------- Part 2 B ------------------------- \n", end="\n")
cv2.imshow("part 2B: Ix", Ix)
cv2.imshow("part 2B: Iy", Iy)
cv2.imshow("part 2B: Ixy", Ixy)
cv2.waitKey(0)

print("\n --------------------- Part 2 C ------------------------- \n", end="\n")
cv2.imshow(f"part 2C: results of corner strength response", res)
cv2.waitKey(0)

print("\n --------------------- Part 2 D ------------------------- \n", end="\n")
output = image.copy()
cv2.drawKeypoints(output, res, None, color=(0,255,0), flags=0)
cv2.imshow(f"part 2D: using openCV draw key points", output)
cv2.waitKey(0)

# print("\n --------------------- Part 3 A ------------------------- \n", end="\n")
# cv2.imshow("Part 3A: SIFT-like descriptors and matches ", part3a_MatchDescriptors())
# cv2.waitKey(0)

# print("\n --------------------- Part 3 B ------------------------- \n", end="\n")
# cv2.imshow("Part 3B: Key points", part3b_DisplayKeypoints())
# cv2.waitKey()

# print("\n --------------------- Part 3 C ------------------------- \n", end="\n")
# cv2.imshow("Part 3C: Match key pionts", part3c_DisplayMatchedKeypoints())
# cv2.waitKey()

# print("\n --------------------- Part 4 ------------------------- \n", end="\n")
# cv2.imshow("Part 4: Pytorch Sobel Operator", part4_ApplySobel())
# cv2.waitKey()

# print("\n --------------------- Part 5 A ------------------------- \n", end="\n")
# cv2.imshow("Part 5A: Contrast invariant feature descriptor", part5a_DescribeFeatures_ContrastInvariant())
# cv2.waitKey()

# print("\n --------------------- Part 5 B ------------------------- \n", end="\n")
# cv2.imshow("Part 5B : adaptive non maximum suppression", part5b_AdaptiveNon_MaxSuppression())
# cv2.waitKey()

# #close all remaining opened windows
# cv2.destroyAllWindows()