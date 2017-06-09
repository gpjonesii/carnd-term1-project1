#!/usr/bin/env python
import os
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=3):

    ysize = img.shape[0]
    xsize = img.shape[1]

    leftlines = []
    rightlines = []

    slope = lambda x1,y1,x2,y2: (y1-y2)/(x1-x2)
    getx  = lambda  b,y1,x2,y2: ((y1-y2)/b)+x2

    for line in lines:
        for x1,y1,x2,y2 in line:
            if slope(x1,y1,x2,y2) < 0:
                leftlines.append(line)
            elif slope(x1,y1,x2,y2) > 0:
                rightlines.append(line)

    rmean = np.array(rightlines).mean(axis=0).astype(int)
    lmean = np.array(leftlines).mean(axis=0).astype(int)

    #draw left lane line in blue
    if type(lmean) is np.ndarray:
        x1,y1,x2,y2 = lmean[0]
        b = slope(x1,y1,x2,y2)
        x1 = int(getx(b,ysize-1,x2,y2))
        x2 = int(getx(b,325,x1,ysize-1))
        cv2.line(img, (x1, ysize-1), (x2, 325), [0, 0, 255], thickness)

    #draw right lane line in red
    if type(rmean) is np.ndarray:
        x1,y1,x2,y2 = rmean[0]
        b = slope(x1,y1,x2,y2)
        x1 = int(getx(b,ysize-1,x2,y2))
        x2 = int(getx(b,325,x1,ysize-1))
        cv2.line(img, (x1, ysize-1), (x2, 325), [255, 0, 0], thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def write_tmp_image(img,filename):
    directory = os.path.dirname("image_tmp/")
    if not os.path.exists(directory):
        os.makedirs(directory)
    newfile = directory + "/" + filename
    cv2.imwrite(newfile,img)


for item in os.scandir("test_images/"):
    file = item.path
    image = mpimg.imread(file)
    ysize = image.shape[0]
    xsize = image.shape[1]
    write_tmp_image(image, "orig-" + os.path.basename(file))

    grey_image = grayscale(image)
    write_tmp_image(grey_image, "gray-" + os.path.basename(file))

    blur_image = gaussian_blur(grey_image, kernel_size=5)
    write_tmp_image(blur_image,"blur-" + os.path.basename(file))

    edge_image = canny(blur_image,low_threshold=30,high_threshold=150)
    write_tmp_image(edge_image,"edge-" + os.path.basename(file))

    vertices = np.array([[(xsize/2-35,325),(xsize/2+35,325), (900,ysize-1), (75,ysize-1)]], dtype=np.int32)
    masked_edges_image = region_of_interest(edge_image,vertices)
    write_tmp_image(masked_edges_image,"masked-edges-" + os.path.basename(file))

    lines_image = np.copy(image)*0
    lines_image = hough_lines(masked_edges_image, rho=1, theta=np.pi / 180, threshold=15, min_line_len=30, max_line_gap=15)
    write_tmp_image(lines_image,"lines-image-" + os.path.basename(file))

    weighted_image = weighted_img(lines_image,image)
    write_tmp_image(weighted_image,"weighted-image-" + os.path.basename(file))
